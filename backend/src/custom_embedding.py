"""
Custom Embedding Service - Sử dụng model riêng thay vì OpenAI
"""

import logging
import hashlib
import os
import re
from functools import lru_cache
from collections import Counter
from typing import List, Union

import requests

logger = logging.getLogger(__name__)

# Configuration
CUSTOM_EMBEDDING_API_URL = os.environ.get(
    "CUSTOM_EMBEDDING_API_URL", "http://localhost:5000"
)
CUSTOM_EMBEDDING_ENABLED = (
    os.environ.get("CUSTOM_EMBEDDING_ENABLED", "true").lower() == "true"
)

# Fallback to OpenAI if custom embedding fails
USE_OPENAI_FALLBACK = os.environ.get("USE_OPENAI_FALLBACK", "true").lower() == "true"
USE_LOCAL_EMBEDDING_FALLBACK = (
    os.environ.get("USE_LOCAL_EMBEDDING_FALLBACK", "true").lower() == "true"
)
LOCAL_EMBEDDING_DIMENSION = int(os.environ.get("LOCAL_EMBEDDING_DIMENSION", "1024"))


class CustomEmbeddingService:
    """Service để generate embeddings từ custom model"""

    def __init__(self, api_url: str = CUSTOM_EMBEDDING_API_URL):
        self.api_url = api_url.rstrip("/")
        self.embedding_endpoint = f"{self.api_url}/embed"
        self.health_endpoint = f"{self.api_url}/health"
        self.timeout = 30  # seconds
        self.service_available = CUSTOM_EMBEDDING_ENABLED
        if self.service_available:
            # Check service health on init
            self.service_available = self._check_health()

    def _bag_of_words_embedding(self, text: str) -> List[float]:
        """Generate a deterministic local fallback embedding (bag-of-words)."""
        normalized = text.lower()
        tokens = re.findall(r"[\wÀ-ỹ]+", normalized, flags=re.UNICODE)

        vector = [0.0] * LOCAL_EMBEDDING_DIMENSION
        if not tokens:
            return vector

        counts = Counter(tokens)
        for token, count in counts.items():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % LOCAL_EMBEDDING_DIMENSION
            weight = 1.0 + (count * 0.1)
            vector[index] += weight

        norm = sum(value * value for value in vector) ** 0.5
        if norm:
            vector = [value / norm for value in vector]

        return vector

    def _local_embedding(self, text: str) -> List[float]:
        """Generate a local embedding using Ollama first, falling back to bag-of-words if needed."""
        ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        try:
            # Connect to http://localhost:11434/api/embeddings
            url = f"{ollama_url.rstrip('/')}/api/embeddings"
            payload = {
                "model": "mxbai-embed-large:latest",
                "prompt": text
            }
            # Short timeout to fail fast if Ollama is busy or offline
            response = requests.post(url, json=payload, timeout=8)
            if response.status_code == 200:
                embedding = response.json().get("embedding", [])
                if len(embedding) == LOCAL_EMBEDDING_DIMENSION:
                    return embedding
        except Exception as e:
            logger.warning(f"⚠️ Local Ollama embedding failed: {e}. Using deterministic word counter fallback.")
            
        return self._bag_of_words_embedding(text)

    def _check_health(self):
        """Check if custom embedding service is healthy"""
        try:
            response = requests.get(self.health_endpoint, timeout=5)
            if response.status_code == 200:
                data = response.json()
                if not isinstance(data, dict) or "embedding_dim" not in data:
                    logger.warning("⚠️ Port is occupied by a different service (missing 'embedding_dim'), bypassing custom embedding.")
                    return False
                logger.info(
                    f"✅ Custom embedding service is healthy: "
                    f"device={data.get('device')}, "
                    f"dim={data.get('embedding_dim')}, "
                    f"model_loaded={data.get('model_loaded')}"
                )
                return True
            else:
                logger.warning(
                    f"⚠️ Custom embedding service health check failed: "
                    f"status={response.status_code}"
                )
                return False
        except Exception as e:
            logger.warning(f"⚠️ Cannot connect to custom embedding service: {e}")
            return False

    def _cohere_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate a batch of embeddings using Cohere Cloud API with retry and key rotation on rate limits."""
        keys_str = os.environ.get("COHERE_API_KEYS", "")
        if keys_str:
            api_keys = [k.strip() for k in keys_str.split(",") if k.strip()]
        else:
            single_key = os.environ.get("COHERE_API_KEY")
            api_keys = [single_key] if single_key else []
            
        if not api_keys:
            raise ValueError("Neither COHERE_API_KEYS nor COHERE_API_KEY is configured")
        
        import time
        import requests
        
        # Split into sub-batches of size 90 to stay within standard limits
        sub_batch_size = 90
        all_embeddings = []
        
        if not hasattr(self.__class__, "_current_key_idx"):
            self.__class__._current_key_idx = 0
        
        for idx in range(0, len(texts), sub_batch_size):
            sub_batch = texts[idx:idx + sub_batch_size]
            max_retries = len(api_keys) * 3
            base_delay = 5 # seconds
            
            sub_embeddings = None
            for attempt in range(max_retries):
                current_key = api_keys[self.__class__._current_key_idx % len(api_keys)]
                headers = {
                    "Authorization": f"Bearer {current_key}",
                    "Content-Type": "application/json"
                }
                
                try:
                    payload = {
                        "texts": sub_batch,
                        "model": "embed-multilingual-v3.0",
                        "input_type": "search_document"
                    }
                    response = requests.post(
                        "https://api.cohere.com/v1/embed",
                        json=payload,
                        headers=headers,
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        sub_embeddings = data.get("embeddings", [])
                        break
                    elif response.status_code == 429:
                        logger.warning(f"⚠️ Cohere Rate Limit (429) hit on key index {self.__class__._current_key_idx % len(api_keys)}. Rotating to next key...")
                        self.__class__._current_key_idx += 1
                        
                        if len(api_keys) == 1:
                            delay = base_delay * (2 ** attempt)
                            logger.info(f"⏱️ Only 1 API key configured. Sleeping for {delay} seconds...")
                            time.sleep(delay)
                        else:
                            if attempt >= len(api_keys):
                                sleep_time = 2.0
                                logger.info(f"⏱️ All keys recently hit rate limits. Sleeping {sleep_time}s before next attempt...")
                                time.sleep(sleep_time)
                    else:
                        logger.error(f"❌ Cohere API failed (Status {response.status_code}) on key index {self.__class__._current_key_idx % len(api_keys)}: {response.text}")
                        self.__class__._current_key_idx += 1
                        if attempt == max_retries - 1:
                            raise Exception(f"Cohere API returned status {response.status_code}")
                        time.sleep(2)
                        
                except Exception as e:
                    logger.error(f"❌ Connection error to Cohere API on key index {self.__class__._current_key_idx % len(api_keys)}: {e}")
                    self.__class__._current_key_idx += 1
                    if attempt == max_retries - 1:
                        raise e
                    time.sleep(2)
            
            if sub_embeddings is None:
                raise Exception("Failed to get embeddings from Cohere after retries across rotated keys.")
            all_embeddings.extend(sub_embeddings)
            
            # Balance load across keys for the next chunk
            self.__class__._current_key_idx += 1
            
            # Delay between sub-batches
            if len(api_keys) == 1 and idx + sub_batch_size < len(texts):
                logger.info("⏱️ Sleeping 6 seconds before next Cohere sub-batch to respect rate limits...")
                time.sleep(6.0)
            elif len(api_keys) > 1:
                time.sleep(0.5)
                
        return all_embeddings

    def _local_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate a batch of embeddings using local Ollama if available, fallback to sequential _local_embedding."""
        ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        try:
            url = f"{ollama_url.rstrip('/')}/v1/embeddings"
            payload = {
                "model": "mxbai-embed-large:latest",
                "input": texts
            }
            response = requests.post(url, json=payload, timeout=25)
            if response.status_code == 200:
                data = response.json().get("data", [])
                embeddings = [item.get("embedding") for item in data]
                if embeddings and all(len(emb) == LOCAL_EMBEDDING_DIMENSION for emb in embeddings):
                    return embeddings
        except Exception as e:
            logger.warning(f"⚠️ Ollama batch embedding failed: {e}. Falling back to sequential embedding.")
            
        return [self._local_embedding(item) for item in texts]

    def get_embedding(
        self, text: Union[str, List[str]], batch_size: int = 32
    ) -> Union[List[float], List[List[float]]]:
        """
        Generate embeddings for text(s)

        Args:
            text: Single text string or list of texts
            batch_size: Batch size for processing

        Returns:
            Single embedding vector or list of embedding vectors
        """
        # Normalize input
        is_single = isinstance(text, str)
        texts = [text] if is_single else text

        if not texts:
            raise ValueError("Input text cannot be empty")

        # Clean texts
        texts = [t.replace("\n", " ").strip() for t in texts]

        # Check if we can use Cohere Cloud embeddings (highly recommended: free, fast, 0% local CPU/GPU)
        cohere_api_key = os.environ.get("COHERE_API_KEY")
        if cohere_api_key and not self.service_available:
            try:
                logger.info(f"☁️ Generating {len(texts)} embeddings on Cohere Cloud to save local machine resources...")
                embeddings = self._cohere_batch_embeddings(texts)
                return embeddings[0] if is_single else embeddings
            except Exception:
                logger.warning("⚠️ Cohere cloud embedding failed, falling back to local/Ollama methods.")

        if USE_LOCAL_EMBEDDING_FALLBACK and not self.service_available:
            logger.warning("⚠️ Custom embedding service unavailable, using local fallback embeddings")
            embeddings = self._local_batch_embeddings(texts)
            return embeddings[0] if is_single else embeddings

        try:
            # Call API
            response = requests.post(
                self.embedding_endpoint,
                json={"texts": texts, "batch_size": batch_size},
                timeout=self.timeout,
            )

            if response.status_code != 200:
                raise Exception(
                    f"API returned status {response.status_code}: " f"{response.text}"
                )

            data = response.json()
            embeddings = data.get("embeddings", [])

            if not embeddings:
                raise Exception("No embeddings returned from API")

            logger.info(
                f"✅ Generated {len(embeddings)} embeddings "
                f"(dim={len(embeddings[0])}, "
                f"time={data.get('processing_time', 0):.3f}s)"
            )

            # Return single embedding or list
            return embeddings[0] if is_single else embeddings

        except Exception as e:
            logger.error(f"❌ Error getting embedding from custom service: {e}")
            if USE_LOCAL_EMBEDDING_FALLBACK:
                logger.warning("⚠️ Falling back to local embeddings after remote embedding failure")
                self.service_available = False
                embeddings = [self._local_embedding(item) for item in texts]
                return embeddings[0] if is_single else embeddings
            raise

    def get_similarity(self, texts1: List[str], texts2: List[str]) -> List[List[float]]:
        """
        Calculate similarity between two sets of texts

        Args:
            texts1: First set of texts
            texts2: Second set of texts

        Returns:
            Similarity matrix [len(texts1), len(texts2)]
        """
        try:
            response = requests.post(
                f"{self.api_url}/similarity",
                json={"texts1": texts1, "texts2": texts2},
                timeout=self.timeout,
            )

            if response.status_code != 200:
                raise Exception(
                    f"API returned status {response.status_code}: " f"{response.text}"
                )

            data = response.json()
            similarities = data.get("similarities", [])

            logger.info(
                f"✅ Calculated similarities "
                f"(shape={data.get('shape')}, "
                f"time={data.get('processing_time', 0):.3f}s)"
            )

            return similarities

        except Exception as e:
            logger.error(f"❌ Error calculating similarity: {e}")
            raise


# Global instance
_embedding_service = None


def get_embedding_service() -> CustomEmbeddingService:
    """Get singleton instance of embedding service"""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = CustomEmbeddingService()
    return _embedding_service


def get_custom_embedding(
    text: Union[str, List[str]], batch_size: int = 32
) -> Union[List[float], List[List[float]]]:
    """
    Convenience function to get embeddings from custom model

    Args:
        text: Single text or list of texts
        batch_size: Batch size for processing

    Returns:
        Embedding vector(s)
    """
    service = get_embedding_service()
    return service.get_embedding(text, batch_size)


def get_custom_similarity(texts1: List[str], texts2: List[str]) -> List[List[float]]:
    """
    Convenience function to calculate similarity

    Args:
        texts1: First set of texts
        texts2: Second set of texts

    Returns:
        Similarity matrix
    """
    service = get_embedding_service()
    return service.get_similarity(texts1, texts2)


# For backward compatibility and easy switching
if __name__ == "__main__":
    # Test the service
    import sys

    print("🧪 Testing Custom Embedding Service...")
    print(f"API URL: {CUSTOM_EMBEDDING_API_URL}")
    print(f"Enabled: {CUSTOM_EMBEDDING_ENABLED}")
    print()

    try:
        # Test single text
        print("Test 1: Single text embedding")
        text = "Luật Dân sự năm 2015 quy định về quyền sở hữu"
        embedding = get_custom_embedding(text)
        print(f"✅ Generated embedding with {len(embedding)} dimensions")
        print(f"   Sample values: {embedding[:5]}")
        print()

        # Test batch
        print("Test 2: Batch embeddings")
        texts = [
            "Luật Dân sự năm 2015",
            "Bộ luật Hình sự năm 2017",
            "Luật Đất đai năm 2013",
        ]
        embeddings = get_custom_embedding(texts)
        print(f"✅ Generated {len(embeddings)} embeddings")
        print()

        # Test similarity
        print("Test 3: Similarity calculation")
        texts1 = ["Quyền sở hữu tài sản"]
        texts2 = ["Tài sản chung", "Luật Hình sự", "Quyền thừa kế"]
        similarities = get_custom_similarity(texts1, texts2)
        print(f"✅ Similarities: {similarities}")
        print()

        print("🎉 All tests passed!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        sys.exit(1)