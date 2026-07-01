import logging
import uuid
from typing import Dict, Optional, List
from qdrant_client.models import Distance, VectorParams, PointStruct
from vectorize import get_client
from brain import get_embedding

logger = logging.getLogger(__name__)

CACHE_COLLECTION_NAME = "semantic_cache"
CACHE_SCORE_THRESHOLD = 0.95  # Strict threshold to ensure semantic identity

def init_semantic_cache():
    """Ensure the semantic cache collection exists in Qdrant."""
    try:
        collections = [c.name for c in get_client().get_collections().collections]
        if CACHE_COLLECTION_NAME not in collections:
            logger.info(f"Creating Qdrant semantic cache collection: {CACHE_COLLECTION_NAME}")
            get_client().create_collection(
                collection_name=CACHE_COLLECTION_NAME,
                vectors_config=VectorParams(size=1024, distance=Distance.DOT),  # DOT product for normalized vectors
            )
            logger.info("✅ Semantic cache collection created successfully")
        else:
            logger.info("✅ Semantic cache collection already exists")
    except Exception as e:
        logger.error(f"Failed to initialize semantic cache: {e}")


def get_cached_response(question: str) -> Optional[Dict]:
    """
    Search the semantic cache for a similar question.
    Returns the cached response dict if found, else None.
    """
    try:
        # Get query embedding
        query_vector = get_embedding(question)
        if not query_vector:
            return None

        # Search in Qdrant semantic_cache collection
        results = get_client().search(
            collection_name=CACHE_COLLECTION_NAME,
            query_vector=query_vector,
            limit=1,
            score_threshold=CACHE_SCORE_THRESHOLD
        )

        if results:
            point = results[0]
            logger.info(f"🎯 Semantic Cache HIT! Score: {point.score:.4f} for query: '{question}'")
            return {
                "response": point.payload.get("response", ""),
                "sources": point.payload.get("sources", []),
                "cached_query": point.payload.get("query", ""),
                "score": point.score
            }
        
        logger.info(f"❄️ Semantic Cache MISS for query: '{question}'")
        return None
    except Exception as e:
        logger.error(f"Error checking semantic cache: {e}")
        return None


def set_cached_response(question: str, response: str, sources: List[Dict]):
    """
    Save a question and its generated response/sources into the semantic cache.
    """
    try:
        query_vector = get_embedding(question)
        if not query_vector:
            return

        # Generate a unique UUID for the cache point
        point_id = str(uuid.uuid4())

        # Upsert point into Qdrant semantic_cache
        get_client().upsert(
            collection_name=CACHE_COLLECTION_NAME,
            points=[
                PointStruct(
                    id=point_id,
                    vector=query_vector,
                    payload={
                        "query": question,
                        "response": response,
                        "sources": sources,
                    }
                )
            ]
        )
        logger.info(f"💾 Saved response to semantic cache for query: '{question}'")
    except Exception as e:
        logger.error(f"Failed to save to semantic cache: {e}")
