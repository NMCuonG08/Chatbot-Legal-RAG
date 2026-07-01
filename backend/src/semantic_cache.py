import logging
import os
import time
import uuid
from typing import Dict, Optional, List
from qdrant_client.models import Distance, VectorParams, PointStruct
from vectorize import get_client
from brain import get_embedding

logger = logging.getLogger(__name__)

CACHE_COLLECTION_NAME = "semantic_cache"
CACHE_SCORE_THRESHOLD = 0.95  # Strict threshold to ensure semantic identity
# Cache entry lifetime in seconds. Entries older than this are treated as a miss
# so stale answers do not surface after the underlying documents are re-indexed.
# 0 disables TTL (legacy behavior). Default: 7 days.
CACHE_TTL_SECONDS = int(os.environ.get("SEMANTIC_CACHE_TTL_SECONDS", str(7 * 24 * 3600)))


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


def _is_within_ttl(payload: Dict) -> bool:
    """True if the cache entry is still within its TTL window.

    Entries written before the TTL field existed (or with TTL disabled via
    ``SEMANTIC_CACHE_TTL_SECONDS=0``) are treated as non-expiring so the change
    stays backward compatible with previously written points.
    """
    if CACHE_TTL_SECONDS <= 0:
        return True
    cached_at = payload.get("cached_at")
    if not cached_at:
        return True  # legacy point without timestamp — do not drop silently
    return (time.time() - float(cached_at)) <= CACHE_TTL_SECONDS


def get_cached_response(question: str) -> Optional[Dict]:
    """
    Search the semantic cache for a similar question.
    Returns the cached response dict if a fresh, similar entry is found, else None.
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
            payload = point.payload or {}
            if not _is_within_ttl(payload):
                logger.info(
                    f"❄️ Semantic Cache HIT but expired (age>={CACHE_TTL_SECONDS}s) for query: '{question}'"
                )
                return None
            logger.info(f"🎯 Semantic Cache HIT! Score: {point.score:.4f} for query: '{question}'")
            return {
                "response": payload.get("response", ""),
                "sources": payload.get("sources", []),
                "cached_query": payload.get("query", ""),
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

    A ``cached_at`` unix timestamp is stored in the payload so entries can expire
    after ``SEMANTIC_CACHE_TTL_SECONDS`` (prevents stale answers after re-index).
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
                        "cached_at": time.time(),
                    }
                )
            ]
        )
        logger.info(f"💾 Saved response to semantic cache for query: '{question}'")
    except Exception as e:
        logger.error(f"Failed to save to semantic cache: {e}")


def clear_semantic_cache() -> int:
    """Delete all points from the semantic cache collection.

    Returns the number of points reported deleted (best-effort). Intended to be
    called after a full collection wipe / bulk re-index so stale cached answers
    cannot surface against freshly embedded documents.
    """
    try:
        client = get_client()
        deleted = client.delete(
            collection_name=CACHE_COLLECTION_NAME,
            points_selector={},  # empty filter = all points
        )
        logger.info("🧹 Cleared semantic cache collection.")
        # qdrant delete returns an UpdateResult; surface a best-effort count
        return getattr(getattr(deleted, "operation", None), "deleted_count", 0) or 0
    except Exception as e:
        logger.error(f"Failed to clear semantic cache: {e}")
        return 0