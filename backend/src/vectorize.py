import logging
import os
import time
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    Range,
    VectorParams,
    HnswConfigDiff,
    OptimizersConfigDiff,
)

from pathlib import Path
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load env variables securely
load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=True)

QDRANT_URL = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

def _should_reconnect_qdrant(e: Exception) -> bool:
    err_str = str(e).lower()
    return (
        "10054" in err_str or
        "connection reset" in err_str or
        "forcibly closed" in err_str or
        "remote host" in err_str or
        "connecterror" in err_str or
        "remoteprotocolerror" in err_str or
        "pool" in err_str or
        "timeout" in err_str or
        "broken pipe" in err_str
    )

class QdrantClientWrapper:
    """Proxy class that wraps QdrantClient and handles connection reset / 10054 drops automatically."""
    def __init__(self):
        self._local_client = None

    def _get_underlying(self) -> QdrantClient:
        if self._local_client is None:
            self._local_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60.0)
        return self._local_client

    def _reconnect(self):
        logger.info("🔄 QdrantClientWrapper: Re-establishing connection to Qdrant...")
        try:
            if self._local_client:
                self._local_client.close()
        except Exception:
            pass
        self._local_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60.0)

    def __getattr__(self, name):
        underlying = self._get_underlying()
        attr = getattr(underlying, name)
        if callable(attr):
            def wrapper(*args, **kwargs):
                max_qdrant_attempts = 3
                for attempt in range(max_qdrant_attempts):
                    try:
                        curr_underlying = self._get_underlying()
                        curr_attr = getattr(curr_underlying, name)
                        return curr_attr(*args, **kwargs)
                    except Exception as e:
                        if _should_reconnect_qdrant(e) and attempt < max_qdrant_attempts - 1:
                            logger.warning(f"⚠️ Qdrant connection lost ({e}) during method '{name}'. Reconnecting and retrying (attempt {attempt + 1}/{max_qdrant_attempts})...")
                            time.sleep(1)
                            self._reconnect()
                        else:
                            raise e
            return wrapper
        return attr

client = QdrantClientWrapper()


def get_client():
    """Return the module-level Qdrant client (picks up any test override).

    Internal functions read the ``client`` module global at call time, so a
    test that calls ``set_qdrant_client(mock)`` is observed by all functions
    in this module. External callers should use this getter rather than
    ``from vectorize import client`` so they also observe overrides.
    """
    return client


def set_qdrant_client(new_client):
    """Test seam: inject a (mock) Qdrant client."""
    global client
    client = new_client


def create_collection(name, vector_size=1024):
    """
    Create a collection with enhanced configuration
    """
    return client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.DOT, on_disk=True),
        hnsw_config=HnswConfigDiff(m=0, on_disk=True),
        optimizers_config=OptimizersConfigDiff(indexing_threshold=999999),
    )


def wipe_collection(name, vector_size=1024):
    """
    Recreate the collection to wipe all vectors instantly
    """
    try:
        client.delete_collection(collection_name=name)
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.DOT, on_disk=True),
            hnsw_config=HnswConfigDiff(m=0, on_disk=True),
            optimizers_config=OptimizersConfigDiff(indexing_threshold=999999),
        )
        logger.info(f"Wiped collection: {name}")
        return True
    except Exception as e:
        logger.error(f"Failed to wipe collection {name}: {e}")
        return False


def delete_vectors_by_ids(collection_name, point_ids):
    """
    Delete points by their IDs from Qdrant collection
    """
    if not point_ids:
        return True
    try:
        client.delete(
            collection_name=collection_name,
            points_selector=point_ids,
            wait=True
        )
        logger.info(f"Deleted {len(point_ids)} points from collection {collection_name}")
        return True
    except Exception as e:
        logger.error(f"Failed to delete points from collection {collection_name}: {e}")
        return False


def delete_vectors_by_filter(collection_name, filters):
    """
    Delete points by payload filters (e.g. {"doc_id": doc_id})
    """
    if not filters:
        return True
    try:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        conditions = []
        for field, value in filters.items():
            conditions.append(
                FieldCondition(key=field, match=MatchValue(value=value))
            )
        if conditions:
            client.delete(
                collection_name=collection_name,
                points_selector=Filter(must=conditions),
                wait=True
            )
            logger.info(f"Deleted points by filter {filters} from collection {collection_name}")
            return True
    except Exception as e:
        logger.error(f"Failed to delete points by filter: {e}")
        return False


def list_collections():
    """Return a lightweight list of collection names and basic metadata."""
    collections = client.get_collections().collections
    return [
        {
            "name": collection.name,
        }
        for collection in collections
    ]


def list_collection_points(
    collection_name: str,
    limit: int = 20,
    offset: int = 0,
    include_vectors: bool = False,
):
    """Return points from a collection using Qdrant scroll."""

    def _vector_preview(vector, preview_size: int = 4):
        if vector is None:
            return None

        def _non_zero_preview(values):
            if not isinstance(values, list):
                return values

            non_zero = []
            for idx, val in enumerate(values):
                if val != 0:
                    non_zero.append({"index": idx, "value": val})
                    if len(non_zero) >= preview_size:
                        break

            if non_zero:
                return non_zero

            # Fallback when vector is dense-zero or very tiny values rounded to zero upstream.
            return [{"index": i, "value": values[i]} for i in range(min(preview_size, len(values)))]

        # Handle both single-vector and named-vector collections.
        if isinstance(vector, dict):
            return {name: _non_zero_preview(values) for name, values in vector.items()}

        if isinstance(vector, list):
            return _non_zero_preview(vector)

        return vector

    points, next_offset = client.scroll(
        collection_name=collection_name,
        limit=limit,
        offset=offset,
        with_payload=True,
        with_vectors=True,
    )

    return {
        "collection_name": collection_name,
        "points": [
            {
                "id": point.id,
                "payload": point.payload,
                "vector_preview": _vector_preview(point.vector),
                "vector": point.vector if include_vectors else None,
            }
            for point in points
        ],
        "limit": limit,
        "offset": offset,
        "next_offset": next_offset,
    }


def add_vector(collection_name, vectors={}, batch_size=100):
    """
    Add vectors with improved batch processing and metadata support

    Args:
        collection_name: Name of the collection
        vectors: Dict with structure {id: {"vector": [...], "payload": {...}}}
        batch_size: Number of vectors to process in each batch
    """
    if not vectors:
        return {"status": "no_vectors_provided"}

    # Convert to points
    points = [
        PointStruct(
            id=k,
            vector=v["vector"],
            payload={
                **v["payload"],
                # Add metadata for better filtering
                "doc_length": len(v["payload"].get("content", "")),
                "has_question": bool(v["payload"].get("question", "")),
                "content_type": detect_content_type(v["payload"].get("content", "")),
            },
        )
        for k, v in vectors.items()
    ]

    # Process in batches for better performance
    results = []
    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        try:
            result = client.upsert(
                collection_name=collection_name,
                wait=True,
                points=batch,
            )
            results.append(result)
            logger.info(
                f"Processed batch {i//batch_size + 1}/{(len(points)-1)//batch_size + 1}"
            )
        except Exception as e:
            logger.error(f"Failed to process batch {i//batch_size + 1}: {e}")
            results.append({"error": str(e)})

    return results


def detect_content_type(content: str) -> str:
    """
    Detect the type of legal content for better filtering
    """
    content_lower = content.lower()

    if any(keyword in content_lower for keyword in ["luật", "bộ luật", "pháp luật"]):
        return "law"
    elif any(keyword in content_lower for keyword in ["nghị định", "quyết định"]):
        return "decree"
    elif any(keyword in content_lower for keyword in ["thông tư", "hướng dẫn"]):
        return "circular"
    elif any(keyword in content_lower for keyword in ["hợp đồng", "giao kèo"]):
        return "contract"
    elif any(keyword in content_lower for keyword in ["án lệ", "phán quyết"]):
        return "case_law"
    else:
        return "general"


def search_vector(collection_name, vector, limit=4, filters=None, score_threshold=0.3):
    """
    Enhanced vector search with filtering and scoring options

    Args:
        collection_name: Name of the collection to search
        vector: Query vector
        limit: Maximum number of results
        filters: Optional filters dict {"field": "value"} or {"field": {"gte": value}}
        score_threshold: Minimum similarity score

    Returns:
        List of documents with scores and metadata
    """
    try:
        # Build filter conditions
        filter_conditions = None
        if filters:
            conditions = []

            for field, value in filters.items():
                if isinstance(value, dict):
                    # Range filter
                    if (
                        "gte" in value
                        or "lte" in value
                        or "gt" in value
                        or "lt" in value
                    ):
                        range_filter = Range()
                        if "gte" in value:
                            range_filter.gte = value["gte"]
                        if "lte" in value:
                            range_filter.lte = value["lte"]
                        if "gt" in value:
                            range_filter.gt = value["gt"]
                        if "lt" in value:
                            range_filter.lt = value["lt"]

                        conditions.append(FieldCondition(key=field, range=range_filter))
                else:
                    # Exact match filter
                    conditions.append(
                        FieldCondition(key=field, match=MatchValue(value=value))
                    )

            if conditions:
                filter_conditions = Filter(must=conditions)

        # Perform search
        results = client.search(
            collection_name=collection_name,
            query_vector=vector,
            limit=limit,
            query_filter=filter_conditions,
            score_threshold=score_threshold,
            with_payload=True,
            with_vectors=False,  # Don't return vectors to save bandwidth
        )

        # Process results
        processed_results = []
        for result in results:
            doc = result.payload
            doc["similarity_score"] = result.score
            doc["search_rank"] = len(processed_results) + 1
            processed_results.append(doc)

        logger.info(
            f"Vector search returned {len(processed_results)} results "
            f"(filtered from {len(results)} candidates)"
        )

        return processed_results

    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        return []


def search_with_multiple_vectors(collection_name, vectors, limit=4, filters=None):
    """
    Search with multiple query vectors (for query expansion)

    Args:
        collection_name: Name of the collection
        vectors: List of query vectors
        limit: Results per vector
        filters: Optional filters

    Returns:
        Merged and deduplicated results
    """
    all_results = []
    seen_content_hashes = set()

    for i, vector in enumerate(vectors):
        try:
            results = search_vector(collection_name, vector, limit, filters)

            for result in results:
                content_hash = hash(result.get("content", ""))
                if content_hash not in seen_content_hashes:
                    seen_content_hashes.add(content_hash)
                    result["query_vector_index"] = i
                    all_results.append(result)

        except Exception as e:
            logger.error(f"Search with vector {i} failed: {e}")
            continue

    # Sort by best similarity score
    all_results.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)

    return all_results[:limit]


def get_collection_stats(collection_name):
    """
    Get statistics about a collection
    """
    try:
        info = client.get_collection(collection_name)
        return {
            "name": collection_name,
            "vectors_count": info.vectors_count,
            "indexed_vectors_count": info.indexed_vectors_count,
            "points_count": info.points_count,
            "status": info.status,
            "optimizer_status": info.optimizer_status,
        }
    except Exception as e:
        logger.error(f"Failed to get collection stats: {e}")
        return {"error": str(e)}


def delete_vectors(collection_name, point_ids):
    """
    Delete vectors by IDs
    """
    try:
        result = client.delete(
            collection_name=collection_name, points_selector=point_ids, wait=True
        )
        logger.info(f"Deleted {len(point_ids)} vectors from {collection_name}")
        return result
    except Exception as e:
        logger.error(f"Failed to delete vectors: {e}")
        return {"error": str(e)}