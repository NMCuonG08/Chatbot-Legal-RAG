import logging
import os
import re
import time
import uuid
from typing import Dict, List, Optional

from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from vectorize import get_client
from brain import get_embedding

logger = logging.getLogger(__name__)

CACHE_COLLECTION_NAME = "semantic_cache"
CACHE_SCORE_THRESHOLD = 0.95  # Strict threshold to ensure semantic identity
# Cache entry lifetime in seconds. Entries older than this are treated as a miss
# so stale answers do not surface after the underlying documents are re-indexed.
# 0 disables TTL (legacy behavior). Default: 7 days.
CACHE_TTL_SECONDS = int(os.environ.get("SEMANTIC_CACHE_TTL_SECONDS", str(7 * 24 * 3600)))

# Patterns that mark a response as NOT cacheable. We never want to cache an
# error/placeholder string: a later hit would replay the failure to every user
# asking a similar question, masking the real (now-fixed) answer.
_ERROR_RESPONSE_PATTERNS = [
    r"^xin\s*lõi",
    r"đã\s*xảy\s*ra\s*lỗi",
    r"không\s*thể\s*xử\s*lý",
    r"lỗi\s*hệ\s*thống",
    r"tạm\s*thời\s*không\s*thể",
    r"please\s*try\s*again",
    r"an\s*error\s*occurred",
]
_ERROR_RESPONSE_RE = re.compile("|".join(_ERROR_RESPONSE_PATTERNS), re.IGNORECASE)

SCOPE_COMMON = "common"
SCOPE_USER_PREFIX = "user:"

# Lightweight in-process counters for observability: cache hit/miss/error
# rates and error-skip (poisoning-prevention) counts. Exposed via the app
# /stats endpoint so ops can alert on cache-down (error spike) or hit-rate
# collapse. Not a full metrics backend — enough for an external alerter to
# scrape. Reset on process restart.
_stats = {"hits": 0, "misses": 0, "error_skips": 0, "errors": 0}


def get_cache_stats() -> Dict:
    """Snapshot of cache counters (for the /stats endpoint / alerter scrape)."""
    total = _stats["hits"] + _stats["misses"]
    hit_rate = (_stats["hits"] / total) if total else 0.0
    return {**_stats, "hit_rate": round(hit_rate, 4), "total_lookups": total}


def _scope_for(user_id: Optional[str], scope: Optional[str]) -> str:
    """Resolve the cache scope: explicit scope wins, else per-user if a user is
    present, else the shared ``common`` scope (used for general_chat greetings)."""
    if scope:
        return scope
    if user_id:
        return f"{SCOPE_USER_PREFIX}{user_id}"
    return SCOPE_COMMON


def _is_cacheable_response(response: str) -> bool:
    """True only for a real, non-empty, non-error response worth caching.

    Caching an error string is actively harmful: a later semantic hit replays
    the failure to every user who asks a similar question, even after the
    underlying bug is fixed. Reject these before they enter the cache.
    """
    if not response or not response.strip():
        return False
    if len(response.strip()) < 15:
        return False
    if _ERROR_RESPONSE_RE.search(response):
        return False
    return True


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


def _read_filter(user_id: Optional[str]) -> Optional[Filter]:
    """Build the Qdrant filter that enforces per-user isolation on read.

    - With a ``user_id``: match either the shared ``common`` scope OR this user's
      private scope (``should`` with ``min_should=1``). Legacy points that have
      no ``scope`` field match neither and are excluded — privacy-safe.
    - Without a ``user_id`` (e.g. general_chat route, no user context): only the
      ``common`` scope is readable.
    """
    common_cond = FieldCondition(key="scope", match=MatchValue(value=SCOPE_COMMON))
    if user_id:
        user_cond = FieldCondition(
            key="scope", match=MatchValue(value=f"{SCOPE_USER_PREFIX}{user_id}")
        )
        # `should` matches a point when at least one condition holds (default
        # min_should=1): the entry is readable if it is in the shared common
        # scope OR this user's private scope.
        return Filter(should=[common_cond, user_cond])
    return Filter(must=[common_cond])


def get_cached_response(question: str, user_id: Optional[str] = None) -> Optional[Dict]:
    """Search the semantic cache for a similar question, scoped to the user.

    Returns the cached response dict if a fresh, in-scope, similar entry is
    found, else ``None``. ``user_id`` scopes the lookup so a private answer
    cached for user X is never returned to user Y (cross-user leak prevention).
    """
    try:
        query_vector = get_embedding(question)
        if not query_vector:
            return None

        results = get_client().query_points(
            collection_name=CACHE_COLLECTION_NAME,
            query=query_vector,
            query_filter=_read_filter(user_id),
            limit=1,
            score_threshold=CACHE_SCORE_THRESHOLD,
            with_payload=True,
        )

        points = getattr(results, "points", None)
        if not points:
            _stats["misses"] += 1
            logger.info(f"❄️ Semantic Cache MISS for query: '{question}' (user={user_id})")
            return None

        point = points[0]
        payload = point.payload or {}
        if not _is_within_ttl(payload):
            _stats["misses"] += 1
            logger.info(
                f"❄️ Semantic Cache HIT but expired (age>={CACHE_TTL_SECONDS}s) for query: '{question}'"
            )
            return None

        # Privacy guard: never return a cached entry whose scope does not match
        # the requested user. Defends against filter-implementation drift.
        entry_scope = payload.get("scope")
        if user_id and entry_scope and entry_scope != SCOPE_COMMON and entry_scope != f"{SCOPE_USER_PREFIX}{user_id}":
            _stats["misses"] += 1
            logger.warning(
                f"🚫 Semantic Cache scope mismatch: entry scope={entry_scope}, "
                f"requested user={user_id}. Treating as MISS (privacy)."
            )
            return None

        _stats["hits"] += 1
        logger.info(
            f"🎯 Semantic Cache HIT! Score: {point.score:.4f} "
            f"scope={entry_scope} for query: '{question}' (user={user_id})"
        )
        return {
            "response": payload.get("response", ""),
            "sources": payload.get("sources", []),
            "cached_query": payload.get("query", ""),
            "scope": entry_scope,
            "score": point.score,
        }
    except Exception as e:
        _stats["errors"] += 1
        logger.error(f"Error checking semantic cache: {e}")
        return None


def set_cached_response(
    question: str,
    response: str,
    sources: List[Dict],
    user_id: Optional[str] = None,
    scope: Optional[str] = None,
) -> bool:
    """Save a question and its response/sources into the semantic cache.

    Scope: ``scope`` overrides; otherwise per-user (``user:<user_id>``) when a
    user is present, or ``common`` for user-less contexts (general_chat).

    Refuses to cache error/empty/placeholder responses so a transient failure
    cannot poison the cache and replay to other users later. Returns ``True``
    if the entry was written, ``False`` if it was rejected.
    """
    if not _is_cacheable_response(response):
        _stats["error_skips"] += 1
        logger.info(
            f"🛑 Skipping semantic cache write: response is error/empty/placeholder "
            f"for query: '{question}'"
        )
        return False

    try:
        query_vector = get_embedding(question)
        if not query_vector:
            return False

        resolved_scope = _scope_for(user_id, scope)
        point_id = str(uuid.uuid4())

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
                        "scope": resolved_scope,
                    },
                )
            ],
        )
        logger.info(
            f"💾 Saved response to semantic cache (scope={resolved_scope}) for query: '{question}'"
        )
        return True
    except Exception as e:
        logger.error(f"Failed to save to semantic cache: {e}")
        return False


def clear_semantic_cache() -> int:
    """Delete all points from the semantic cache collection.

    Returns the number of points reported deleted (best-effort). Intended to be
    called after a full collection wipe / bulk re-index so stale cached answers
    cannot surface against freshly embedded documents. Also wipes legacy
    unscoped points so the new scope-filtered reads start from a clean slate.
    """
    try:
        from qdrant_client.models import Filter
        client = get_client()
        deleted = client.delete(
            collection_name=CACHE_COLLECTION_NAME,
            points_selector=Filter(),  # empty Filter matches all points
        )
        logger.info("🧹 Cleared semantic cache collection.")
        # qdrant delete returns an UpdateResult; surface a best-effort count
        return getattr(getattr(deleted, "operation", None), "deleted_count", 0) or 0
    except Exception as e:
        logger.error(f"Failed to clear semantic cache: {e}")
        return 0


def maybe_wipe_legacy_cache() -> int:
    """One-time deploy migration: wipe legacy unscoped cache points.

    Pre-fix points have no ``scope`` field and are excluded by the new read
    filter (privacy-safe) but linger in storage. Run ONCE at deploy, BEFORE new
    traffic, to start clean. Gated by env ``SEMANTIC_CACHE_WIPE_LEGACY=1`` so it
    cannot fire on every restart. At this deploy moment all existing points are
    legacy (no scoped points exist yet), so a full wipe is correct and safe.
    Returns the deleted count, or 0 if the gate is unset.
    """
    flag = os.environ.get("SEMANTIC_CACHE_WIPE_LEGACY", "").strip().lower()
    if flag not in ("1", "true", "yes", "on"):
        logger.info("SEMANTIC_CACHE_WIPE_LEGACY not set — skipping legacy cache wipe.")
        return 0
    logger.info(f"🧹 SEMANTIC_CACHE_WIPE_LEGACY={flag} — wiping legacy cache points.")
    return clear_semantic_cache()