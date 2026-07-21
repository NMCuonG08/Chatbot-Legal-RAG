"""Centralized short-term (rolling summary) memory — cross-worker source of truth.

Replaces the per-process ``OrderedDict`` rolling-summary caches that lived in
``agent.py`` (``_conv_summaries`` / ``_conv_summarized_counts``). Those were
RAM-local to a single uvicorn/Celery process, so a conversation that routed turn
1 to worker A and turn 2 to worker B lost its summary -> "amnesia" / drift across
turns (CoALA/MemGPT violation: short-term memory MUST live in a centralized store).

This module is the single source of truth, backed by Redis (keyed by
``(user_id, conversation_id)``) with a 7-day TTL. An optional in-process LRU
read-through cache avoids a Redis round-trip within one process; Redis remains
authoritative, so cross-worker reads still see the latest write.

Graceful degrade: when Redis is unreachable (or ``MEMORY_REDIS_ENABLED=false``),
the module falls back to an in-process ``OrderedDict`` (legacy behavior) so a
Redis outage never breaks the chat flow. The flag defaults to the NEW behavior
(Redis); set ``MEMORY_REDIS_ENABLED=false`` for one release to roll back.

Public API (all no-ops returning defaults when user_id/conv_id falsy):
- get_rolling_summary / set_rolling_summary
- get_summarized_count / set_summarized_count
- clear_short_term
"""

from __future__ import annotations

import logging
import os
import threading
from collections import OrderedDict
from typing import Optional, Tuple

from database import settings

logger = logging.getLogger(__name__)

# P6c — Prometheus counters for short-term hit/miss (no-op if prom absent).
try:
    from memory_metrics import inc_short_term_hit, inc_short_term_miss
except Exception:  # pragma: no cover
    def inc_short_term_hit() -> None:
        pass

    def inc_short_term_miss() -> None:
        pass

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Default to the NEW (Redis) behavior. Set to "false" to fall back to the legacy
# in-process OrderedDicts for one release window (rollback safety, P7).
_REDIS_ENABLED = os.environ.get("MEMORY_REDIS_ENABLED", "true").lower() == "true"

# 7 days. Refreshed on every write so an active conversation never expires mid-chat.
_TTL_SECONDS = int(os.environ.get("MEMORY_SHORT_TERM_TTL", str(7 * 24 * 3600)))

_KEY_PREFIX = "mem:short"
_LRU_CAP = 32

# ---------------------------------------------------------------------------
# Redis client (lazy singleton)
# ---------------------------------------------------------------------------

_redis_client = None
_redis_lock = threading.Lock()


def _get_redis_client():
    """Lazily build a decode_responses Redis client. Returns None if unavailable."""
    global _redis_client
    if not _REDIS_ENABLED:
        return None
    with _redis_lock:
        if _redis_client is None:
            try:
                import redis  # local import keeps test collection light
                _redis_client = redis.from_url(settings.redis_url, decode_responses=True)
                _redis_client.ping()  # fail fast if broker down
                logger.info(f"[SHORT-TERM] Redis client ready: {settings.redis_url}")
            except Exception as exc:  # pragma: no cover - depends on env
                logger.warning(f"[SHORT-TERM] Redis unavailable, falling back to in-process: {exc}")
                _redis_client = None
        return _redis_client


# ---------------------------------------------------------------------------
# In-process fallback (legacy behavior) — used only when Redis is off/down.
# Kept symmetric to the old agent.py OrderedDicts so the fallback is behaviorally
# identical, not a silent regression.
# ---------------------------------------------------------------------------

_fallback_summaries: "OrderedDict[Tuple[Optional[str], Optional[str]], str]" = OrderedDict()
_fallback_counts: "OrderedDict[Tuple[Optional[str], Optional[str]], int]" = OrderedDict()
_fallback_lock = threading.Lock()


def _fallback_move_to_end(key):
    _fallback_summaries.move_to_end(key)
    _fallback_counts.move_to_end(key)
    if len(_fallback_summaries) > _LRU_CAP:
        ev_k, _ = _fallback_summaries.popitem(last=False)
        _fallback_counts.pop(ev_k, None)
    if len(_fallback_counts) > _LRU_CAP:
        _fallback_counts.popitem(last=False)


# ---------------------------------------------------------------------------
# Local read-through LRU cache for the summary (perf only; Redis is authoritative).
# ---------------------------------------------------------------------------

_summary_cache: "OrderedDict[Tuple[Optional[str], Optional[str]], str]" = OrderedDict()
_cache_lock = threading.Lock()


def _cache_get(key) -> Optional[str]:
    with _cache_lock:
        val = _summary_cache.get(key)
        if val is not None:
            _summary_cache.move_to_end(key)
        return val


def _cache_put(key, value: str) -> None:
    with _cache_lock:
        _summary_cache[key] = value
        _summary_cache.move_to_end(key)
        if len(_summary_cache) > _LRU_CAP:
            _summary_cache.popitem(last=False)


def _cache_invalidate(key) -> None:
    with _cache_lock:
        _summary_cache.pop(key, None)


# ---------------------------------------------------------------------------
# Key helpers
# ---------------------------------------------------------------------------

def _key(user_id: Optional[str], conv_id: Optional[str], suffix: str) -> str:
    return f"{_KEY_PREFIX}:{user_id}:{conv_id}:{suffix}"


def _norm(user_id: Optional[str], conv_id: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    # Falsy ids -> no-op (anonymous / not-yet-persisted conversation).
    return (user_id or "", conv_id or "")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_rolling_summary(user_id: Optional[str], conv_id: Optional[str]) -> str:
    """Return the rolling summary, or "" if none / unavailable."""
    u, c = _norm(user_id, conv_id)
    if not u or not c:
        return ""
    cached = _cache_get((u, c))
    if cached is not None:
        if cached:
            inc_short_term_hit()
        else:
            inc_short_term_miss()
        return cached

    client = _get_redis_client()
    if client is not None:
        try:
            val = client.get(_key(u, c, "summary")) or ""
            _cache_put((u, c), val)
            if val:
                inc_short_term_hit()
            else:
                inc_short_term_miss()
            return val
        except Exception as exc:  # pragma: no cover
            logger.warning(f"[SHORT-TERM] Redis get summary failed: {exc}")
    with _fallback_lock:
        val = _fallback_summaries.get((u, c), "")
        if val:
            inc_short_term_hit()
        else:
            inc_short_term_miss()
        return val


def set_rolling_summary(user_id: Optional[str], conv_id: Optional[str], summary: str) -> None:
    u, c = _norm(user_id, conv_id)
    if not u or not c:
        return
    _cache_put((u, c), summary)
    client = _get_redis_client()
    if client is not None:
        try:
            client.set(_key(u, c, "summary"), summary, ex=_TTL_SECONDS)
            return
        except Exception as exc:  # pragma: no cover
            logger.warning(f"[SHORT-TERM] Redis set summary failed: {exc}")
    # Fallback in-process.
    with _fallback_lock:
        _fallback_summaries[(u, c)] = summary
        _fallback_counts.setdefault((u, c), 0)
        _fallback_move_to_end((u, c))


def get_summarized_count(user_id: Optional[str], conv_id: Optional[str]) -> int:
    u, c = _norm(user_id, conv_id)
    if not u or not c:
        return 0
    client = _get_redis_client()
    if client is not None:
        try:
            raw = client.get(_key(u, c, "count"))
            return int(raw) if raw is not None else 0
        except Exception as exc:  # pragma: no cover
            logger.warning(f"[SHORT-TERM] Redis get count failed: {exc}")
    with _fallback_lock:
        return _fallback_counts.get((u, c), 0)


def set_summarized_count(user_id: Optional[str], conv_id: Optional[str], count: int) -> None:
    u, c = _norm(user_id, conv_id)
    if not u or not c:
        return
    client = _get_redis_client()
    if client is not None:
        try:
            client.set(_key(u, c, "count"), int(count), ex=_TTL_SECONDS)
            return
        except Exception as exc:  # pragma: no cover
            logger.warning(f"[SHORT-TERM] Redis set count failed: {exc}")
    with _fallback_lock:
        _fallback_counts[(u, c)] = int(count)
        _fallback_summaries.setdefault((u, c), "")
        _fallback_move_to_end((u, c))


def clear_short_term(user_id: Optional[str], conv_id: Optional[str] = None) -> int:
    """Drop short-term entries for a user (and optional conversation).

    Returns the number of cache entries removed (best-effort). Only affects the
    process it runs in for the fallback path; with Redis, the delete is global.
    A Celery worker purge dispatches ``clear_user_runtime_caches_task`` so the
    worker's local LRU is dropped too (see tasks.clear_user_runtime_caches_task).
    """
    u, c = _norm(user_id, conv_id)
    if not u:
        return 0
    removed = 0

    # Local LRU: drop matching keys.
    with _cache_lock:
        for k in list(_summary_cache.keys()):
            if k[0] == u and (c == "" or k[1] == c):
                _summary_cache.pop(k, None)
                removed += 1
    # Fallback stores.
    with _fallback_lock:
        for store in (_fallback_summaries, _fallback_counts):
            for k in list(store.keys()):
                if k[0] == u and (c == "" or k[1] == c):
                    store.pop(k, None)
                    removed += 1
    # Redis: delete keys (global).
    client = _get_redis_client()
    if client is not None:
        try:
            if c:
                client.delete(_key(u, c, "summary"), _key(u, c, "count"))
            else:
                # No conv_id: scan all mem:short:{u}:* and delete.
                pattern = f"{_KEY_PREFIX}:{u}:*"
                keys = list(client.scan_iter(match=pattern, count=100))
                if keys:
                    client.delete(*keys)
        except Exception as exc:  # pragma: no cover
            logger.warning(f"[SHORT-TERM] Redis clear failed: {exc}")
    logger.info(f"[SHORT-TERM] Cleared user={u} conv={c or '*'} entries_removed={removed}")
    return removed