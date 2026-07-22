"""Per-conversation Redis distributed lock (audit 2.2).

Two concurrent Celery workers handling the same ``conversation_id`` race on the
same LangGraph checkpoint (keyed by ``thread_id``) and overwrite each other's
done-flags (``web_to_agent_done`` / ``agent_to_rag_done`` / ...). A Redis lock
keyed on the thread id ensures only one worker mutates graph state at a time.

Three outcomes:
- ``acquired``  -> lock held; caller runs the graph, then releases.
- ``contended`` -> another worker holds it; caller rejects (returns busy) so it
                  does NOT overwrite the in-flight run.
- ``unavailable`` -> Redis down / error; caller proceeds best-effort (no lock),
                  matching the existing document-index lock's degrade policy.

Pure helpers + a Redis client arg so it is unit-testable with fakeredis.
"""
from __future__ import annotations

import logging
import os
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

CONV_LOCK_PREFIX = "lock:conv"
DEFAULT_BLOCKING_TIMEOUT = float(os.environ.get("CONV_LOCK_BLOCKING_TIMEOUT", "5"))


def conv_lock_key(thread_id: str) -> str:
    """Redis key for the per-conversation lock."""
    return f"{CONV_LOCK_PREFIX}:{thread_id}"


def acquire_conversation_lock(
    thread_id: str,
    client,
    ttl_s: float,
    blocking_timeout: Optional[float] = None,
) -> Tuple[Optional[object], str]:
    """Try to acquire the per-conversation lock.

    Returns ``(lock, status)`` where status is ``acquired`` / ``contended`` /
    ``unavailable``. ``ttl_s`` must exceed the graph run timeout so the lock
    does not expire mid-run (which would let another worker grab it). Use
    ``blocking_timeout=0`` for non-blocking probes.
    """
    if client is None:
        return None, "unavailable"
    bt = DEFAULT_BLOCKING_TIMEOUT if blocking_timeout is None else blocking_timeout
    try:
        lock = client.lock(
            conv_lock_key(thread_id),
            timeout=int(max(1, ttl_s)),
            blocking_timeout=bt,
        )
        got = lock.acquire(blocking=True, blocking_timeout=bt)
        if got:
            return lock, "acquired"
        return None, "contended"
    except Exception as exc:
        logger.warning("conversation lock unavailable: %s", exc)
        return None, "unavailable"


def release_conversation_lock(lock) -> None:
    """Release the lock; no-op on None; never raises (best-effort cleanup)."""
    if lock is None:
        return
    try:
        lock.release()
    except Exception as exc:
        logger.debug("conversation lock release skipped: %s", exc)