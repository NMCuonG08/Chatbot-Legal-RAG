"""Audit 2.2 — per-conversation Redis distributed lock.

Two concurrent Celery workers fielding the same conversation_id race on the
same LangGraph checkpoint (thread_id) and overwrite each other's done-flags.
A Redis lock keyed on thread_id ensures only one worker mutates graph state at
a time. Redis down -> best-effort proceed (unavailable); lock held by another
worker -> contended (caller rejects instead of overwriting).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend" / "src"))

import fakeredis
import conversation_lock as cl


def _client():
    return fakeredis.FakeStrictRedis(decode_responses=True)


# ---------------------------------------------------------------------------
# acquire_conversation_lock
# ---------------------------------------------------------------------------
def test_acquire_on_free_key_returns_acquired():
    c = _client()
    lock, status = cl.acquire_conversation_lock("conv-1", c, ttl_s=60, blocking_timeout=0)
    assert status == "acquired"
    assert lock is not None


def test_second_acquire_while_held_is_contended():
    c = _client()
    lock1, s1 = cl.acquire_conversation_lock("conv-1", c, ttl_s=60, blocking_timeout=0)
    assert s1 == "acquired"
    lock2, s2 = cl.acquire_conversation_lock("conv-1", c, ttl_s=60, blocking_timeout=0)
    assert s2 == "contended"
    assert lock2 is None


def test_acquire_after_release_succeeds():
    c = _client()
    lock1, s1 = cl.acquire_conversation_lock("conv-1", c, ttl_s=60, blocking_timeout=0)
    cl.release_conversation_lock(lock1)
    lock2, s2 = cl.acquire_conversation_lock("conv-1", c, ttl_s=60, blocking_timeout=0)
    assert s2 == "acquired"


def test_none_client_returns_unavailable():
    lock, status = cl.acquire_conversation_lock("conv-1", None, ttl_s=60)
    assert status == "unavailable"
    assert lock is None


def test_distinct_conversations_lock_independently():
    c = _client()
    lock_a, sa = cl.acquire_conversation_lock("conv-a", c, ttl_s=60, blocking_timeout=0)
    lock_b, sb = cl.acquire_conversation_lock("conv-b", c, ttl_s=60, blocking_timeout=0)
    assert sa == "acquired" and sb == "acquired"


# ---------------------------------------------------------------------------
# release_conversation_lock
# ---------------------------------------------------------------------------
def test_release_none_is_noop():
    cl.release_conversation_lock(None)  # must not raise


def test_release_unblocks_other_worker():
    c = _client()
    lock1, _ = cl.acquire_conversation_lock("conv-1", c, ttl_s=60, blocking_timeout=0)
    _, s2 = cl.acquire_conversation_lock("conv-1", c, ttl_s=60, blocking_timeout=0)
    assert s2 == "contended"
    cl.release_conversation_lock(lock1)
    _, s3 = cl.acquire_conversation_lock("conv-1", c, ttl_s=60, blocking_timeout=0)
    assert s3 == "acquired"


# ---------------------------------------------------------------------------
# key derivation
# ---------------------------------------------------------------------------
def test_lock_key_is_namespaced_per_conversation():
    assert cl.conv_lock_key("conv-1") == "lock:conv:conv-1"
    assert cl.conv_lock_key("conv-1") != cl.conv_lock_key("conv-2")