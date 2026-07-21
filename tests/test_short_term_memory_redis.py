"""P1 — Centralized short-term memory tests (Redis-backed, cross-worker).

Covers FLAW 1 fix: rolling summary + count now live in Redis, so two workers
(uvicorn/Celery processes) reading the same ``(user_id, conv_id)`` see the same
summary instead of one worker's in-process OrderedDict drifting.

Uses ``fakeredis`` (dev dep, 2.34.1) so no live Redis needed.
"""
from __future__ import annotations

import pytest
import fakeredis

import memory_short_term as mst


@pytest.fixture
def fake_redis(monkeypatch):
    """Swap the module's Redis client for an in-memory fakeredis instance."""
    server = fakeredis.FakeServer()
    client = fakeredis.FakeStrictRedis(server=server, decode_responses=True)
    monkeypatch.setattr(mst, "_REDIS_ENABLED", True)
    monkeypatch.setattr(mst, "_redis_client", client)
    monkeypatch.setattr(mst, "_summary_cache", type(mst._summary_cache)())
    monkeypatch.setattr(mst, "_fallback_summaries", type(mst._fallback_summaries)())
    monkeypatch.setattr(mst, "_fallback_counts", type(mst._fallback_counts)())
    yield client
    client.flushall()


# ---------------------------------------------------------------------------
# Roundtrip + defaults
# ---------------------------------------------------------------------------

def test_set_then_get_roundtrip(fake_redis):
    assert mst.get_rolling_summary("u1", "c1") == ""
    mst.set_rolling_summary("u1", "c1", "FACTS:\n- sinh nam 1990")
    assert mst.get_rolling_summary("u1", "c1") == "FACTS:\n- sinh nam 1990"
    assert mst.get_summarized_count("u1", "c1") == 0
    mst.set_summarized_count("u1", "c1", 7)
    assert mst.get_summarized_count("u1", "c1") == 7


def test_falsy_ids_are_noops(fake_redis):
    mst.set_rolling_summary("", "c1", "x")
    mst.set_rolling_summary(None, None, "y")
    assert mst.get_rolling_summary("", "c1") == ""
    assert mst.get_summarized_count(None, None) == 0


def test_ttl_is_set_on_write(fake_redis):
    mst.set_rolling_summary("u1", "c1", "s")
    # TTL must be positive (7-day default). fakeredis honors `ex`.
    assert fake_redis.ttl("mem:short:u1:c1:summary") > 0
    mst.set_summarized_count("u1", "c1", 3)
    assert fake_redis.ttl("mem:short:u1:c1:count") > 0


def test_clear_short_term_removes_keys(fake_redis):
    mst.set_rolling_summary("u1", "c1", "s1")
    mst.set_summarized_count("u1", "c1", 2)
    mst.set_rolling_summary("u1", "c2", "s2")
    removed = mst.clear_short_term("u1", "c1")
    assert removed >= 1
    assert mst.get_rolling_summary("u1", "c1") == ""
    # Other conversation for same user survives.
    assert mst.get_rolling_summary("u1", "c2") == "s2"
    # Clear all for user.
    mst.clear_short_term("u1")
    assert mst.get_rolling_summary("u1", "c2") == ""


# ---------------------------------------------------------------------------
# Cross-worker consistency (MLOps-critical for FLAW 1)
# ---------------------------------------------------------------------------

def test_cross_worker_consistency():
    """Two independent clients sharing one Redis see same summary.

    Simulates worker A writing + worker B reading — exact scenario that
    broke with old in-process OrderedDicts.
    """
    server = fakeredis.FakeServer()
    worker_a = fakeredis.FakeStrictRedis(server=server, decode_responses=True)
    worker_b = fakeredis.FakeStrictRedis(server=server, decode_responses=True)

    # Worker A: write via module pointing at worker_a's client.
    mst._redis_client = worker_a
    mst._REDIS_ENABLED = True
    mst._summary_cache.clear()
    mst.set_rolling_summary("u1", "c1", "FACTS:\n- nam, 1990, Ha Noi")
    mst.set_summarized_count("u1", "c1", 4)

    # Worker B: fresh local cache, same Redis backing.
    mst._redis_client = worker_b
    mst._summary_cache.clear()
    assert mst.get_rolling_summary("u1", "c1") == "FACTS:\n- nam, 1990, Ha Noi"
    assert mst.get_summarized_count("u1", "c1") == 4


# ---------------------------------------------------------------------------
# Graceful degrade: Redis off -> in-process fallback (never breaks chat)
# ---------------------------------------------------------------------------

def test_redis_disabled_falls_back_in_process(monkeypatch):
    """MEMORY_REDIS_ENABLED=false -> in-process OrderedDict fallback works."""
    monkeypatch.setattr(mst, "_REDIS_ENABLED", False)
    monkeypatch.setattr(mst, "_redis_client", None)
    monkeypatch.setattr(mst, "_summary_cache", type(mst._summary_cache)())
    monkeypatch.setattr(mst, "_fallback_summaries", type(mst._fallback_summaries)())
    monkeypatch.setattr(mst, "_fallback_counts", type(mst._fallback_counts)())
    mst.set_rolling_summary("u1", "c1", "fallback-summary")
    mst.set_summarized_count("u1", "c1", 5)
    assert mst.get_rolling_summary("u1", "c1") == "fallback-summary"
    assert mst.get_summarized_count("u1", "c1") == 5


class _RedisErr(Exception):
    pass


class _BrokenClient:
    def get(self, *a, **k):
        raise _RedisErr("connection lost")

    def set(self, *a, **k):
        raise _RedisErr("connection lost")


def test_redis_exception_falls_back(monkeypatch):
    """Redis command error mid-flight falls back to in-process, not a crash."""
    monkeypatch.setattr(mst, "_REDIS_ENABLED", True)
    monkeypatch.setattr(mst, "_get_redis_client", lambda: _BrokenClient())
    monkeypatch.setattr(mst, "_summary_cache", type(mst._summary_cache)())
    monkeypatch.setattr(mst, "_fallback_summaries", type(mst._fallback_summaries)())
    monkeypatch.setattr(mst, "_fallback_counts", type(mst._fallback_counts)())
    # set: Redis set raises -> fallback in-process store holds value.
    mst.set_rolling_summary("u1", "c1", "s")
    assert mst.get_rolling_summary("u1", "c1") == "s"


# ---------------------------------------------------------------------------
# agent._summarize_chat_history writes through Redis (integration)
# ---------------------------------------------------------------------------

class _FakeResp:
    def __init__(self, text):
        self.text = text


class _FakeLLM:
    """Stand-in for the llama_index LLM used inside _summarize_chat_history."""

    def __init__(self, text="FACTS:\n- nam, 1990"):
        self._text = text

    def complete(self, prompt):  # noqa: D401
        return _FakeResp(self._text)


def test_summarize_chat_history_writes_redis(fake_redis, monkeypatch):
    """The agent refactor persists rolling summary + count to Redis."""
    import agent

    # Inject a fake LLM so the summary path runs without Groq/Ollama.
    monkeypatch.setattr(agent, "_llm", _FakeLLM("FACTS:\n- nam, 1990, Ha Noi"))

    # 4 prior turns -> old_turns=2 (beyond raw window of 2) -> triggers summary.
    history = [
        {"role": "user", "content": "Toi ten A, sinh 1990"},
        {"role": "assistant", "content": "Chao ban A"},
        {"role": "user", "content": "Toi o Ha Noi"},
        {"role": "assistant", "content": "Da ro"},
    ]
    summary, short_term_raw = agent._summarize_chat_history(history, "u1", "c1")
    # Summary persisted to Redis (cross-worker source of truth).
    assert summary == "FACTS:\n- nam, 1990, Ha Noi"
    assert mst.get_rolling_summary("u1", "c1") == summary
    assert mst.get_summarized_count("u1", "c1") == 2  # len(old_turns)