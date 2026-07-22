"""Audit 2.2 — run_chat_graph per-conversation lock integration.

Stubs the heavy import chain (same pattern as test_canary_shadow) so ``tasks``
is importable, then exercises the real ``run_chat_graph`` with a fake graph and
a monkeypatched ``acquire_conversation_lock`` / ``release_conversation_lock``.
"""
import sys
import types

import pytest


@pytest.fixture
def tasks_module(monkeypatch):
    for name in ("agent", "verify_answer", "metacognitive", "rlhf_store",
                 "summarizer", "guardrails_manager"):
        mod = types.ModuleType(name)
        if name == "agent":
            mod.ai_agent_handle = lambda *a, **k: None
            mod.clear_user_runtime_caches = lambda *a, **k: None
            mod.filter_tools_for_query = lambda *a, **k: []
        if name == "verify_answer":
            mod.judge_answer = lambda *a, **k: None
        if name == "metacognitive":
            mod.build_escalation = lambda *a, **k: None
            mod.ESCALATION_PREFIX = "[ESCALATION]"
        if name == "rlhf_store":
            mod.find_similar_good = lambda *a, **k: None
        if name == "summarizer":
            mod.summarize_text = lambda *a, **k: "summary"
        if name == "guardrails_manager":
            class _GM:
                initialized = False
            mod.LegalGuardrailsManager = lambda: _GM()
        monkeypatch.setitem(sys.modules, name, mod)
    import importlib
    import tasks as tasks_mod
    importlib.reload(tasks_mod)
    import config as _cfg
    monkeypatch.setattr(_cfg, "COST_ROUTING_ENABLED", False)
    monkeypatch.setattr(_cfg, "SHADOW_MODE_ENABLED", False)
    yield tasks_mod


def _fake_graph(seen):
    class _FakeGraph:
        def invoke(self, state, config=None):
            seen.append(1)
            return {
                "response": "answer", "sources": [], "route": "general_chat",
                "reflection_count": 0, "tool_calls": [],
                "verify_score": 0.9, "verify_verdict": "supported",
                "retry_verify": 0,
            }
    return _FakeGraph()


def _patch_lock(tasks_module, monkeypatch, status, record):
    """Replace acquire/release with stubs returning ``status`` and recording."""
    fake_lock = object() if status == "acquired" else None

    def _acquire(thread_id, client, ttl_s, blocking_timeout=None):
        record["acquired"].append(thread_id)
        return fake_lock, status

    def _release(lock):
        record["released"].append(lock)

    monkeypatch.setattr(tasks_module, "acquire_conversation_lock", _acquire)
    monkeypatch.setattr(tasks_module, "release_conversation_lock", _release)
    monkeypatch.setattr(tasks_module, "redis_client", object())  # non-None client


# ---------------------------------------------------------------------------
def test_contended_returns_busy_and_does_not_invoke_graph(tasks_module, monkeypatch):
    seen = []
    monkeypatch.setattr(tasks_module, "get_chat_graph", lambda: _fake_graph(seen))
    record = {"acquired": [], "released": []}
    _patch_lock(tasks_module, monkeypatch, "contended", record)

    result = tasks_module.run_chat_graph([], "hello", user_id="u", conversation_id="c1")

    assert result["route"] == "busy"
    assert "đang xử lý" in result["response"]
    assert seen == []  # graph.invoke never called -> no state overwrite
    assert record["acquired"] == ["c1"]


def test_acquired_invokes_graph_then_releases(tasks_module, monkeypatch):
    seen = []
    monkeypatch.setattr(tasks_module, "get_chat_graph", lambda: _fake_graph(seen))
    record = {"acquired": [], "released": []}
    _patch_lock(tasks_module, monkeypatch, "acquired", record)

    result = tasks_module.run_chat_graph([], "hello", user_id="u", conversation_id="c1")

    assert result["response"] == "answer"
    assert seen == [1]  # graph invoked
    assert record["acquired"] == ["c1"]
    assert len(record["released"]) == 1  # lock released in finally


def test_unavailable_proceeds_best_effort(tasks_module, monkeypatch):
    seen = []
    monkeypatch.setattr(tasks_module, "get_chat_graph", lambda: _fake_graph(seen))
    record = {"acquired": [], "released": []}
    _patch_lock(tasks_module, monkeypatch, "unavailable", record)

    result = tasks_module.run_chat_graph([], "hello", user_id="u", conversation_id="c1")

    assert result["response"] == "answer"  # Redis down -> still serves
    assert seen == [1]
    assert record["released"] == []  # no lock object to release


def test_skip_conv_lock_bypasses_acquire(tasks_module, monkeypatch):
    seen = []
    monkeypatch.setattr(tasks_module, "get_chat_graph", lambda: _fake_graph(seen))
    record = {"acquired": [], "released": []}
    _patch_lock(tasks_module, monkeypatch, "contended", record)

    result = tasks_module.run_chat_graph(
        [], "hello", user_id="u", conversation_id="c1", skip_conv_lock=True)

    assert result["response"] == "answer"  # not blocked despite "contended" stub
    assert seen == [1]
    assert record["acquired"] == []  # lock never acquired (shadow path)