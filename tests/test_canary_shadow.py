"""Unit tests for P6 canary/shadow: variant contextvar override + shadow dual-run.

Stubs the heavy import chain (agent/verify_answer/metacognitive/rlhf_store) so
``tasks`` is importable without langchain_groq, then exercises the real
``run_chat_graph`` with a fake graph that records the active model contextvar.
"""
import sys
import types

import pytest


@pytest.fixture
def tasks_module(monkeypatch):
    """Import tasks with heavy deps stubbed. Yields the tasks module."""
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
    # run_chat_graph reads flags via `from config import ...` at call time, so
    # patch the config module itself (not the tasks namespace copy).
    import config as _cfg
    monkeypatch.setattr(_cfg, "COST_ROUTING_ENABLED", False)
    monkeypatch.setattr(_cfg, "SHADOW_MODE_ENABLED", False)
    yield tasks_mod


def _fake_graph(seen_models):
    class _FakeGraph:
        def invoke(self, state, config=None):
            from brain import LLM_MODEL_CONTEXTVAR
            seen_models.append(LLM_MODEL_CONTEXTVAR.get())
            return {
                "response": "answer", "sources": [], "route": "general_chat",
                "reflection_count": 0, "tool_calls": [],
                "verify_score": 0.9, "verify_verdict": "supported",
                "retry_verify": 0,
            }
    return _FakeGraph()


def test_variant_overrides_model_contextvar(tasks_module, monkeypatch):
    seen = []
    monkeypatch.setattr(tasks_module, "get_chat_graph", lambda: _fake_graph(seen))
    import config as _cfg; monkeypatch.setattr(_cfg, "COST_ROUTING_ENABLED", False)
    import config as _cfg; monkeypatch.setattr(_cfg, "SHADOW_MODE_ENABLED", False)
    result = tasks_module.run_chat_graph(
        [], "hello", user_id="u", variant="llama-3.1-8b-instant", shadow=False)
    assert seen == ["llama-3.1-8b-instant"]
    assert result["variant"] == "llama-3.1-8b-instant"
    assert result["response"] == "answer"


def test_cost_routing_applies_small_for_general(tasks_module, monkeypatch):
    seen = []
    monkeypatch.setattr(tasks_module, "get_chat_graph", lambda: _fake_graph(seen))
    import config as _cfg; monkeypatch.setattr(_cfg, "COST_ROUTING_ENABLED", True)
    import config as _cfg; monkeypatch.setattr(_cfg, "SHADOW_MODE_ENABLED", False)
    tasks_module.run_chat_graph([], "hello world", user_id="u")
    from evaluation.cost_routing import SMALL_MODEL
    assert seen == [SMALL_MODEL]


def test_cost_routing_applies_big_for_legal(tasks_module, monkeypatch):
    seen = []
    monkeypatch.setattr(tasks_module, "get_chat_graph", lambda: _fake_graph(seen))
    import config as _cfg; monkeypatch.setattr(_cfg, "COST_ROUTING_ENABLED", True)
    import config as _cfg; monkeypatch.setattr(_cfg, "SHADOW_MODE_ENABLED", False)
    tasks_module.run_chat_graph([], "Điều 10 Bộ luật Dân sự nói gì?", user_id="u")
    from evaluation.cost_routing import BIG_MODEL
    assert seen == [BIG_MODEL]


def test_shadow_runs_candidate_and_user_gets_primary(tasks_module, monkeypatch):
    seen = []
    monkeypatch.setattr(tasks_module, "get_chat_graph", lambda: _fake_graph(seen))
    import config as _cfg; monkeypatch.setattr(_cfg, "COST_ROUTING_ENABLED", False)
    import config as _cfg; monkeypatch.setattr(_cfg, "SHADOW_MODE_ENABLED", True)

    fake_trace = types.ModuleType("trace")
    fake_trace.emit_step = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "trace", fake_trace)

    result = tasks_module.run_chat_graph(
        [], "hello", user_id="u", run_id="run-1", variant=None, shadow=True)
    assert result["response"] == "answer"   # primary to user
    assert len(seen) == 2                    # primary + shadow candidate


def test_shadow_disabled_when_flag_off(tasks_module, monkeypatch):
    seen = []
    monkeypatch.setattr(tasks_module, "get_chat_graph", lambda: _fake_graph(seen))
    import config as _cfg; monkeypatch.setattr(_cfg, "COST_ROUTING_ENABLED", False)
    import config as _cfg; monkeypatch.setattr(_cfg, "SHADOW_MODE_ENABLED", False)
    tasks_module.run_chat_graph([], "hello", user_id="u", run_id="r", shadow=True)
    assert len(seen) == 1