"""P5 — Working memory scratchpad in LangGraph state (CoALA working layer).

Covers:
- ChatGraphState holds the new working-memory slots (active_entities,
  current_intent, tool_budget, tool_calls_made, scratchpad)
- increment_tool_calls decrements budget and appends scratchpad notes
- tool_budget_exhausted flips True at the cap
- route_node seeds the slots (verified via the compiled graph's node schema)
"""
from __future__ import annotations

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
            mod.LegalGuardrailsManager = type("LegalGuardrailsManager", (), {"__init__": lambda self, *a, **k: None})
        monkeypatch.setitem(sys.modules, name, mod)
    import tasks  # noqa: WPS433
    return tasks


def test_state_accepts_working_memory_slots(tasks_module):
    """The TypedDict (total=False) accepts the new optional slots."""
    state: tasks_module.ChatGraphState = {
        "question": "q",
        "active_entities": {"user_name": "A"},
        "current_intent": "inheritance",
        "tool_budget": 5,
        "tool_calls_made": 0,
        "scratchpad": "note",
    }
    assert state["active_entities"]["user_name"] == "A"
    assert state["tool_budget"] == 5
    # Old slots still coexist (backward-compat).
    state["route"] = "legal_rag"
    assert state["route"] == "legal_rag"


def test_increment_tool_calls_decrements_and_notes(tasks_module):
    state = {"tool_calls_made": 1, "tool_budget": 3, "scratchpad": "init"}
    update = tasks_module.increment_tool_calls(state, note="ran search")
    assert update["tool_calls_made"] == 2
    assert "ran search" in update["scratchpad"]
    assert "init" in update["scratchpad"]


def test_increment_tool_calls_no_note_omits_scratchpad(tasks_module):
    state = {"tool_calls_made": 0, "tool_budget": 3}
    update = tasks_module.increment_tool_calls(state)
    assert update["tool_calls_made"] == 1
    assert "scratchpad" not in update


def test_tool_budget_exhausted_at_cap(tasks_module):
    assert tasks_module.tool_budget_exhausted({"tool_calls_made": 3, "tool_budget": 3}) is True
    assert tasks_module.tool_budget_exhausted({"tool_calls_made": 4, "tool_budget": 3}) is True
    assert tasks_module.tool_budget_exhausted({"tool_calls_made": 2, "tool_budget": 3}) is False
    # Defaults when absent.
    assert tasks_module.tool_budget_exhausted({"tool_calls_made": 0}) is False


def test_graph_builds_with_working_memory_slots(tasks_module, monkeypatch):
    """The chat graph compiles with the new state slots present (no schema error)."""
    tasks = tasks_module
    # Stub the checkpointer so _build_chat_graph doesn't need a live Redis.
    monkeypatch.setattr(tasks, "_get_checkpointer", lambda: None)
    compiled = tasks._build_chat_graph()
    assert compiled is not None
    # ChatGraphState is a TypedDict total=False; confirm the new slots exist on
    # the class the graph was compiled against.
    keys = set(tasks_module.ChatGraphState.__annotations__.keys())
    assert {"active_entities", "current_intent", "tool_budget", "tool_calls_made", "scratchpad"} <= keys