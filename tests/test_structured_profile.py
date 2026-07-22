"""P4 — Structured UserProfile + procedural memory tests (CoALA layers).

Covers:
- merge_user_profile updates ONLY non-null fields (idempotent, no null overwrite)
- get_user_profile returns dict / None
- procedural_memory.workflow_block: injected for known case_type, "" for unknown
- _get_agent_system_prompt appends workflow block when profile.case_type known
- save_episodic_memory_task merges structured fields when flag on
- recall_user_memory_tool returns "profile" key when flag on
"""
from __future__ import annotations

import json
import sys
import types

import pytest

from procedural_memory import workflow_block, CASE_WORKFLOWS


# ---------------------------------------------------------------------------
# UserProfile accessors (sqlite-backed)
# ---------------------------------------------------------------------------

def test_get_user_profile_none_when_absent(sqlite_db):
    import models
    assert models.get_user_profile("nope") is None


def test_merge_partial_update_only_non_null(sqlite_db):
    """Merge never overwrites an existing field with null; adds new fields."""
    import models
    models.merge_user_profile("u1", {"name": "Nguyen A", "birth_year": 1990})
    models.merge_user_profile("u1", {"location": "Ha Noi", "birth_year": None, "name": None})
    profile = models.get_user_profile("u1")
    assert profile["name"] == "Nguyen A"          # not overwritten by null
    assert profile["birth_year"] == 1990          # not overwritten by null
    assert profile["location"] == "Ha Noi"        # new field merged


def test_merge_coerces_birth_year_string_to_int(sqlite_db):
    import models
    models.merge_user_profile("u1", {"birth_year": "1995", "gender": "nam"})
    profile = models.get_user_profile("u1")
    assert profile["birth_year"] == 1995
    assert profile["gender"] == "nam"


def test_merge_ignores_unknown_fields(sqlite_db):
    import models
    models.merge_user_profile("u1", {"name": "A", "evil_column": "drop table"})
    profile = models.get_user_profile("u1")
    assert profile["name"] == "A"
    assert "evil_column" not in profile


def test_merge_empty_payload_returns_current(sqlite_db):
    import models
    models.merge_user_profile("u1", {"name": "A"})
    out = models.merge_user_profile("u1", {"name": None, "location": None})
    assert out["name"] == "A"


# ---------------------------------------------------------------------------
# procedural_memory.workflow_block
# ---------------------------------------------------------------------------

def test_workflow_block_known_case_type():
    block = workflow_block("land")
    assert "Thủ tục đất đai" in block
    assert "B1:" in block
    assert "[Quy trình thủ tục" in block


def test_workflow_block_unknown_case_type_empty():
    assert workflow_block("xyz") == ""
    assert workflow_block(None) == ""
    assert workflow_block("") == ""


def test_case_workflows_cover_expected_types():
    assert set(CASE_WORKFLOWS) == {"inheritance", "land", "marriage", "business", "traffic", "other"}
    for ct, wf in CASE_WORKFLOWS.items():
        assert 4 <= len(wf["steps"]) <= 6, f"{ct} must have 4-6 steps"


# ---------------------------------------------------------------------------
# _get_agent_system_prompt injects workflow block for known case_type
# ---------------------------------------------------------------------------

def test_system_prompt_appends_workflow_block(sqlite_db):
    import models
    models.merge_user_profile("u1", {"case_type": "inheritance"})
    import agent
    prompt = agent._get_agent_system_prompt(history_summary="", user_id="u1")
    assert "Thủ tục thừa kế" in prompt
    assert "B1:" in prompt


def test_system_prompt_no_workflow_block_for_unknown_case(sqlite_db):
    import models
    models.merge_user_profile("u1", {"case_type": "unknown_ct"})
    import agent
    prompt = agent._get_agent_system_prompt(history_summary="", user_id="u1")
    assert "Quy trình thủ tục" not in prompt


def test_system_prompt_no_workflow_block_when_flag_off(sqlite_db, monkeypatch):
    import models
    models.merge_user_profile("u1", {"case_type": "inheritance"})
    monkeypatch.setenv("PROCEDURAL_WORKFLOW_ENABLED", "false")
    import agent
    prompt = agent._get_agent_system_prompt(history_summary="", user_id="u1")
    assert "Thủ tục thừa kế" not in prompt


# ---------------------------------------------------------------------------
# save_episodic_memory_task merges structured fields when flag on
# ---------------------------------------------------------------------------

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


def test_episodic_task_merges_structured_profile(tasks_module, sqlite_db, monkeypatch):
    """STRUCTURED_PROFILE_ENABLED=true -> extract JSON merges into UserProfile."""
    tasks = tasks_module
    monkeypatch.setenv("STRUCTURED_PROFILE_ENABLED", "true")
    monkeypatch.setattr(tasks, "openai_chat_complete", lambda msgs: json.dumps({
        "facts": ["user ten A, sinh 1990, o Ha Noi"],
        "structured": {"name": "Nguyen A", "birth_year": 1990, "location": "Ha Noi",
                       "case_type": "land", "case_summary": "tranh chap dat"},
    }, ensure_ascii=False))
    monkeypatch.setattr(tasks, "get_embedding", lambda text: [0.1, 0.2, 0.3])
    monkeypatch.setattr(tasks, "search_vector", lambda **kw: None)
    import vectorize
    monkeypatch.setattr(vectorize, "add_vector", lambda **kw: None)

    tasks.save_episodic_memory_task(
        "u1", "c1", delta_message={"role": "user", "content": "Toi ten A, sinh 1990, o Ha Noi, tranh chap dat"}
    )
    import models
    profile = models.get_user_profile("u1")
    assert profile is not None
    assert profile["name"] == "Nguyen A"
    assert profile["birth_year"] == 1990
    assert profile["case_type"] == "land"


# ---------------------------------------------------------------------------
# recall_user_memory_tool returns "profile" key when flag on
# ---------------------------------------------------------------------------

def test_recall_tool_returns_profile_when_flag_on(sqlite_db, monkeypatch):
    import models
    models.merge_user_profile("u1", {"name": "A", "case_type": "inheritance"})
    monkeypatch.setenv("STRUCTURED_PROFILE_ENABLED", "true")

    import agent_tool_wrappers as wrappers
    import vectorize
    monkeypatch.setattr(vectorize, "search_vector",
                        lambda **kw: [{"content": "fact", "score": 0.9}])
    from agent_tool_tracking import agent_user_id
    token = agent_user_id.set("u1")
    try:
        out = wrappers.recall_user_memory_tool("như tôi đã kể")
    finally:
        agent_user_id.reset(token)
    data = json.loads(out)
    assert data["status"] == "ok"
    assert "profile" in data
    assert data["profile"]["case_type"] == "inheritance"


def test_recall_tool_omits_profile_when_flag_off(sqlite_db, monkeypatch):
    import models
    models.merge_user_profile("u1", {"name": "A"})
    monkeypatch.setenv("STRUCTURED_PROFILE_ENABLED", "false")
    import agent_tool_wrappers as wrappers
    import vectorize
    monkeypatch.setattr(vectorize, "search_vector",
                        lambda **kw: [{"content": "fact", "score": 0.9}])
    from agent_tool_tracking import agent_user_id
    token = agent_user_id.set("u1")
    try:
        out = wrappers.recall_user_memory_tool("như tôi đã kể")
    finally:
        agent_user_id.reset(token)
    data = json.loads(out)
    assert "profile" not in data