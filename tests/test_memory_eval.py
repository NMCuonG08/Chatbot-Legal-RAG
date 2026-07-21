"""P6a/P6d — Memory eval golden regression suite (MLOps/LLMOps gate).

Consolidates the cross-phase invariants into ONE runnable suite so CI can gate
on `pytest tests/test_memory_eval.py`. Each test asserts a memory-architecture
contract; a failure = a regression reintroduced a FLAW.

Invariants:
- FLAW 1: cross-worker short-term consistency (Redis source of truth)
- FLAW 2: episodic extraction never ingests bot legal text (pollution = 0)
- FLAW 3: RAG system prompt not contaminated by stale episodic facts
- P4a: structured profile merge is idempotent (no null overwrite)
- P4b: procedural workflow block correct per case_type, empty for unknown
- P5: working-memory slots carried in graph state + tool_budget decrements
- P6c: Prometheus counters increment on memory ops (no-op safe when prom absent)

This suite reuses the proven stub patterns from the per-phase test files.
"""
from __future__ import annotations

import json
import sys
import types

import fakeredis
import pytest


# ---------------------------------------------------------------------------
# Shared stubs
# ---------------------------------------------------------------------------

@pytest.fixture
def tasks_module(monkeypatch):
    for name in ("agent", "verify_answer", "metacognitive", "rlhf_store",
                 "summarizer", "guardrails_manager"):
        mod = types.ModuleType(name)
        if name == "agent":
            mod.ai_agent_handle = lambda *a, **k: None
            mod.clear_user_runtime_caches = lambda *a, **k: None
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


def _sample_counter(counter) -> float:
    """Read a prometheus counter's current value (0 for the no-op stub)."""
    try:
        for metric in counter.collect():
            for s in metric.samples:
                # Pick the total sample (skip the *_created gauge prom adds).
                if s.name.endswith("_total") and not s.labels:
                    return float(s.value)
        return 0.0
    except Exception:
        return 0.0


def _sample_counter_label(counter, label: str) -> float:
    try:
        for metric in counter.collect():
            for s in metric.samples:
                if s.labels.get("result") == label:
                    return float(s.value)
        return 0.0
    except Exception:
        return 0.0


# ===========================================================================
# FLAW 1 — cross-worker short-term consistency
# ===========================================================================

def test_flaw1_cross_worker_summary_consistency():
    """Two workers sharing Redis see the same rolling summary (MLOps-critical)."""
    import memory_short_term as mst
    server = fakeredis.FakeServer()
    a = fakeredis.FakeStrictRedis(server=server, decode_responses=True)
    b = fakeredis.FakeStrictRedis(server=server, decode_responses=True)
    mst._redis_client = a
    mst._REDIS_ENABLED = True
    mst._summary_cache.clear()
    mst.set_rolling_summary("u1", "c1", "FACTS:\n- nam, 1990")
    mst._redis_client = b  # worker B, fresh local cache
    mst._summary_cache.clear()
    assert mst.get_rolling_summary("u1", "c1") == "FACTS:\n- nam, 1990"
    assert mst.get_summarized_count("u1", "c1") == 0


# ===========================================================================
# FLAW 2 — episodic pollution = 0
# ===========================================================================

def test_flaw2_no_bot_legal_text_ingested(tasks_module, monkeypatch):
    """Bot law text in delta must not reach the extraction prompt (pollution=0)."""
    tasks = tasks_module
    captured = {}
    def fake_llm(msgs):
        captured["prompt"] = msgs[1]["content"]
        return json.dumps({"facts": ["user o Ha Noi"], "structured": {}}, ensure_ascii=False)
    monkeypatch.setattr(tasks, "openai_chat_complete", fake_llm)
    monkeypatch.setattr(tasks, "get_embedding", lambda t: [0.1])
    monkeypatch.setattr(tasks, "search_vector", lambda **k: None)
    import vectorize
    monkeypatch.setattr(vectorize, "add_vector", lambda **k: None)
    import models
    monkeypatch.setattr(models, "save_user_episode", lambda uid, txt: None)
    tasks.save_episodic_memory_task(
        "u1", "c1", delta_message={"role": "user",
                                   "content": "Theo Điều 100 Bộ luật dân sự thừa kế. Tôi ở Hà Nội."}
    )
    assert "Bộ luật dân sự" not in captured["prompt"]
    assert "Hà Nội" in captured["prompt"]


# ===========================================================================
# FLAW 3 — RAG context not contaminated
# ===========================================================================

def test_flaw3_rag_prompt_clean_of_stale_fact(tasks_module, monkeypatch):
    """Inheritance fact must not bleed into an unrelated vehicle question."""
    tasks = tasks_module
    tasks._retrieve_episodic_context = lambda uid, q: "- Tôi có nhà đất thừa kế"
    captured = {}
    monkeypatch.setattr(tasks, "vietnamese_llm_chat_complete",
                        lambda msgs: (captured.__setitem__("m", msgs), "ok")[1])
    monkeypatch.setattr(tasks, "rewrite_query_to_multi_queries", lambda q, num_queries=3: [q])
    monkeypatch.setattr(tasks, "retrieve_with_hybrid_search", lambda qs, top_k=4: [])
    monkeypatch.setattr(tasks, "retrieve_with_multi_query_fallback", lambda qs, top_k=4: [])
    monkeypatch.setattr(tasks, "rerank_documents", lambda d, q, top_n=5: d)
    monkeypatch.setattr(tasks, "blend_hybrid_rerank", lambda d: d)
    monkeypatch.setattr(tasks, "gen_doc_prompt", lambda d: "DOC")
    try:
        tasks.guardrails_manager.initialized = False
    except Exception:
        pass
    tasks.generate_rag_answer(history=[], question="đăng ký xe máy ở đâu", user_id="u1")
    assert "thừa kế" not in captured["m"][0]["content"]


# ===========================================================================
# P4a — structured profile merge idempotency
# ===========================================================================

def test_p4a_merge_idempotent_no_null_overwrite(sqlite_db):
    import models
    models.merge_user_profile("u1", {"name": "A", "birth_year": 1990, "case_type": "land"})
    models.merge_user_profile("u1", {"name": None, "birth_year": None, "location": "Hà Nội"})
    p = models.get_user_profile("u1")
    assert p["name"] == "A"
    assert p["birth_year"] == 1990
    assert p["case_type"] == "land"
    assert p["location"] == "Hà Nội"


# ===========================================================================
# P4b — procedural workflow correct per case_type, empty for unknown
# ===========================================================================

def test_p4b_workflow_correct_per_case_type():
    from procedural_memory import workflow_block, CASE_WORKFLOWS
    for ct in ("inheritance", "land", "marriage", "business", "traffic", "other"):
        block = workflow_block(ct)
        assert block, f"{ct} must have a workflow block"
        assert "Quy trình thủ tục" in block
        assert CASE_WORKFLOWS[ct]["title"] in block
    assert workflow_block("xyz") == ""
    assert workflow_block(None) == ""


# ===========================================================================
# P5 — working-memory state slots + tool_budget
# ===========================================================================

def test_p5_state_carries_entities_and_decrements(tasks_module, monkeypatch):
    tasks = tasks_module
    state: tasks.ChatGraphState = {
        "question": "q",
        "active_entities": {"user_name": "A"},
        "current_intent": "inheritance",
        "tool_budget": 2,
        "tool_calls_made": 0,
    }
    assert state["active_entities"]["user_name"] == "A"
    state.update(tasks.increment_tool_calls(state, note="t1"))
    assert state["tool_calls_made"] == 1
    assert tasks.tool_budget_exhausted(state) is False
    state.update(tasks.increment_tool_calls(state, note="t2"))
    assert state["tool_calls_made"] == 2
    assert tasks.tool_budget_exhausted(state) is True


# ===========================================================================
# P6c — Prometheus counters increment on memory ops (no-op safe)
# ===========================================================================

def test_p6c_counters_increment(monkeypatch):
    """Short-term hit/miss + episodic extraction counters increment monotonically."""
    import memory_metrics as mm
    import memory_short_term as mst
    server = fakeredis.FakeServer()
    client = fakeredis.FakeStrictRedis(server=server, decode_responses=True)
    monkeypatch.setattr(mst, "_REDIS_ENABLED", True)
    monkeypatch.setattr(mst, "_redis_client", client)
    mst._summary_cache.clear()

    before_miss = _sample_counter(mm.SHORT_TERM_MISSES)
    before_hit = _sample_counter(mm.SHORT_TERM_HITS)
    mst.get_rolling_summary("u1", "c1")  # miss
    mst.set_rolling_summary("u1", "c1", "FACTS:\n- x")
    mst._summary_cache.clear()  # force Redis read
    mst.get_rolling_summary("u1", "c1")  # hit
    assert _sample_counter(mm.SHORT_TERM_MISSES) >= before_miss + 1
    assert _sample_counter(mm.SHORT_TERM_HITS) >= before_hit + 1

    before_dup = _sample_counter_label(mm.EPISODIC_EXTRACTIONS, "skipped_duplicate")
    mm.inc_episodic_extraction("skipped_duplicate")
    assert _sample_counter_label(mm.EPISODIC_EXTRACTIONS, "skipped_duplicate") >= before_dup + 1


# ===========================================================================
# Regression gate meta — assert the suite keeps covering every flaw
# ===========================================================================

def test_suite_covers_all_flaws():
    """Guard: the golden suite must keep covering FLAW 1-3 + P4/P5/P6."""
    import inspect
    names = [n for n, _ in inspect.getmembers(sys.modules[__name__], inspect.isfunction)]
    required = ["test_flaw1_", "test_flaw2_", "test_flaw3_", "test_p4a_",
                "test_p4b_", "test_p5_", "test_p6c_"]
    for prefix in required:
        assert any(n.startswith(prefix) for n in names), f"missing coverage: {prefix}"