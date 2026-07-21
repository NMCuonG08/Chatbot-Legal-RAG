"""P3 — RAG context no longer contaminated by episodic auto-inject (FLAW 3 fix).

Covers:
- _episodic_background_block returns "" by default (auto-inject OFF)
- RAG system prompt (generate_rag_answer + CRAG generate_node) contains NO
  stale user fact when the question is unrelated (inheritance fact -> vehicle Q)
- rollback: RAG_AUTO_INJECT_DISABLED=false re-enables the legacy block
- recall_user_memory_tool threshold 0.65 rejects a 0.55 match
"""
from __future__ import annotations

import json
import sys
import types

import pytest


@pytest.fixture
def tasks_module(monkeypatch):
    """Import tasks with heavy deps stubbed (same pattern as test_canary_shadow)."""
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


# ---------------------------------------------------------------------------
# _episodic_background_block gating
# ---------------------------------------------------------------------------

def test_background_block_empty_by_default(tasks_module):
    """FLAW 3: auto-inject OFF by default -> no contamination."""
    tasks_module._retrieve_episodic_context = lambda uid, q: "- Tôi có nhà đất thừa kế từ cha"
    assert tasks_module._episodic_background_block("u1", "tôi muốn đăng ký xe máy") == ""


def test_background_block_legacy_rollback(tasks_module, monkeypatch):
    """RAG_AUTO_INJECT_DISABLED=false -> legacy block reappears (rollback path)."""
    monkeypatch.setattr(tasks_module, "_RAG_AUTO_INJECT_DISABLED", False)
    tasks_module._retrieve_episodic_context = lambda uid, q: "- fact thừa kế"
    block = tasks_module._episodic_background_block("u1", "anything")
    assert "Ngữ cảnh phụ" in block
    assert "fact thừa kế" in block


# ---------------------------------------------------------------------------
# generate_rag_answer system prompt has NO episodic fact (contamination check)
# ---------------------------------------------------------------------------

def _stub_rag_chain(tasks, monkeypatch, capture):
    # generate_rag_answer calls vietnamese_llm_chat_complete (not openai_chat_complete).
    monkeypatch.setattr(tasks, "vietnamese_llm_chat_complete",
                        lambda msgs: (capture.__setitem__("m", msgs), "ok")[1])
    monkeypatch.setattr(tasks, "rewrite_query_to_multi_queries", lambda q, num_queries=3: [q])
    monkeypatch.setattr(tasks, "retrieve_with_hybrid_search", lambda qs, top_k=4: [])
    monkeypatch.setattr(tasks, "retrieve_with_multi_query_fallback", lambda qs, top_k=4: [])
    monkeypatch.setattr(tasks, "rerank_documents", lambda docs, q, top_n=5: docs)
    monkeypatch.setattr(tasks, "blend_hybrid_rerank", lambda docs: docs)
    monkeypatch.setattr(tasks, "gen_doc_prompt", lambda docs: "DOC_CTX")
    # Skip the NeMo guardrails output check in unit tests.
    try:
        tasks.guardrails_manager.initialized = False
    except Exception:
        pass


def test_rag_answer_system_prompt_clean(tasks_module, monkeypatch):
    """Inheritance fact must NOT bleed into a vehicle question's system prompt."""
    tasks = tasks_module
    tasks._retrieve_episodic_context = lambda uid, q: "- Tôi có nhà đất thừa kế từ cha"
    captured = {}
    _stub_rag_chain(tasks, monkeypatch, captured)
    tasks.generate_rag_answer(history=[], question="Tôi muốn đăng ký xe máy ở Hà Nội", user_id="u1")
    sys_prompt = captured["m"][0]["content"]
    assert "thừa kế" not in sys_prompt
    assert "Ngữ cảnh phụ" not in sys_prompt


def test_rag_answer_legacy_rollback_injects(tasks_module, monkeypatch):
    """Rollback flag on -> system prompt DOES carry the episodic block."""
    tasks = tasks_module
    monkeypatch.setattr(tasks, "_RAG_AUTO_INJECT_DISABLED", False)
    tasks._retrieve_episodic_context = lambda uid, q: "- fact thừa kế"
    captured = {}
    _stub_rag_chain(tasks, monkeypatch, captured)
    tasks.generate_rag_answer(history=[], question="đăng ký xe máy", user_id="u1")
    assert "fact thừa kế" in captured["m"][0]["content"]


# ---------------------------------------------------------------------------
# recall_user_memory_tool threshold 0.65 rejects weak match
# ---------------------------------------------------------------------------

def test_recall_tool_threshold_rejects_weak_match(tasks_module, monkeypatch):
    """A 0.55-score 'fact' must NOT be recalled (threshold raised 0.5 -> 0.65)."""
    import agent_tool_wrappers as wrappers

    captured = {}
    def fake_search(**kw):
        captured["threshold"] = kw.get("score_threshold")
        # Simulate Qdrant: threshold filters out sub-threshold hits.
        if kw.get("score_threshold", 0) <= 0.55:
            return [{"content": "fact thừa kế", "score": 0.55}]
        return []  # 0.65 threshold rejects the 0.55 hit
    # The tool imports search_vector lazily from vectorize -> patch source.
    import vectorize
    monkeypatch.setattr(vectorize, "search_vector", fake_search)
    monkeypatch.setenv("STRUCTURED_PROFILE_ENABLED", "false")

    from agent_tool_tracking import agent_user_id
    token = agent_user_id.set("u1")
    try:
        out = wrappers.recall_user_memory_tool("như tôi đã kể về thừa kế")
    finally:
        agent_user_id.reset(token)
    data = json.loads(out)
    assert data["status"] == "no_match"
    assert captured["threshold"] == 0.65