"""Phase 1 unit tests — PEV verify_answer gate.

Covers:
- ``judge_answer`` pure logic: empty / no-sources / supported / partial / unsupported.
- Graph-level recovery loop: unsupported verdict -> rewrite_query -> retrieve ->
  generate -> verify again, capped at VERIFY_MAX_RETRIES, then degrades to
  metacognitive (END) with the response preserved.
"""
import tasks
import verify_answer


# ---- judge_answer pure logic ----

def test_judge_answer_empty_is_unsupported():
    verdict = verify_answer.judge_answer("q", "", [{"content": "x"}])
    assert verdict["verdict"] == "unsupported"
    assert verdict["score"] == 0.0


def test_judge_answer_too_short_is_unsupported():
    verdict = verify_answer.judge_answer("q", "short", [{"content": "x"}])
    assert verdict["verdict"] == "unsupported"


def test_judge_answer_no_sources_skips_to_supported():
    # Non-RAG routes carry no sources -> cannot check citations -> pass.
    verdict = verify_answer.judge_answer("q", "a real answer here", [])
    assert verdict["verdict"] == "supported"
    assert verdict["score"] == 1.0


def test_judge_answer_sources_without_content_skips_to_supported():
    verdict = verify_answer.judge_answer("q", "a real answer here", [{"content": ""}])
    assert verdict["verdict"] == "supported"


def test_judge_answer_supported(monkeypatch):
    class _JR:
        score = 0.9
        rationale = "all claims grounded"
    monkeypatch.setattr(verify_answer, "evaluate_faithfulness", lambda *a, **k: _JR())
    verdict = verify_answer.judge_answer("q", "a real answer here", [{"content": "ctx"}])
    assert verdict["verdict"] == "supported"
    assert verdict["score"] == 0.9


def test_judge_answer_partial(monkeypatch):
    class _JR:
        score = 0.5
        rationale = "some claims"
    monkeypatch.setattr(verify_answer, "evaluate_faithfulness", lambda *a, **k: _JR())
    verdict = verify_answer.judge_answer("q", "a real answer here", [{"content": "ctx"}])
    assert verdict["verdict"] == "partial"


def test_judge_answer_unsupported(monkeypatch):
    class _JR:
        score = 0.1
        rationale = "no claim grounded"
    monkeypatch.setattr(verify_answer, "evaluate_faithfulness", lambda *a, **k: _JR())
    verdict = verify_answer.judge_answer("q", "a real answer here", [{"content": "ctx"}])
    assert verdict["verdict"] == "unsupported"


def test_judge_answer_judge_error_is_unsupported(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("judge down")
    monkeypatch.setattr(verify_answer, "evaluate_faithfulness", boom)
    verdict = verify_answer.judge_answer("q", "a real answer here", [{"content": "ctx"}])
    assert verdict["verdict"] == "unsupported"
    assert "judge_error" in verdict["rationale"]


# ---- graph-level recovery loop ----

def _patch_heavy(monkeypatch):
    """Deterministic routing + retrieval + generation (relevant docs, no infra)."""
    monkeypatch.setattr(tasks, "detect_route", lambda history, q: "legal_rag")
    monkeypatch.setattr(tasks, "follow_up_question", lambda history, q: q)
    monkeypatch.setattr(tasks, "rewrite_query_to_multi_queries", lambda q, num_queries=3: [q])
    monkeypatch.setattr(tasks, "retrieve_with_multi_query_fallback", lambda queries, top_k=4: [{"content": "doc1", "relevance_score": 0.9}])
    monkeypatch.setattr(tasks, "rerank_documents", lambda docs, q, top_n=5: docs)
    monkeypatch.setattr(tasks, "_llm_judge_relevance", lambda q, docs: {})
    monkeypatch.setattr(tasks, "_retrieve_episodic_context", lambda user_id, q: "")
    monkeypatch.setattr(tasks, "gen_doc_prompt", lambda docs: "CTX")
    monkeypatch.setattr(tasks, "vietnamese_llm_chat_complete", lambda msgs: "RAG_ANSWER")
    monkeypatch.setattr(tasks, "rewrite_query_with_context", lambda q, history: q + "_rw")
    tasks.guardrails_manager.initialized = False


def test_verify_loop_retries_then_degrades_to_metacognitive(monkeypatch):
    """Always-unsupported verdict -> recover via rewrite_query twice, then
    degrade (retry_verify >= VERIFY_MAX_RETRIES) -> metacognitive -> END.
    Response must still be returned (graceful degradation, no infinite loop)."""
    _patch_heavy(monkeypatch)
    retrieve_calls = {"n": 0}

    def fake_retrieve(queries, top_k=4):
        retrieve_calls["n"] += 1
        return [{"content": "doc1", "relevance_score": 0.9}]
    monkeypatch.setattr(tasks, "retrieve_with_hybrid_search", fake_retrieve)
    monkeypatch.setattr(tasks, "judge_answer", lambda q, a, s: {"score": 0.1, "rationale": "bad", "verdict": "unsupported"})

    result = tasks.run_chat_graph(history=[], question="q", user_id=None, conversation_id="test-verify-loop")

    # Degraded path still returns a response — never infinite-loops.
    assert result["response"] == "RAG_ANSWER"
    # verify ran twice (retry 1 -> rewrite, retry 2 -> degrade). retrieve ran
    # once per generate pass -> 2 retrieve calls.
    assert retrieve_calls["n"] == 2
    assert result.get("retry_verify", 0) >= tasks.VERIFY_MAX_RETRIES


def test_verify_supported_skips_recovery(monkeypatch):
    """Supported verdict -> straight to metacognitive, no rewrite loop."""
    _patch_heavy(monkeypatch)
    retrieve_calls = {"n": 0}

    def fake_retrieve(queries, top_k=4):
        retrieve_calls["n"] += 1
        return [{"content": "doc1", "relevance_score": 0.9}]
    monkeypatch.setattr(tasks, "retrieve_with_hybrid_search", fake_retrieve)
    monkeypatch.setattr(tasks, "judge_answer", lambda q, a, s: {"score": 0.95, "rationale": "ok", "verdict": "supported"})

    result = tasks.run_chat_graph(history=[], question="q", user_id=None, conversation_id="test-verify-ok")

    assert result["response"] == "RAG_ANSWER"
    assert retrieve_calls["n"] == 1          # no recovery -> single retrieve
    assert result.get("verify_verdict") == "supported"
    assert result.get("retry_verify", 0) == 1