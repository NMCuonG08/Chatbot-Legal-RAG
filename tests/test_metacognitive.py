"""Phase 2 unit tests — metacognitive escalation gate.

Covers:
- ``classify_stakes`` tiering (high criminal / medium civil / low).
- ``should_escalate`` decision matrix (high always, medium only below
  confidence threshold, low never).
- ``build_escalation`` payload shape.
- E2E: graph prepends the lawyer-escalation prefix on a criminal question
  with a low-confidence (unsupported) verify verdict, and does NOT prepend it
  on a low-stakes high-confidence answer.
"""
import metacognitive
import tasks


# ---- classify_stakes ----

HIGH_FIXTURES = [
    "tôi bị khởi tố hình sự về tội trộm cắp, cần bào chữa thế nào?",
    "bị cáo giết người có thể được giảm tội danh không?",
    "mua bán ma túy khối lượng bao nhiêu thì truy tố hình sự?",
    "cố ý gây thương tích trong khi chống người thi hành công vụ?",
    "đối mặt với án tử hình, phúc thẩm hình sự có cứu được không?",
]

MEDIUM_FIXTURES = [
    "tôi muốn kiện người hàng xóm vì tranh chấp ranh đất.",
    "ly hôn thì ai được nuôi con và chia tài sản?",
    "hợp đồng vay tiền, đòi nợ không trả thì làm sao?",
    "bồi thường thiệt hại do ô nhiễm môi trường bao nhiêu?",
    "phúc thẩm dân sự về một tranh chấp hợp đồng chuyển nhượng.",
]

LOW_FIXTURES = [
    "điều 35 bộ luật dân sự nói về vấn đề gì?",
    "đăng ký kết hôn cần thủ tục và giấy tờ gì?",
    "khái niệm pháp nhân theo luật là gì?",
    "công chứng giấy tờ ở đâu?",
    "thủ tục đăng ký doanh nghiệp gồm những bước nào?",
    "di chúc hợp pháp phải đáp ứng điều kiện gì?",
    "luật đất đai quy định thế nào về quyền sử dụng đất?",
    "căn cứ pháp lý của chế định sở hữu là gì?",
    "độ tuổi lao động tối thiểu theo bộ luật lao động?",
    "thời gian mang thai được nghỉ thai sản bao nhiêu ngày?",
]


def test_classify_stakes_high():
    for q in HIGH_FIXTURES:
        assert metacognitive.classify_stakes(q) == "high", q


def test_classify_stakes_medium():
    for q in MEDIUM_FIXTURES:
        assert metacognitive.classify_stakes(q) == "medium", q


def test_classify_stakes_low():
    for q in LOW_FIXTURES:
        assert metacognitive.classify_stakes(q) == "low", q


def test_classify_stakes_high_wins_over_medium():
    # A criminal topic present alongside a medium keyword -> high.
    q = "kiện tụp bào chữa về tội trộm cắp trong tranh chấp tài sản"
    assert metacognitive.classify_stakes(q) == "high"


def test_classify_stakes_empty_is_low():
    assert metacognitive.classify_stakes("") == "low"
    assert metacognitive.classify_stakes(None) == "low"


# ---- should_escalate ----

def test_should_escalate_high_always():
    assert metacognitive.should_escalate("high", 0.99) is True
    assert metacognitive.should_escalate("high", 0.0) is True


def test_should_escalate_low_never():
    assert metacognitive.should_escalate("low", 0.1) is False
    assert metacognitive.should_escalate("low", 0.9) is False


def test_should_escalate_medium_threshold():
    # Below threshold -> escalate; at/above -> pass.
    assert metacognitive.should_escalate("medium", 0.0) is True
    assert metacognitive.should_escalate("medium", 0.5) is True
    assert metacognitive.should_escalate("medium", 0.61) is False


def test_should_escalate_invalid_confidence_treated_as_low():
    assert metacognitive.should_escalate("medium", None) is True
    assert metacognitive.should_escalate("medium", "oops") is True


# ---- build_escalation ----

def test_build_escalation_payload_shape():
    payload = metacognitive.build_escalation("tôi bị truy tố hình sự", 0.2)
    assert set(payload) == {"stakes", "confidence", "escalate"}
    assert payload["stakes"] == "high"
    assert payload["confidence"] == 0.2
    assert payload["escalate"] is True


def test_build_escalation_invalid_confidence_coerced():
    payload = metacognitive.build_escalation("tranh chấp đất đai", None)
    assert payload["confidence"] == 0.0
    assert payload["escalate"] is True


# ---- E2E graph: prefix prepended on escalation ----

def _patch_heavy(monkeypatch, *, verify_verdict, verify_score):
    monkeypatch.setattr(tasks, "detect_route", lambda history, q: "legal_rag")
    monkeypatch.setattr(tasks, "follow_up_question", lambda history, q: q)
    monkeypatch.setattr(tasks, "rewrite_query_to_multi_queries", lambda q, num_queries=3: [q])
    monkeypatch.setattr(tasks, "retrieve_with_hybrid_search", lambda queries, top_k=4: [{"content": "doc1", "relevance_score": 0.9}])
    monkeypatch.setattr(tasks, "retrieve_with_multi_query_fallback", lambda queries, top_k=4: [{"content": "doc1", "relevance_score": 0.9}])
    monkeypatch.setattr(tasks, "rerank_documents", lambda docs, q, top_n=5: docs)
    monkeypatch.setattr(tasks, "_llm_judge_relevance", lambda q, docs: {})
    monkeypatch.setattr(tasks, "_retrieve_episodic_context", lambda user_id, q: "")
    monkeypatch.setattr(tasks, "gen_doc_prompt", lambda docs: "CTX")
    monkeypatch.setattr(tasks, "vietnamese_llm_chat_complete", lambda msgs: "RAG_ANSWER")
    monkeypatch.setattr(tasks, "rewrite_query_with_context", lambda q, history: q + "_rw")
    monkeypatch.setattr(tasks, "judge_answer", lambda q, a, s: {"score": verify_score, "rationale": "mock", "verdict": verify_verdict})
    tasks.guardrails_manager.initialized = False


def test_graph_escalates_on_high_stakes_low_confidence(monkeypatch):
    """Criminal question + unsupported verify -> metacognitive prepends prefix."""
    _patch_heavy(monkeypatch, verify_verdict="unsupported", verify_score=0.1)

    result = tasks.run_chat_graph(
        history=[], question="tôi bị truy tố hình sự về tội trộm cắp, bào chữa thế nào?",
        user_id=None, conversation_id="test-meta-escalate",
    )

    assert result["response"].startswith(metacognitive.ESCALATION_PREFIX)
    assert "RAG_ANSWER" in result["response"]   # original answer preserved below prefix


def test_graph_no_escalation_on_low_stakes_high_confidence(monkeypatch):
    """Low-stakes question + supported verify -> no prefix."""
    _patch_heavy(monkeypatch, verify_verdict="supported", verify_score=0.95)

    result = tasks.run_chat_graph(
        history=[], question="điều 35 bộ luật dân sự nói về vấn đề gì?",
        user_id=None, conversation_id="test-meta-noop",
    )

    assert result["response"] == "RAG_ANSWER"
    assert metacognitive.ESCALATION_PREFIX not in result["response"]