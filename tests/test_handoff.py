"""Phase 3 — supervisor handoff decision tests (no real LLM, no DB)."""
from supervisor import (
    END,
    MAX_HANDOFF_STEPS,
    SPECIALISTS,
    build_supervisor_prompt,
    heuristic_handoff,
    parse_supervisor_decision,
    supervisor_decide,
)


# ---- parse ----


def test_parse_handoff_rag():
    d = parse_supervisor_decision('<handoff next="rag" rationale="cần tra cứu" />')
    assert d == {"next": "rag", "rationale": "cần tra cứu"}


def test_parse_handoff_end():
    d = parse_supervisor_decision('<handoff next="END" rationale="đủ rồi" />')
    assert d["next"] == "end"


def test_parse_handoff_loose():
    d = parse_supervisor_decision("prose next=tool rationale='tính toán' tail")
    assert d == {"next": "tool", "rationale": "tính toán"}


def test_parse_handoff_alias_stop():
    d = parse_supervisor_decision('<handoff next="done" rationale="xong" />')
    assert d["next"] == "end"


def test_parse_handoff_empty():
    assert parse_supervisor_decision("") == {"next": "", "rationale": ""}
    assert parse_supervisor_decision("no tag here") == {"next": "", "rationale": ""}


# ---- heuristic fallback ----


def test_heuristic_tool_needs_lookup_to_rag():
    assert heuristic_handoff("tool", "Tôi cần tra cứu văn bản luật") == "rag"


def test_heuristic_tool_done_to_end():
    assert heuristic_handoff("tool", "Phạt vi phạm là 24 triệu đồng.") == END


def test_heuristic_rag_not_found_to_web():
    assert heuristic_handoff("rag", "Không tìm thấy thông tin về điều này.") == "web"


def test_heuristic_web_needs_lookup_to_tool():
    assert heuristic_handoff("web", "hãy tham khảo văn bản liên quan") == "tool"


def test_heuristic_chat_always_end():
    assert heuristic_handoff("chat", "bất kỳ câu trả lời nào") == END


def test_heuristic_empty_answer_end():
    assert heuristic_handoff("tool", "") == END


# ---- supervisor_decide ----


def test_decide_no_llm_uses_heuristic():
    d = supervisor_decide("q", "tool", "Tôi cần tra cứu văn bản", [])
    assert d["next"] == "rag"
    assert d["source"] == "heuristic"


def test_decide_llm_used_when_parseable():
    calls = []

    def fake_llm(prompt):
        calls.append(prompt)
        return '<handoff next="web" rationale="cần thông tin mới" />'

    d = supervisor_decide("q", "rag", "Không tìm thấy", [], llm_call=fake_llm)
    assert d["next"] == "web"
    assert d["source"] == "llm"
    assert len(calls) == 1


def test_decide_llm_unparseable_falls_back():
    def fake_llm(prompt):
        return "tôi không hiểu, hãy tra cứu thêm"

    d = supervisor_decide("q", "tool", "Tôi cần tra cứu văn bản", [], llm_call=fake_llm)
    assert d["source"] == "heuristic"
    assert d["next"] == "rag"


def test_decide_llm_raises_falls_back():
    def fake_llm(prompt):
        raise RuntimeError("groq down")

    d = supervisor_decide("q", "rag", "Không tìm thấy thông tin", [], llm_call=fake_llm)
    assert d["source"] == "heuristic"
    assert d["next"] == "web"


def test_decide_step_guard_forces_end():
    def fake_llm(prompt):
        return '<handoff next="rag" rationale="tiếp tục" />'

    d = supervisor_decide("q", "tool", "Tôi cần tra cứu", [], steps_taken=MAX_HANDOFF_STEPS, llm_call=fake_llm)
    assert d["next"] == "end"
    assert d["source"] == "guard"


def test_decide_end_from_llm():
    def fake_llm(prompt):
        return '<handoff next="END" rationale="đã đủ" />'

    d = supervisor_decide("q", "tool", "xong", [], llm_call=fake_llm)
    assert d["next"] == "end"
    assert d["source"] == "llm"


def test_build_supervisor_prompt_contains_fields():
    p = build_supervisor_prompt("câu hỏi", "tool", "đáp án", [{"specialist": "rag", "goal": "g"}])
    assert "câu hỏi" in p
    assert "tool" in p
    assert "đáp án" in p
    assert "rag: g" in p


def test_next_values_are_valid():
    valid = set(SPECIALISTS) | {"end"}
    for cur in SPECIALISTS:
        for ans in ["Tôi cần tra cứu văn bản", "Không tìm thấy thông tin", "xong", ""]:
            d = supervisor_decide("q", cur, ans, [])
            assert d["next"] in valid