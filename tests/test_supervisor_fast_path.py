"""Audit 2.1 — supervisor rule-based fast path.

When the planner already produced a fixed plan, the supervisor must advance to
the next planned specialist via a deterministic transition and SKIP the LLM
supervisor call at intermediate steps (saving 1.2-2.0s per handoff). It falls
through to the LLM/heuristic only when: no plan, current not in plan, or the
answer signals failure (not-found / needs-lookup markers).
"""
from supervisor import END, SPECIALISTS, plan_fast_path, supervisor_decide


# ---------------------------------------------------------------------------
# plan_fast_path — deterministic next specialist from a fixed plan
# ---------------------------------------------------------------------------
def test_fast_path_advances_to_next_planned_specialist():
    plan = [
        {"specialist": "rag", "goal": "tra cứu văn bản"},
        {"specialist": "tool", "goal": "tính toán phạt"},
    ]
    assert plan_fast_path("rag", plan, answer="Điều 15 quy định...") == "tool"


def test_fast_path_returns_end_when_current_is_last_planned():
    plan = [{"specialist": "rag", "goal": "tra cứu"}]
    assert plan_fast_path("rag", plan, answer="kết quả đầy đủ") == END


def test_fast_path_skips_non_specialist_plan_steps():
    # verify_answer is a graph node, not a supervisor handoff target.
    plan = [
        {"specialist": "rag", "goal": "tra cứu"},
        {"specialist": "verify_answer", "goal": "kiểm chứng"},
        {"specialist": "tool", "goal": "tính toán"},
    ]
    assert plan_fast_path("rag", plan, answer="đã tra cứu") == "tool"


def test_fast_path_none_when_no_plan():
    assert plan_fast_path("rag", [], answer="x") is None
    assert plan_fast_path("rag", None, answer="x") is None


def test_fast_path_none_when_current_not_in_plan():
    plan = [{"specialist": "tool", "goal": "tính"}]
    assert plan_fast_path("web", plan, answer="kết quả") is None


def test_fast_path_none_on_not_found_marker():
    plan = [{"specialist": "rag", "goal": "tra cứu"}, {"specialist": "web", "goal": "web"}]
    assert plan_fast_path("rag", plan, answer="Không tìm thấy thông tin về vấn đề này.") is None


def test_fast_path_none_on_needs_lookup_marker():
    plan = [{"specialist": "rag", "goal": "tra cứu"}, {"specialist": "tool", "goal": "tính"}]
    assert plan_fast_path("rag", plan, answer="cần tra cứu văn bản chi tiết hơn") is None


def test_fast_path_normalizes_aliases():
    # planner may emit "retrieve" (alias for rag) and "agent_tools" (alias tool).
    plan = [{"specialist": "retrieve", "goal": "tra cứu"}, {"specialist": "agent_tools", "goal": "tính"}]
    assert plan_fast_path("rag", plan, answer="ok") == "tool"


# ---------------------------------------------------------------------------
# supervisor_decide — fast path skips the LLM call entirely
# ---------------------------------------------------------------------------
def test_supervisor_decide_uses_fast_path_without_llm_call():
    calls = []

    def llm_call(prompt):
        calls.append(prompt)
        return '{"next": "chat", "rationale": "không dùng"}'

    plan = [{"specialist": "rag", "goal": "tra cứu"}, {"specialist": "tool", "goal": "tính"}]
    decision = supervisor_decide("question", "rag", "kết quả tra cứu đầy đủ", plan, llm_call=llm_call)
    assert decision["next"] == "tool"
    assert decision["source"] == "fast-path"
    assert calls == []  # LLM never invoked


def test_supervisor_decide_fast_path_end_when_plan_complete():
    def llm_call(prompt):
        raise AssertionError("LLM should not be called on a completed plan")

    plan = [{"specialist": "rag", "goal": "tra cứu"}]
    decision = supervisor_decide("question", "rag", "kết quả đầy đủ", plan, llm_call=llm_call)
    assert decision["next"] == END
    assert decision["source"] == "fast-path"


def test_supervisor_decide_falls_through_to_llm_on_failure_marker():
    calls = []

    def llm_call(prompt):
        calls.append(prompt)
        return '{"next": "web", "rationale": "rag không có -> web"}'

    plan = [{"specialist": "rag", "goal": "tra cứu"}, {"specialist": "tool", "goal": "tính"}]
    decision = supervisor_decide("question", "rag", "Không tìm thấy thông tin.", plan, llm_call=llm_call)
    assert decision["next"] == "web"
    assert decision["source"] == "llm"
    assert len(calls) == 1  # fast path bypassed -> LLM consulted


def test_supervisor_decide_falls_through_to_heuristic_when_no_plan():
    decision = supervisor_decide("question", "rag", "Không tìm thấy thông tin.", [], llm_call=None)
    assert decision["source"] == "heuristic"
    assert decision["next"] == "web"  # heuristic: rag not-found -> web