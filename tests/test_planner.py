"""Phase 3 — planner parse/validate tests (no LLM, no DB)."""
from planner import (
    MAX_PLAN_STEPS,
    SPECIALISTS,
    build_plan_prompt,
    fallback_plan,
    parse_plan,
    validate_plan,
)


def test_parse_plan_two_steps():
    text = '<plan>\n<step specialist="tool" goal="tính trợ cấp thôi việc" />\n<step specialist="rag" goal="dẫn điều luật áp dụng" />\n</plan>'
    plan = parse_plan(text)
    assert plan == [
        {"specialist": "tool", "goal": "tính trợ cấp thôi việc"},
        {"specialist": "rag", "goal": "dẫn điều luật áp dụng"},
    ]


def test_parse_plan_single_step():
    plan = parse_plan('<step specialist="chat" goal="chào hỏi" />')
    assert plan == [{"specialist": "chat", "goal": "chào hỏi"}]


def test_parse_plan_ignores_unknown_specialist():
    text = '<step specialist="tool" goal="a" /><step specialist="unknown" goal="b" /><step specialist="rag" goal="c" />'
    plan = parse_plan(text)
    assert [s["specialist"] for s in plan] == ["tool", "rag"]


def test_parse_plan_tolerates_aliases():
    text = '<step specialist="agent_tools" goal="x" /><step specialist="legal_rag" goal="y" />'
    plan = parse_plan(text)
    assert [s["specialist"] for s in plan] == ["tool", "rag"]


def test_parse_plan_tolerates_loose_quotes():
    text = "some prose\nspecialist='web' goal='tim web moi' \nmore"
    plan = parse_plan(text)
    assert plan == [{"specialist": "web", "goal": "tim web moi"}]


def test_parse_plan_empty_returns_empty():
    assert parse_plan("") == []
    assert parse_plan("no plan here at all") == []


def test_parse_plan_single_quotes():
    plan = parse_plan("<step specialist='tool' goal='x' />")
    assert plan == [{"specialist": "tool", "goal": "x"}]


def test_validate_plan_caps_at_max_steps():
    plan = [{"specialist": "rag", "goal": str(i)} for i in range(MAX_PLAN_STEPS + 3)]
    out = validate_plan(plan)
    assert len(out) == MAX_PLAN_STEPS


def test_validate_plan_dedupes_consecutive_identical():
    plan = [
        {"specialist": "rag", "goal": "g"},
        {"specialist": "rag", "goal": "g"},
        {"specialist": "tool", "goal": "t"},
    ]
    out = validate_plan(plan)
    assert out == [{"specialist": "rag", "goal": "g"}, {"specialist": "tool", "goal": "t"}]


def test_validate_plan_empty():
    assert validate_plan([]) == []


def test_fallback_plan_maps_routes():
    assert fallback_plan("legal_rag") == [{"specialist": "rag", "goal": "xử lý theo route phân loại"}]
    assert fallback_plan("agent_tools")[0]["specialist"] == "tool"
    assert fallback_plan("web_search")[0]["specialist"] == "web"
    assert fallback_plan("general_chat")[0]["specialist"] == "chat"
    assert fallback_plan("unknown")[0]["specialist"] == "chat"


def test_build_plan_prompt_contains_question_and_limits():
    p = build_plan_prompt("tính phạt vi phạm", "người dùng hỏi về hợp đồng")
    assert "tính phạt vi phạm" in p
    assert str(MAX_PLAN_STEPS) in p
    assert "specialist" in p


def test_specialists_set():
    assert set(SPECIALISTS) == {"rag", "tool", "web", "chat"}