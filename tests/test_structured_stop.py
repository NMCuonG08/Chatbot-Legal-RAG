"""Structured stop-condition tests (criterion #1+#2: structured JSON terminal).

Marker: unit. Validates the JSON-primary / regex-fallback stop+plan parsers:

- ``llm_json.extract_json`` robust to fences, prose, nested braces in strings.
- ``supervisor.parse_supervisor_decision`` parses a JSON handoff object and
  schema-validates ``next``; falls back to the legacy ``<handoff>`` tag.
- ``planner.parse_plan`` parses a JSON ``{"steps":[...]}`` object and
  schema-validates each step; falls back to the legacy ``<step>`` tag.

The legacy regex paths stay covered by test_handoff.py / test_planner.py; this
file covers the NEW structured-JSON contract added in the stop-condition fix.
"""
import pytest

pytestmark = pytest.mark.unit


# ---- llm_json.extract_json ----


def test_extract_json_bare_object():
    from llm_json import extract_json

    assert extract_json('{"next": "rag", "rationale": "x"}') == {
        "next": "rag",
        "rationale": "x",
    }


def test_extract_json_fenced():
    from llm_json import extract_json

    text = 'prose\n```json\n{"next": "END"}\n```\nmore'
    assert extract_json(text) == {"next": "END"}


def test_extract_json_prose_around():
    from llm_json import extract_json

    text = 'Sure! Here is the plan:\n{"steps": [{"specialist": "rag", "goal": "g"}]}\nDone.'
    obj = extract_json(text)
    assert obj["steps"][0]["specialist"] == "rag"


def test_extract_json_braces_inside_strings():
    from llm_json import extract_json

    # a "}" inside a string value must not close the scan early
    text = '{"rationale": "không } phải dấu đóng", "next": "rag"}'
    assert extract_json(text)["next"] == "rag"


def test_extract_json_none_when_no_json():
    from llm_json import extract_json

    assert extract_json("just prose, no json") is None
    assert extract_json("") is None


# ---- supervisor.parse_supervisor_decision (JSON primary) ----


def test_supervisor_parses_json_handoff():
    from supervisor import parse_supervisor_decision

    d = parse_supervisor_decision('{"next": "rag", "rationale": "cần tra cứu"}')
    assert d == {"next": "rag", "rationale": "cần tra cứu"}


def test_supervisor_parses_json_fenced_end():
    from supervisor import parse_supervisor_decision

    d = parse_supervisor_decision('```json\n{"next": "END", "rationale": "đủ"}\n```')
    assert d["next"] == "end"
    assert d["rationale"] == "đủ"


def test_supervisor_json_alias_normalized():
    from supervisor import parse_supervisor_decision

    d = parse_supervisor_decision('{"next": "done", "rationale": "xong"}')
    assert d["next"] == "end"


def test_supervisor_json_invalid_next_falls_back_to_regex():
    from supervisor import parse_supervisor_decision

    # JSON present but next not a specialist/END -> fall back to legacy tag
    d = parse_supervisor_decision('{"next": "bogo"} <handoff next="web" rationale="cần mới" />')
    assert d == {"next": "web", "rationale": "cần mới"}


def test_supervisor_legacy_tag_still_works():
    from supervisor import parse_supervisor_decision

    d = parse_supervisor_decision('<handoff next="tool" rationale="tính" />')
    assert d == {"next": "tool", "rationale": "tính"}


def test_supervisor_decide_uses_json_llm():
    from supervisor import supervisor_decide

    def fake_llm(prompt):
        return '{"next": "web", "rationale": "cần thông tin mới"}'

    d = supervisor_decide("q", "rag", "Không tìm thấy", [], llm_call=fake_llm)
    assert d["next"] == "web"
    assert d["source"] == "llm"


# ---- planner.parse_plan (JSON primary) ----


def test_planner_parses_json_plan():
    from planner import parse_plan

    text = '{"steps": [{"specialist": "tool", "goal": "tính trợ cấp"}, {"specialist": "rag", "goal": "dẫn điều"}]}'
    assert parse_plan(text) == [
        {"specialist": "tool", "goal": "tính trợ cấp"},
        {"specialist": "rag", "goal": "dẫn điều"},
    ]


def test_planner_parses_json_fenced():
    from planner import parse_plan

    text = '```json\n{"steps": [{"specialist": "chat", "goal": "chào"}]}\n```'
    assert parse_plan(text) == [{"specialist": "chat", "goal": "chào"}]


def test_planner_json_drops_invalid_step():
    from planner import parse_plan

    # unknown specialist + empty goal dropped; alias normalized
    text = '{"steps": [{"specialist": "agent_tools", "goal": "ok"}, {"specialist": "bogo", "goal": "x"}, {"specialist": "rag", "goal": ""}]}'
    assert parse_plan(text) == [{"specialist": "tool", "goal": "ok"}]


def test_planner_json_invalid_falls_back_to_regex():
    from planner import parse_plan

    # JSON present but "steps" missing -> fall back to legacy <step> tags
    text = '{"plan": "oops"}\n<step specialist="web" goal="tim web" />'
    assert parse_plan(text) == [{"specialist": "web", "goal": "tim web"}]


def test_planner_legacy_tags_still_work():
    from planner import parse_plan

    text = '<step specialist="tool" goal="x" />'
    assert parse_plan(text) == [{"specialist": "tool", "goal": "x"}]