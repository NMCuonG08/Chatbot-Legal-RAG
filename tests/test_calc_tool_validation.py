"""#2 — Pydantic input validation on calc tools (Flaw 4).

Root cause verified empirically:
- severance_pay_tool(monthly_salary="5 năm", ...) -> SILENT WRONG ("5 VNĐ")
- legal_age_tool(birth_year="kết hôn", ...) -> RAISES TypeError (str < int)

A Pydantic fn_schema + runtime validation decorator must turn both into a
clean JSON error (no raise, no garbage), while still accepting valid input and
coercing numeric strings ("15000000" -> 15000000.0).
"""
import json

import agent_tool_wrappers as atw


def _content(tool, **kw):
    return tool.call(**kw).content


def test_severance_rejects_nonnumeric_string_no_silent_garbage():
    out = _content(atw.severance_pay_func_tool, monthly_salary="5 năm", months_worked=36)
    data = json.loads(out)
    assert "error" in data, f"expected clean error, got silent garbage: {out}"
    assert "5 VNĐ" not in out


def test_severance_rejects_months_string():
    out = _content(atw.severance_pay_func_tool, monthly_salary=15_000_000, months_worked="ba mươi")
    data = json.loads(out)
    assert "error" in data


def test_legal_age_rejects_nonnumeric_birthyear_without_raising():
    # Currently raises TypeError -> must become clean error.
    out = _content(atw.legal_age_tool, birth_year="kết hôn", action_type="marriage")
    data = json.loads(out)
    assert "error" in data


def test_pit_rejects_nonnumeric_income():
    out = _content(atw.pit_monthly_func_tool, taxable_income="không có")
    data = json.loads(out)
    assert "error" in data


def test_severance_valid_input_still_computes():
    out = _content(atw.severance_pay_func_tool, monthly_salary=15_000_000, months_worked=36)
    data = json.loads(out)
    assert "severance_allowance" in data or "allowance" in str(data).lower()


def test_severance_coerces_numeric_string():
    """Numeric string '15000000' must coerce to float and compute (not error)."""
    out = _content(atw.severance_pay_func_tool, monthly_salary="15000000", months_worked=36)
    data = json.loads(out)
    assert "error" not in data, f"numeric string should coerce, got: {out}"
    assert "severance_allowance" in data or "allowance" in str(data).lower()


def test_severance_rejects_negative_salary():
    out = _content(atw.severance_pay_func_tool, monthly_salary=-1, months_worked=36)
    data = json.loads(out)
    assert "error" in data


def test_court_fee_rejects_nonnumeric_claim():
    out = _content(atw.court_fee_func_tool, claim_value="mười triệu", case_type="civil_first")
    data = json.loads(out)
    assert "error" in data


def test_land_fee_rejects_nonnumeric_property():
    out = _content(atw.land_registration_fee_func_tool, property_value="hai tỷ")
    data = json.loads(out)
    assert "error" in data


def test_inheritance_rejects_nonnumeric_total():
    out = _content(atw.inheritance_tool, total_value="năm trăm triệu", heirs_json="[]")
    data = json.loads(out)
    assert "error" in data