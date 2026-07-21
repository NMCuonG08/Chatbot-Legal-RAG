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


# ---- VN-number normalization (consolidation: accept VN format, reject garbage) ----

def test_vn_dotted_thousands_accepted():
    """'15.000.000' (VN thousands-dot) must coerce to 15000000.0 and compute."""
    out = _content(atw.severance_pay_func_tool, monthly_salary="15.000.000", months_worked=36)
    data = json.loads(out)
    assert "error" not in data, f"VN-formatted number must be accepted, got: {out}"
    assert "severance_allowance" in data or "allowance" in str(data).lower()


def test_vn_currency_suffix_accepted():
    out = _content(atw.severance_pay_func_tool, monthly_salary="15000000 vnđ", months_worked=36)
    data = json.loads(out)
    assert "error" not in data


def test_vn_multiplier_word_rejected():
    """'5 triệu' is ambiguous -> must be rejected (no silent extract of '5')."""
    out = _content(atw.severance_pay_func_tool, monthly_salary="5 triệu", months_worked=36)
    data = json.loads(out)
    assert "error" in data, f"multiplier word must be rejected, got: {out}"


def test_contract_penalty_rejects_nonnumeric():
    out = _content(atw.contract_penalty_tool, contract_value="abc", penalty_rate=0.12, days_late=10)
    data = json.loads(out)
    assert "error" in data


def test_contract_penalty_vn_format_accepted():
    out = _content(atw.contract_penalty_tool, contract_value="100.000.000", penalty_rate="0,12", days_late=30)
    data = json.loads(out)
    assert "error" not in data, f"VN-format penalty rate must be accepted, got: {out}"


def test_overtime_rejects_bad_hours():
    out = _content(atw.overtime_pay_func_tool, hourly_wage=50000, hours="nhiều")
    data = json.loads(out)
    assert "error" in data


def test_vehicle_fee_rejects_bad_value():
    out = _content(atw.vehicle_registration_fee_func_tool, vehicle_value="hai tỷ", vehicle_type="car")
    data = json.loads(out)
    assert "error" in data


def test_child_support_rejects_bad_income():
    out = _content(atw.child_support_func_tool, payer_income="không có", num_children=2)
    data = json.loads(out)
    assert "error" in data


def test_business_name_rejects_empty():
    out = _content(atw.business_name_tool, business_name="")
    data = json.loads(out)
    assert "error" in data


def test_statute_lookup_valid_case_type():
    out = _content(atw.statute_tool, case_type="civil")
    # statute_lookup returns info for valid case types; must not be a pydantic error
    data = json.loads(out)
    assert "error" not in data or "Tham số" not in str(data)