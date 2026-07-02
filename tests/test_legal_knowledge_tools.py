"""Tests for legal_knowledge_tools.py — data-driven Vietnamese legal tools.

Covers severance, overtime, PIT, registration fees, court fee, admin fine,
child support, procedure wizard, jurisdiction, document template, law
version, and disclaimer/escalation. Deterministic (no Qdrant/LLM needed).
"""
import json

import pytest

from legal_knowledge_tools import (
    calculate_child_support,
    calculate_court_fee,
    calculate_land_registration_fee,
    calculate_overtime_pay,
    calculate_pit_monthly,
    calculate_severance_pay,
    calculate_vehicle_registration_fee,
    get_law_version,
    legal_disclaimer_check,
    lookup_administrative_fine,
)
from legal_procedure_tools import (
    generate_document_template,
    jurisdiction_resolver,
    procedure_wizard,
)


# ===== Severance (Đ48 BLLĐ 2019) =====


class TestSeverance:
    def test_one_month_per_year_post_2009(self):
        # Arrange: 15M/month, 36 months = 3 years -> 3 * 15M = 45M
        # Act
        result = calculate_severance_pay(15_000_000, 36)
        # Assert
        assert "error" not in result
        assert result["severance_allowance"] == "45,000,000 VNĐ"
        assert "48" in result["legal_basis"]

    def test_invalid_salary_rejected(self):
        result = calculate_severance_pay(-1, 12)
        assert "error" in result

    def test_invalid_months_rejected(self):
        result = calculate_severance_pay(10_000_000, 9999)
        assert "error" in result


# ===== Overtime (Đ107 BLLĐ 2019) =====


class TestOvertime:
    @pytest.mark.parametrize("day_type,expected_mult", [
        ("weekday", 1.5),
        ("rest_day", 2.0),
        ("holiday", 3.0),
    ])
    def test_multipliers(self, day_type, expected_mult):
        # 50k/hr * 4h * mult
        result = calculate_overtime_pay(50_000, 4, day_type=day_type)
        assert "error" not in result
        assert result["multiplier"] == f"{expected_mult}x"
        assert result["overtime_pay"] == f"{50_000 * expected_mult * 4:,.0f} VNĐ"

    def test_invalid_day_type(self):
        result = calculate_overtime_pay(50_000, 4, day_type="monday")
        assert "error" in result


# ===== PIT (lũy tiến) =====


class TestPIT:
    def test_zero_income_no_tax(self):
        result = calculate_pit_monthly(0)
        assert result["tax"] == 0

    def test_first_bracket_only(self):
        # 4M taxable -> 5% = 200k
        result = calculate_pit_monthly(4_000_000)
        assert "error" not in result
        assert result["tax_owed"] == "200,000 VNĐ"

    def test_multi_bracket_progressive(self):
        # 20M: 5M*5% + 5M*10% + 8M*15% + 2M*20% = 250k+500k+1.2M+400k = 2.35M
        result = calculate_pit_monthly(20_000_000)
        assert result["tax_owed"] == "2,350,000 VNĐ"

    def test_absurd_income_rejected(self):
        result = calculate_pit_monthly(10**12)
        assert "error" in result


# ===== Land registration fee =====


class TestLandRegistrationFee:
    def test_half_percent(self):
        # 2B * 0.5% = 10M
        result = calculate_land_registration_fee(2_000_000_000)
        assert result["registration_fee"] == "10,000,000 VNĐ"
        assert "0.5%" in result["rate"]

    def test_invalid_value(self):
        assert "error" in calculate_land_registration_fee(0)


# ===== Vehicle registration fee =====


class TestVehicleRegistrationFee:
    def test_car_first_time_hn_hcm(self):
        # 1B * 10% = 100M
        result = calculate_vehicle_registration_fee(1_000_000_000, "car", True)
        assert result["registration_fee"] == "100,000,000 VNĐ"
        assert result["rate"] == "10%"

    def test_motorcycle(self):
        result = calculate_vehicle_registration_fee(50_000_000, "motorcycle", True)
        assert result["rate"] == "5%"

    def test_invalid_type(self):
        assert "error" in calculate_vehicle_registration_fee(50_000_000, "boat")


# ===== Court fee (NQ 326/2016) =====


class TestCourtFee:
    def test_civil_first_5_percent(self):
        # 100M * 5% = 5M, above min 300k
        result = calculate_court_fee(100_000_000, "civil_first")
        assert result["court_fee"] == "5,000,000 VNĐ"

    def test_min_fee_applied(self):
        # small claim below 6M -> 5% < 300k -> min 300k
        result = calculate_court_fee(1_000_000, "civil_first")
        assert result["court_fee"] == "300,000 VNĐ"
        assert result["min_fee_applied"] is True

    def test_no_value_300k(self):
        result = calculate_court_fee(0, "no_value")
        assert result["court_fee"] == "300,000 VNĐ"

    def test_invalid_case_type(self):
        assert "error" in calculate_court_fee(10_000_000, "criminal")


# ===== Admin fine =====


class TestAdminFine:
    def test_known_violation(self):
        result = lookup_administrative_fine("traffic_alcohol_motorbike")
        assert "error" not in result
        assert "fine" in result
        assert "123/2021" in result["legal_basis"]

    def test_unknown_lists_keys(self):
        result = lookup_administrative_fine("nonexistent")
        assert "error" in result
        assert "available_types" in result


# ===== Child support (Đ82 HNGĐ) =====


class TestChildSupport:
    def test_one_child_25_percent(self):
        result = calculate_child_support(10_000_000, 1)
        assert "error" not in result
        assert result["per_child_rate_guideline"] == "25%"
        assert result["total_estimated"] == "2,500,000 VNĐ/tháng"

    def test_warning_present(self):
        result = calculate_child_support(10_000_000, 1)
        assert "warning" in result

    def test_invalid_children(self):
        assert "error" in calculate_child_support(10_000_000, 0)


# ===== Procedure wizard =====


class TestProcedureWizard:
    def test_divorce_has_docs_and_authority(self):
        result = procedure_wizard("divorce")
        assert "error" not in result
        assert result["competent_authority"].startswith("Tòa án")
        assert any("Giấy chứng nhận kết hôn" in d for d in result["required_docs"])

    def test_unknown_procedure_lists_keys(self):
        result = procedure_wizard("bankruptcy")
        assert "error" in result
        assert "available_types" in result


# ===== Jurisdiction resolver =====


class TestJurisdiction:
    def test_civil_high_value_notes_provincial_court(self):
        result = jurisdiction_resolver("civil", claim_value=600_000_000)
        assert "500 triệu" in result["value_note"]

    def test_family_default_district(self):
        result = jurisdiction_resolver("family")
        assert "cấp huyện" in result["authority"]

    def test_invalid_type(self):
        assert "error" in jurisdiction_resolver("immigration")


# ===== Document template =====


class TestDocumentTemplate:
    def test_lawsuit_template_fills_params(self):
        params = json.dumps({
            "noi_nhan": "TAND quận X",
            "nguyen_don": "Nguyễn Văn A",
            "bi_don": "Trần Thị B",
            "yeu_cau": "buộc trả nợ 100tr",
        })
        result = generate_document_template("don_khoi_kien_civil", params_json=params)
        assert "error" not in result
        assert "Nguyễn Văn A" in result["document"]
        assert "Trần Thị B" in result["document"]
        assert "TAND quận X" in result["document"]
        assert "warning" in result

    def test_invalid_json_rejected(self):
        result = generate_document_template("don_ly_hon", params_json="{bad")
        assert "error" in result

    def test_unknown_doc_type(self):
        result = generate_document_template("will", params_json="{}")
        assert "error" in result
        assert "available_types" in result

    def test_default_placeholders_used_when_missing(self):
        result = generate_document_template("don_ly_hon", params_json="{}")
        assert "....." in result["document"]


# ===== Law version =====


class TestLawVersion:
    def test_known_law(self):
        result = get_law_version("blld_2019")
        assert "error" not in result
        assert result["full_name"] == "Bộ luật Lao động 2019"
        assert result["status_at_year"] == "in_force"

    def test_not_yet_effective_when_year_before(self):
        # BLĐ 2019 effective 2021 -> 2020 not yet effective
        result = get_law_version("blld_2019", effective_year=2020)
        assert result["status_at_year"] == "not_yet_effective"

    def test_unknown_key(self):
        result = get_law_version("nonexistent_law")
        assert "error" in result


# ===== Disclaimer / escalation =====


class TestDisclaimer:
    def test_criminal_topic_escalates(self):
        result = legal_disclaimer_check("Tôi bị khởi tố hình sự về tội trộm cắp, cần bào chữa thế nào?")
        assert result["escalate"] is True
        assert len(result["matched_topics"]) > 0
        assert "LUẬT SƯ" in result["disclaimer"]

    def test_benign_question_no_escalate(self):
        result = legal_disclaimer_check("Điều kiện đăng ký kết hôn là gì?")
        assert result["escalate"] is False
        assert "tham khảo" in result["disclaimer"].lower()

    def test_disclaimer_always_returned(self):
        for q in ["", "abc", "tính phạt hợp đồng"]:
            r = legal_disclaimer_check(q)
            assert "disclaimer" in r and r["disclaimer"]