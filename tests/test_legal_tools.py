"""Legal correctness tests for legal_tools.py.

These tests encode CORRECT Vietnamese legal behavior. They exist because the
previous implementation gave legally wrong advice (marriage age, penalty cap,
inheritance tiers). See BLDS 2015, Luật HNGĐ 2014, Luật Doanh nghiệp 2020.
"""
import pytest

from legal_tools import (
    calculate_contract_penalty,
    calculate_inheritance_share,
    check_legal_entity_age,
)


# ===== Contract penalty (Điều 418 BLDS 2015) =====


class TestContractPenalty:
    def test_penalty_capped_at_8_percent_statutory(self):
        # 100M * 0.5%/day * 30 days = 15M, exceeds 8% cap (8M) -> capped.
        result = calculate_contract_penalty(
            contract_value=100_000_000, penalty_rate=0.5, days_late=30
        )
        assert "error" not in result
        assert result["penalty_amount"] == "8,000,000 VNĐ"
        assert "8%" in result["note"]
        assert "418" in result["legal_basis"]

    def test_penalty_below_cap_uses_agreed_rate(self):
        # 100M * 0.01%/day * 10 days = 100,000, well below 8M cap.
        result = calculate_contract_penalty(
            contract_value=100_000_000, penalty_rate=0.01, days_late=10
        )
        assert result["penalty_amount"] == "100,000 VNĐ"
        assert "thỏa thuận" in result["note"].lower()

    def test_penalty_zero_days(self):
        result = calculate_contract_penalty(
            contract_value=50_000_000, penalty_rate=0.1, days_late=0
        )
        assert result["penalty_amount"] == "0 VNĐ"


# ===== Legal age (Điều 8 Luật HNGĐ 2014, Điều 21 BLDS 2015) =====


class TestLegalAge:
    def test_marriage_male_under_20_not_eligible(self):
        # Born 2007 -> age 19 in 2026.
        result = check_legal_entity_age(2007, "marriage", gender="male")
        assert result["eligible"] is False
        assert "20" in result["legal_basis"]

    def test_marriage_male_20_eligible(self):
        result = check_legal_entity_age(2006, "marriage", gender="male")
        assert result["eligible"] is True

    def test_marriage_female_18_eligible(self):
        result = check_legal_entity_age(2008, "marriage", gender="female")
        assert result["eligible"] is True

    def test_marriage_female_17_not_eligible(self):
        result = check_legal_entity_age(2009, "marriage", gender="female")
        assert result["eligible"] is False

    def test_marriage_without_gender_requires_it(self):
        result = check_legal_entity_age(2007, "marriage")
        assert "error" in result or result.get("eligible") == "partial"
        # Must not silently tell a 19yo male he can marry.
        assert result.get("eligible") is not True

    def test_sign_contract_18_eligible(self):
        result = check_legal_entity_age(2008, "sign_contract")
        assert result["eligible"] is True

    def test_sign_contract_15_partial(self):
        result = check_legal_entity_age(2011, "sign_contract")
        assert result["eligible"] == "partial"

    def test_invalid_gender_for_marriage(self):
        result = check_legal_entity_age(2000, "marriage", gender="other")
        assert "error" in result


# ===== Inheritance (Điều 651 BLDS 2015) =====


class TestInheritance:
    def test_first_tier_divides_equally(self):
        heirs = [
            {"name": "Vợ", "relation": "spouse"},
            {"name": "Con", "relation": "child"},
            {"name": "Cha", "relation": "parent"},
        ]
        result = calculate_inheritance_share(900_000_000, heirs)
        assert result["num_heirs"] == 3
        assert result["share_per_heir"] == "300,000,000 VNĐ"
        assert "651" in result["legal_basis"]

    def test_mixed_tier_lower_tier_excluded_when_first_tier_present(self):
        # Child (tier 1) + sibling (tier 2). Sibling must NOT inherit.
        heirs = [
            {"name": "Con", "relation": "child"},
            {"name": "Anh ruột", "relation": "sibling"},
        ]
        result = calculate_inheritance_share(500_000_000, heirs)
        assert result["num_heirs"] == 1
        assert result["share_per_heir"] == "500,000,000 VNĐ"
        names = [d["name"] for d in result["distribution"]]
        assert "Con" in names
        assert "Anh ruột" not in names

    def test_second_tier_inherits_when_no_first_tier(self):
        heirs = [
            {"name": "Anh ruột", "relation": "sibling"},
            {"name": "Ông nội", "relation": "grandparent"},
        ]
        result = calculate_inheritance_share(400_000_000, heirs)
        assert result["num_heirs"] == 2
        assert result["share_per_heir"] == "200,000,000 VNĐ"

    def test_empty_heirs_error(self):
        result = calculate_inheritance_share(100_000_000, [])
        assert "error" in result

    def test_minor_gets_note(self):
        heirs = [{"name": "Con nhỏ", "relation": "child", "is_minor": True}]
        result = calculate_inheritance_share(100_000_000, heirs)
        assert any("đại diện" in d.get("note", "") for d in result["distribution"])

    def test_unknown_relation_excluded_with_warning(self):
        heirs = [
            {"name": "Con", "relation": "child"},
            {"name": "Người lạ", "relation": "stranger"},
        ]
        result = calculate_inheritance_share(300_000_000, heirs)
        # Stranger is not a legal heir -> only child inherits.
        assert result["num_heirs"] == 1