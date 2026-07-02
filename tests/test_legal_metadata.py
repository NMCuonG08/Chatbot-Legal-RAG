"""Tests for legal_metadata.extract_legal_metadata — regex-based structured
extraction of law_name + article_number from free Vietnamese legal text.

Deterministic, no network/Qdrant needed.
"""
from legal_metadata import extract_legal_metadata


class TestArticleNumber:
    def test_extracts_article(self):
        result = extract_legal_metadata("Điều 418 Bộ luật Dân sự 2015 quy định...")
        assert result["article_number"] == 418

    def test_case_insensitive_dieu(self):
        result = extract_legal_metadata("theo điều 107 thì...")
        assert result["article_number"] == 107

    def test_no_article_returns_none_key(self):
        result = extract_legal_metadata("Bộ luật Dân sự 2015 không có điều cụ thể")
        assert "article_number" not in result


class TestLawName:
    def test_bo_luat_code(self):
        result = extract_legal_metadata("Theo Bộ luật Dân sự 2015 thì...")
        assert result["law_name"] == "Bộ luật Dân sự 2015"

    def test_luat_act(self):
        result = extract_legal_metadata("Luật Đất đai 2024 quy định...")
        assert result["law_name"] == "Luật Đất đai 2024"

    def test_nghi_dinh(self):
        result = extract_legal_metadata("Nghị định 10/2022/NĐ-CP quy định lệ phí")
        assert result["law_name"].startswith("Nghị định 10/2022")

    def test_thong_tu(self):
        result = extract_legal_metadata("Theo Thông tư 80/2020/TT-BTC")
        assert result["law_name"].startswith("Thông tư 80/2020")

    def test_nghi_quyet_with_ubtvqh(self):
        result = extract_legal_metadata("Nghị quyết 326/2016/UBTVQH14 về án phí")
        assert result["law_name"] == "Nghị quyết 326/2016/UBTVQH14"

    def test_code_precedence_over_decree(self):
        # "Bộ luật" should win over a decree mention in the same text.
        result = extract_legal_metadata("Bộ luật Lao động 2019 và Nghị định 10/2022")
        assert result["law_name"] == "Bộ luật Lao động 2019"


class TestEmptyAndUnknown:
    def test_empty_string(self):
        assert extract_legal_metadata("") == {}

    def test_none_safe(self):
        assert extract_legal_metadata(None) == {}

    def test_no_legal_refs(self):
        assert extract_legal_metadata("Xin chào, hôm nay thời tiết đẹp") == {}

    def test_only_article_no_law(self):
        result = extract_legal_metadata("Điều 418 nói về phạt vi phạm")
        assert result == {"article_number": 418}


class TestCombined:
    def test_article_and_law(self):
        result = extract_legal_metadata("Điều 48 Bộ luật Lao động 2019 về thôi việc")
        assert result == {"law_name": "Bộ luật Lao động 2019", "article_number": 48}

    def test_returns_first_article_only(self):
        # Multiple article refs — first one wins.
        result = extract_legal_metadata("Điều 418 và Điều 419 của Bộ luật Dân sự 2015")
        assert result["article_number"] == 418
        assert result["law_name"] == "Bộ luật Dân sự 2015"