"""Tests for legal_retrieval_tools.py — Qdrant-backed tools.

Qdrant/LLM are NOT hit here: we monkeypatch the module's ``_hybrid_search``
and ``_vector_search`` helpers to return synthetic hits. This exercises the
formatting, filtering, and verify_citation overlap logic deterministically.
"""
import pytest

import legal_retrieval_tools as lrt


# ===== lookup_article =====


class TestLookupArticle:
    def test_validation_requires_law_name(self):
        result = lrt.lookup_article("", 418)
        assert "error" in result

    def test_found_returns_matches(self, monkeypatch):
        # Arrange: fake vector hit
        hits = [{"content": "Điều 418 BLDS 2015: phạt vi phạm không quá 8%.", "question": "",
                 "source": "BLDS2015", "doc_id": 1, "similarity_score": 0.9}]
        monkeypatch.setattr(lrt, "_vector_search", lambda q, limit=5, filters=None: hits)
        # Act
        result = lrt.lookup_article("Bộ luật Dân sự 2015", 418)
        # Assert
        assert result["found"] is True
        assert result["count"] == 1
        assert "418" in result["matches"][0]["content"]
        assert result["matches"][0]["source"] == "BLDS2015"

    def test_not_found_falls_back_then_message(self, monkeypatch):
        monkeypatch.setattr(lrt, "_vector_search", lambda q, limit=5, filters=None: [])
        monkeypatch.setattr(lrt, "_hybrid_search", lambda q, limit=5: [])
        result = lrt.lookup_article("Luật Vật lý 2099", 1)
        assert result["found"] is False
        assert "message" in result

    def test_vector_empty_falls_back_to_hybrid(self, monkeypatch):
        calls = {"vector": 0, "hybrid": 0}

        def fake_vector(q, limit=5, filters=None):
            calls["vector"] += 1
            return []

        def fake_hybrid(q, limit=5):
            calls["hybrid"] += 1
            return [{"content": "fallback hit", "question": "", "source": "s", "doc_id": 2,
                     "hybrid_score": 0.5}]

        monkeypatch.setattr(lrt, "_vector_search", fake_vector)
        monkeypatch.setattr(lrt, "_hybrid_search", fake_hybrid)
        result = lrt.lookup_article("Bộ luật Dân sự 2015")
        assert result["found"] is True
        assert calls["hybrid"] == 1

    def test_exact_match_uses_article_number_filter(self, monkeypatch):
        """When article_number is given, the first vector search filters by it;
        a hit carrying that article_number yields exact_match=True."""
        seen_filters = []

        def fake_vector(q, limit=5, filters=None):
            seen_filters.append(filters)
            # Only the exact-filter call returns a hit.
            if filters and filters.get("article_number") == 418:
                return [{"content": "Điều 418 BLDS 2015: phạt vi phạm.",
                         "question": "", "source": "BLDS2015", "doc_id": 1,
                         "law_name": "Bộ luật Dân sự 2015", "article_number": 418,
                         "similarity_score": 0.95}]
            return []

        monkeypatch.setattr(lrt, "_vector_search", fake_vector)
        result = lrt.lookup_article("Bộ luật Dân sự 2015", 418)
        assert result["found"] is True
        assert result["exact_match"] is True
        assert result["matches"][0]["article_number"] == 418
        assert result["matches"][0]["law_name"] == "Bộ luật Dân sự 2015"
        # First call must carry the structured article_number filter.
        assert seen_filters[0]["article_number"] == 418

    def test_no_exact_match_falls_back_to_semantic(self, monkeypatch):
        """Exact-filter returns nothing -> semantic content_type filter used."""
        calls = []

        def fake_vector(q, limit=5, filters=None):
            calls.append(filters)
            # Exact-filter (article_number present) empty; semantic (only content_type) hits.
            if filters and "article_number" in filters:
                return []
            return [{"content": "semantic hit about Điều 418", "question": "",
                     "source": "s", "doc_id": 3, "similarity_score": 0.7}]

        monkeypatch.setattr(lrt, "_vector_search", fake_vector)
        result = lrt.lookup_article("Bộ luật Dân sự 2015", 418)
        assert result["found"] is True
        # Hit lacks article_number metadata -> not an exact match.
        assert result["exact_match"] is False
        assert len(calls) == 2  # exact-filter then semantic


# ===== precedent_lookup =====


class TestPrecedentLookup:
    def test_requires_fact_pattern(self):
        assert "error" in lrt.precedent_lookup("")

    def test_found_via_case_law_filter(self, monkeypatch):
        hits = [{"content": "Án lệ 04/2016/AL: vay không trả.", "question": "",
                 "source": "AL", "doc_id": 7, "similarity_score": 0.88}]
        monkeypatch.setattr(lrt, "_vector_search", lambda q, limit=5, filters=None: hits)
        result = lrt.precedent_lookup("vay tiền không trả có giấy vay")
        assert result["found"] is True
        assert result["precedents"][0]["source"] == "AL"


# ===== cross_reference =====


class TestCrossReference:
    def test_requires_both_args(self):
        assert "error" in lrt.cross_reference("", 418)
        assert "error" in lrt.cross_reference("BLDS 2015", 0)

    def test_filters_to_mentioning_articles(self, monkeypatch):
        hits = [
            {"content": "Theo Điều 418 BLDS thì phạt 8%", "question": "", "source": "s1", "doc_id": 1},
            {"content": "Điều 100 không liên quan", "question": "", "source": "s2", "doc_id": 2},
        ]
        monkeypatch.setattr(lrt, "_hybrid_search", lambda q, limit=5: hits)
        result = lrt.cross_reference("Bộ luật Dân sự 2015", 418)
        assert result["count"] == 1
        assert "418" in result["referencing_texts"][0]["content"]


# ===== verify_citation =====


class TestVerifyCitation:
    def test_validation_requires_all_args(self):
        assert "error" in lrt.verify_citation("", 418, "text")
        assert "error" in lrt.verify_citation("BLDS", 0, "text")
        assert "error" in lrt.verify_citation("BLDS", 418, "")

    def test_cannot_verify_when_corpus_empty(self, monkeypatch):
        monkeypatch.setattr(lrt, "_vector_search", lambda q, limit=5, filters=None: [])
        monkeypatch.setattr(lrt, "_hybrid_search", lambda q, limit=5: [])
        result = lrt.verify_citation("Luật Vật lý 2099", 1, "some claim")
        assert result["verified"] is False
        assert result["verdict"] == "cannot_verify"

    def test_consistent_when_high_overlap(self, monkeypatch):
        corpus = "phạt vi phạm không quá 8 giá trị nghĩa vụ bị vi phạm điều 418"
        hits = [{"content": corpus, "question": "", "source": "BLDS", "doc_id": 1, "similarity_score": 0.9}]
        monkeypatch.setattr(lrt, "_vector_search", lambda q, limit=5, filters=None: hits)
        # Claim reuses most of the same tokens
        result = lrt.verify_citation("Bộ luật Dân sự 2015", 418,
                                     "phạt vi phạm không quá 8 giá trị nghĩa vụ bị vi phạm")
        assert result["verdict"] == "consistent"
        assert result["verified"] is True

    def test_contradicts_when_no_overlap(self, monkeypatch):
        hits = [{"content": "hoàn toàn khác biệt nội dung không trùng token", "question": "",
                 "source": "BLDS", "doc_id": 1, "similarity_score": 0.9}]
        monkeypatch.setattr(lrt, "_vector_search", lambda q, limit=5, filters=None: hits)
        result = lrt.verify_citation("Bộ luật Dân sự 2015", 418,
                                     "completely different claim about something else here")
        assert result["verdict"] == "contradicts_or_unsupported"
        assert result["verified"] is False