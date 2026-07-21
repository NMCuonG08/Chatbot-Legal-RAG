"""Tests for legal_graph_relations.extract_relations — rule+regex cross-
reference extractor for CITES / AMENDS / REPEALS / REPLACED_BY edges.

Deterministic, no network/Neo4j needed.
"""
from legal_graph_relations import (
    RELATION_AMENDS,
    RELATION_CITES,
    RELATION_REPEALS,
    RELATION_REPLACED_BY,
    extract_relations,
)

_SRC_LAW = "Bộ luật Dân sự 2015"
_SRC_ART = 35


class TestCites:
    def test_dieu_lead_chiếu(self):
        text = "Khoản 2 Điều 35 dẫn chiếu Điều 40 của Bộ luật này."
        rels = extract_relations(text, source_law=_SRC_LAW, source_article=_SRC_ART)
        cites = [r for r in rels if r["relation"] == RELATION_CITES]
        assert any(r["target_art"] == 40 for r in cites)

    def test_theo_cites_named_law(self):
        text = "Theo Điều 22 Bộ luật Dân sự 2005, người có năng lực hành vi..."
        rels = extract_relations(text, source_law=_SRC_LAW, source_article=_SRC_ART)
        c = [r for r in rels if r["relation"] == RELATION_CITES]
        assert c and c[0]["target_art"] == 22 and "Bộ luật Dân sự 2005" in c[0]["target_law"]


class TestAmends:
    def test_sua_doi(self):
        text = "Điều này sửa đổi Điều 22 Bộ luật Dân sự 2005."
        rels = extract_relations(text, source_law=_SRC_LAW, source_article=_SRC_ART)
        a = [r for r in rels if r["relation"] == RELATION_AMENDS]
        assert a and a[0]["target_art"] == 22

    def test_bo_sung(self):
        text = "Bổ sung Điều 10 Luật Hôn nhân gia đình 2014."
        rels = extract_relations(text, source_law=_SRC_LAW, source_article=_SRC_ART)
        a = [r for r in rels if r["relation"] == RELATION_AMENDS]
        assert a and a[0]["target_art"] == 10


class TestRepeals:
    def test_bai_bo(self):
        text = "Điều này bãi bỏ Điều 10 Luật Hôn nhân gia đình 2000."
        rels = extract_relations(text, source_law=_SRC_LAW, source_article=_SRC_ART)
        r = [r for r in rels if r["relation"] == RELATION_REPEALS]
        assert r and r[0]["target_art"] == 10


class TestReplacedBy:
    def test_thay_the_bang(self):
        text = "Bộ luật này thay thế bằng Điều 1 Bộ luật Dân sự 2015."
        rels = extract_relations(text, source_law="Bộ luật Dân sự 2005", source_article=2)
        rb = [r for r in rels if r["relation"] == RELATION_REPLACED_BY]
        assert rb and rb[0]["target_art"] == 1
        assert "Bộ luật Dân sự 2015" in rb[0]["target_law"]


class TestDedupAndSelfRef:
    def test_dedup_same_target(self):
        # Two "theo Điều 40" mentions -> one CITES edge.
        text = "Theo Điều 40. Xem lại theo Điều 40."
        rels = extract_relations(text, source_law=_SRC_LAW, source_article=_SRC_ART)
        cites = [r for r in rels if r["relation"] == RELATION_CITES and r["target_art"] == 40]
        assert len(cites) == 1

    def test_self_reference_filtered(self):
        # "Theo Điều 35 [this law]" cites the source article itself -> dropped.
        text = "Theo Điều 35 Bộ luật Dân sự 2015, quy định về năng lực hành vi."
        rels = extract_relations(text, source_law=_SRC_LAW, source_article=_SRC_ART)
        self_cites = [r for r in rels if r["target_art"] == _SRC_ART and r["target_law"] == _SRC_LAW]
        assert not self_cites


class TestEmptyAndGuard:
    def test_no_source_law_drops(self):
        assert extract_relations("Theo Điều 40", source_law=None, source_article=35) == []

    def test_no_source_article_drops(self):
        assert extract_relations("Theo Điều 40", source_law=_SRC_LAW, source_article=None) == []

    def test_empty_text(self):
        assert extract_relations("", source_law=_SRC_LAW, source_article=_SRC_ART) == []

    def test_no_article_ref(self):
        # Verb present but no "Điều X" target -> nothing emitted.
        text = "Theo quy định tại điểm a khoản 2."
        assert extract_relations(text, source_law=_SRC_LAW, source_article=_SRC_ART) == []