"""Unit tests for inline-citation pipeline.

Covers:
- ``citations.build_search_url`` — Google search link, URL-encoded, priority
  query (Điều n + law_name > law_name > question > content).
- ``citations.normalize_sources`` — stable 1-based id matching input order,
  title priority, url (web passthrough vs corpus Google link), kind tag,
  preserves existing fields (content/doc_id/...).
- ``citation_render.render_answer_html`` — marker [n] / [n][m] replacement,
  missing source stays literal, HTML escape (XSS), newline -> <br>.
"""
import sys
from pathlib import Path

# Frontend renderer lives outside backend/src; add it to sys.path.
_FRONTEND = Path(__file__).resolve().parent.parent / "frontend"
if str(_FRONTEND) not in sys.path:
    sys.path.insert(0, str(_FRONTEND))

import citations
import citation_render


# ---- build_search_url ----

def test_build_search_url_article_and_law():
    url = citations.build_search_url("Bộ luật Dân sự 2015", 418, None, None)
    assert "google.com/search?q=" in url
    # Vietnamese encoded with quote_plus (spaces -> +).
    assert "%C4%90i%E1%BB%81u+418" in url or "Điều+418" in url
    assert "B%E1%BB%99+lu%E1%BA%ADt" in url  # "Bộ luật"


def test_build_search_url_law_only():
    url = citations.build_search_url("Luật Đất đai 2024", None, None, None)
    assert "google.com/search?q=" in url
    assert "Lu%E1%BA%ADt+%C4%90%E1%BA%A5t+%C4%91ai" in url


def test_build_search_url_fallback_to_question():
    url = citations.build_search_url(None, None, "vay tiền không trả", None)
    assert "google.com/search?q=" in url
    assert "vay+ti%E1%BB%81n" in url


def test_build_search_url_fallback_to_content():
    url = citations.build_search_url(None, None, None, "Nội dung dài nào đó"[:60])
    assert "google.com/search?q=" in url


def test_build_search_url_empty_returns_none():
    assert citations.build_search_url(None, None, None, None) is None


# ---- normalize_sources ----

def test_normalize_sources_ids_match_order():
    docs = [
        {"content": "a", "law_name": "BLDS 2015", "article_number": 418},
        {"content": "b", "question": "q2"},
        {"content": "c", "source": "file_x.pdf"},
    ]
    out = citations.normalize_sources(docs, kind="corpus")
    assert [s["id"] for s in out] == [1, 2, 3]
    # Order preserved == input order (matches gen_doc_prompt numbering).
    assert out[0]["content"] == "a"
    assert out[1]["content"] == "b"


def test_normalize_sources_title_priority():
    docs = [
        {"content": "x", "law_name": "Bộ luật Dân sự 2015", "article_number": 418},
        {"content": "y", "question": "Thời hiệu khởi kiện là多久"},
        {"content": "z", "source": "nghidinh.pdf"},
    ]
    out = citations.normalize_sources(docs, kind="corpus")
    assert "418" in out[0]["title"] and "Dân sự" in out[0]["title"]
    assert out[1]["title"] == "Thời hiệu khởi kiện là多久"
    assert out[2]["title"] == "nghidinh.pdf"


def test_normalize_sources_corpus_gets_google_url():
    docs = [{"content": "x", "law_name": "BLDS 2015", "article_number": 418}]
    out = citations.normalize_sources(docs, kind="corpus")
    assert out[0]["url"].startswith("https://www.google.com/search?q=")
    assert out[0]["kind"] == "corpus"


def test_normalize_sources_web_keeps_real_url():
    docs = [{"content": "x", "url": "https://vbpl.vn/123", "title": "Nghị định 168"}]
    out = citations.normalize_sources(docs, kind="web")
    assert out[0]["url"] == "https://vbpl.vn/123"
    assert out[0]["kind"] == "web"
    # Web title preserved when present.
    assert out[0]["title"] == "Nghị định 168"


def test_normalize_sources_preserves_existing_fields():
    docs = [{"content": "x", "doc_id": "abc", "score": 0.9,
             "law_name": "BLDS", "article_number": 1, "content_type": "law"}]
    out = citations.normalize_sources(docs, kind="corpus")
    s = out[0]
    assert s["doc_id"] == "abc"
    assert s["score"] == 0.9
    assert s["law_name"] == "BLDS"
    assert s["content_type"] == "law"


def test_normalize_sources_empty():
    assert citations.normalize_sources([], kind="corpus") == []


# ---- render_answer_html ----

def test_render_replaces_single_marker():
    sources = [{"id": 1, "title": "Điều 418 BLDS 2015", "url": "https://x.test",
                "content": "Người gây thiệt hại phải bồi thường", "source": "corpus"}]
    html = citation_render.render_answer_html("Theo [1], bồi thường.", sources)
    assert '<sup class="cite">' in html
    assert 'href="#source-1"' in html
    # No hover card in the answer anymore (detail lives in the side panel).
    assert "card" not in html


def test_render_replaces_compound_marker():
    sources = [
        {"id": 1, "title": "A", "url": "https://a.test", "content": "ca", "source": "s"},
        {"id": 3, "title": "C", "url": "https://c.test", "content": "cc", "source": "s"},
    ]
    html = citation_render.render_answer_html("quy định [1][3].", sources)
    # Two separate sup tags.
    assert html.count('<sup class="cite">') == 2
    assert 'href="#source-1"' in html
    assert 'href="#source-3"' in html


def test_render_missing_source_stays_literal():
    sources = [{"id": 1, "title": "A", "url": "https://a.test", "content": "ca", "source": "s"}]
    html = citation_render.render_answer_html("Theo [1] và [99].", sources)
    assert '<sup class="cite">' in html  # [1] replaced
    assert "[99]" in html  # [99] literal, not fabricated


def test_render_escapes_html_in_answer():
    sources = [{"id": 1, "title": "A", "url": "https://a.test", "content": "ca", "source": "s"}]
    html = citation_render.render_answer_html("<script>alert(1)</script> [1]", sources)
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_render_marker_without_url_renders_drawer_link():
    sources = [{"id": 1, "title": "A", "url": "", "content": "ca", "source": "s"}]
    html = citation_render.render_answer_html("Theo [1].", sources)
    assert 'href="#source-1"' in html  # always links to drawer


def test_render_newlines_to_br():
    sources = []
    html = citation_render.render_answer_html("dòng 1\ndòng 2", sources)
    assert "<br>" in html or "<br/>" in html


def test_render_no_sources_returns_escaped_text():
    html = citation_render.render_answer_html("plain answer <b>", [])
    assert "&lt;b&gt;" in html
    assert "<sup" not in html


# ---- render_sources_panel (grouped by kind) ----

def test_render_sources_panel_groups_by_kind():
    sources = [
        {"id": 1, "title": "Điều 418 BLDS 2015", "url": "https://g.test",
         "content": "c1", "source": "corpus.pdf", "kind": "corpus"},
        {"id": 2, "title": "Nghị định 168", "url": "https://vbpl.vn/168",
         "content": "c2", "source": "", "kind": "web"},
        {"id": 3, "title": "Điều 419 BLDS 2015", "url": "https://g2.test",
         "content": "c3", "source": "x.pdf", "kind": "corpus"},
    ]
    panel = citation_render.render_sources_panel(sources)
    # Two group headers with counts.
    assert "📚 Nguồn tài liệu (2)" in panel
    assert "🌐 Kết quả Web search (1)" in panel
    # Corpus section contains cả 2 corpus items, web section contains web item.
    assert "https://g.test" in panel and "https://g2.test" in panel
    assert "https://vbpl.vn/168" in panel
    # Each item keyed by its [n] marker.
    assert "[1]" in panel and "[2]" in panel and "[3]" in panel


def test_render_sources_panel_omits_empty_group():
    sources = [
        {"id": 1, "title": "A", "url": "https://a.test", "content": "c", "source": "", "kind": "web"},
    ]
    panel = citation_render.render_sources_panel(sources)
    assert "🌐 Kết quả Web search (1)" in panel
    assert "Nguồn tài liệu" not in panel  # no corpus -> group omitted


def test_render_sources_panel_empty_note():
    panel = citation_render.render_sources_panel([])
    assert "Không có dẫn chứng" in panel


def test_render_sources_panel_escapes_html():
    sources = [{"id": 1, "title": "<b>A</b>", "url": "https://a.test",
                "content": "<img src=x onerror=alert(1)>", "source": "s", "kind": "corpus"}]
    panel = citation_render.render_sources_panel(sources)
    assert "<img" not in panel
    assert "<b>A</b>" not in panel
    assert "&lt;b&gt;" in panel


def test_render_sources_panel_web_derives_host_from_url():
    sources = [{"id": 1, "title": "T", "url": "https://vbpl.vn/123",
                "content": "c", "source": "", "kind": "web"}]
    panel = citation_render.render_sources_panel(sources)
    assert "vbpl.vn" in panel  # host shown when source empty


def test_render_footnote_keyed_by_id():
    sources = [
        {"id": 1, "title": "Điều 418", "url": "https://a.test", "content": "c", "source": "s"},
        {"id": 2, "title": "Điều 419", "url": "https://b.test", "content": "c2", "source": "s"},
    ]
    foot = citation_render.render_sources_footnote(sources)
    assert "1" in foot and "Điều 418" in foot
    assert "2" in foot and "Điều 419" in foot
    assert "https://a.test" in foot


# ---- render_sources_drawers_html ----

def test_render_sources_drawers_html_corpus():
    sources = [
        {"id": 1, "title": "Điều 418 BLDS 2015", "url": "https://www.google.com/search?q=abc",
         "content": "bồi thường thiệt hại", "source": "Bộ luật Dân sự", "kind": "corpus"},
    ]
    drawers_html = citation_render.render_sources_drawers_html(sources)
    assert 'id="source-1"' in drawers_html
    assert 'class="drawer-overlay"' in drawers_html
    assert 'class="drawer-backdrop"' in drawers_html
    assert '📚 Kho Luật Việt Nam' in drawers_html
    assert 'Điều 418 BLDS 2015' in drawers_html
    assert 'bồi thường thiệt hại' in drawers_html
    # Should NOT have original page view button (since it has a Google Search fallback URL)
    assert 'Xem trang gốc' not in drawers_html
    assert 'google.com/search' not in drawers_html


def test_render_sources_drawers_html_web():
    sources = [
        {"id": 2, "title": "Nghị định 168", "url": "https://vbpl.vn/168",
         "content": "quy định chi tiết", "source": "", "kind": "web"},
    ]
    drawers_html = citation_render.render_sources_drawers_html(sources)
    assert 'id="source-2"' in drawers_html
    assert '🌐 Tìm kiếm Web' in drawers_html
    # Should have original page view button for web source
    assert 'Xem trang gốc' in drawers_html
    assert 'href="https://vbpl.vn/168"' in drawers_html