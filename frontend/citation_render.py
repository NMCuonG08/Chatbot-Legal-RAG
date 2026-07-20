"""Frontend renderer: inline ``[n]`` citation links in the answer + a grouped
sources panel for the right-hand column.

Design (revised after hover-popover feedback — popovers were hard to see):
- The answer keeps inline ``[n]`` markers rendered as small clickable superscript
  links opening the source URL in a new tab. No hover popup.
- The *source detail* lives in a separate panel rendered beside the answer
  (Streamlit right column / Gradio below the answer). The panel groups sources
  by ``kind`` so the user can tell web-search results apart from in-corpus legal
  documents at a glance:
    📚 Nguồn tài liệu   (kind="corpus" — chunks from the vector store; link is a
                        Google search for the law/article since the corpus has
                        no real URL)
    🌐 Kết quả Web search (kind="web" — Tavily results; link is the real page URL)

Contract — each source dict (from ``backend/src/citations.normalize_sources``):
    {
      "id": 1, "title": "Điều 418 BLDS 2015", "url": "https://...",
      "kind": "corpus" | "web", "content": "...", "source": "...",
      # ...law_name, article_number, doc_id, score, content_type
    }

No backend imports — pure so it can be imported from both the Streamlit
frontend and the Gradio ``backend/chat_ui.py``.
"""

from __future__ import annotations

import html
import re
from typing import Any, Dict, List

# One ``[n]`` marker at a time. Run on HTML-escaped text so brackets survive.
_MARKER_RE = re.compile(r"\[(\d+)\]")

CITATION_CSS: str = """
<style>
/* Inline citation markers in the answer */
.cite > a {
  color: var(--st-primary-color, #2563eb);
  text-decoration: none;
  font-weight: 700;
  font-size: 0.85em;
  vertical-align: super;
  padding: 1px 4px;
  background-color: var(--st-secondary-background-color, #eff6ff);
  border: 1px solid var(--st-secondary-background-color, #bfdbfe);
  border-radius: 4px;
  margin-left: 2px;
  transition: all 0.2s ease;
}
.cite > a:hover {
  background-color: var(--st-primary-color, #2563eb);
  color: var(--st-background-color, #ffffff) !important;
  border-color: var(--st-primary-color, #2563eb);
  text-decoration: none;
}

/* Right-hand sources panel */
.src-panel { font-size: 13px; line-height: 1.5; }
.src-group { margin-bottom: 14px; }
.src-group-h {
  font-weight: 700; font-size: 13px; color: var(--st-text-color, #111);
  border-bottom: 2px solid var(--st-secondary-background-color, #e5e7eb);
  padding-bottom: 4px; margin-bottom: 8px;
}
.src-group-h.web { border-bottom-color: #f59e0b; }
.src-group-h.corpus { border-bottom-color: var(--st-primary-color, #2563eb); }
.src-item {
  border: 1px solid var(--st-secondary-background-color, #e5e7eb);
  border-radius: 8px; padding: 8px 10px;
  margin-bottom: 8px; background: var(--st-background-color, #fff);
}
.src-item.web { border-left: 3px solid #f59e0b; }
.src-item.corpus { border-left: 3px solid var(--st-primary-color, #2563eb); }
.src-n {
  display: inline-block; min-width: 22px; padding: 0 5px;
  background: var(--st-primary-color, #2563eb); color: var(--st-background-color, #fff);
  border-radius: 5px; font-size: 11px; font-weight: 700; margin-right: 6px;
}
.src-item.web .src-n { background: #f59e0b; }
.src-title { font-weight: 600; color: var(--st-text-color, #111); }
.src-title a { color: var(--st-primary-color, #2563eb); text-decoration: none; }
.src-title a:hover { text-decoration: underline; }
.src-origin { color: #6b7280; font-size: 11px; margin-top: 3px; }
.src-snippet {
  color: var(--st-text-color, #374151);
  background: var(--st-secondary-background-color, #f6f8fa);
  border-radius: 6px; padding: 5px 7px; margin-top: 5px; font-size: 12px;
  max-height: 96px; overflow: auto; white-space: normal;
}
.src-link { display: block; margin-top: 5px; color: var(--st-primary-color, #2563eb);
  text-decoration: none; font-size: 11px; word-break: break-all; }
.src-link:hover { text-decoration: underline; }
.src-empty { color: #9ca3af; font-style: italic; font-size: 12px; }

/* Drawer overlay container */
.drawer-overlay {
  position: fixed;
  top: 0;
  left: 0;
  width: 100vw;
  height: 100vh;
  visibility: hidden;
  opacity: 0;
  transition: opacity 0.25s ease, visibility 0.25s ease;
  z-index: 999999;
}
.drawer-overlay:target {
  visibility: visible;
  opacity: 1;
}

/* Drawer backdrop (clicking closes drawer) */
.drawer-backdrop {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: rgba(0, 0, 0, 0.55);
  backdrop-filter: blur(2px);
  cursor: default;
}

/* Drawer content slide-in panel */
.drawer-content {
  position: absolute;
  top: 0;
  right: -460px;
  width: 440px;
  max-width: 90vw;
  height: 100%;
  background: var(--st-background-color, #ffffff);
  box-shadow: -10px 0 25px -5px rgba(0, 0, 0, 0.25);
  transition: right 0.25s cubic-bezier(0.4, 0, 0.2, 1);
  padding: 28px 24px;
  box-sizing: border-box;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  color: var(--st-text-color, #1e293b);
  border-left: 1px solid var(--st-secondary-background-color, #e2e8f0);
  font-family: var(--font, system-ui, -apple-system, sans-serif);
}
.drawer-overlay:target .drawer-content {
  right: 0;
}

/* Header inside drawer */
.drawer-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  border-bottom: 1px solid var(--st-secondary-background-color, #e2e8f0);
  padding-bottom: 14px;
  margin-bottom: 18px;
}
.drawer-badge {
  font-size: 11px;
  text-transform: uppercase;
  font-weight: 700;
  letter-spacing: 0.05em;
  padding: 4px 8px;
  border-radius: 12px;
}
.drawer-badge.corpus {
  background-color: rgba(37, 99, 235, 0.1);
  color: var(--st-primary-color, #2563eb);
}
.drawer-badge.web {
  background-color: rgba(245, 158, 11, 0.1);
  color: #f59e0b;
}
.drawer-close {
  font-size: 26px;
  font-weight: 300;
  color: #64748b;
  text-decoration: none !important;
  cursor: pointer;
  line-height: 1;
  transition: color 0.15s ease;
}
.drawer-close:hover {
  color: var(--st-primary-color, #2563eb);
}

/* Title & Origin */
.drawer-title {
  font-size: 18px;
  font-weight: 700;
  color: var(--st-text-color, #0f172a);
  margin-top: 0;
  margin-bottom: 8px;
  line-height: 1.35;
}
.drawer-origin {
  font-size: 13px;
  color: #64748b;
  margin-bottom: 18px;
}

/* Section Title */
.drawer-section-title {
  font-size: 11px;
  font-weight: 600;
  color: #64748b;
  text-transform: uppercase;
  margin-bottom: 8px;
  letter-spacing: 0.03em;
}

/* Content Snippet inside drawer */
.drawer-snippet {
  background-color: var(--st-secondary-background-color, #f8fafc);
  border: 1px solid var(--st-secondary-background-color, #e2e8f0);
  border-left: 4px solid var(--st-primary-color, #2563eb);
  border-radius: 8px;
  padding: 16px;
  color: var(--st-text-color, #334155);
  font-size: 14px;
  line-height: 1.6;
  margin-bottom: 24px;
  white-space: pre-wrap;
}
.drawer-snippet.web {
  border-left-color: #f59e0b;
}

/* View original button */
.drawer-btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  padding: 10px 16px;
  background-color: var(--st-primary-color, #2563eb);
  color: var(--st-background-color, #ffffff) !important;
  font-weight: 600;
  font-size: 13px;
  text-decoration: none !important;
  border-radius: 6px;
  transition: background-color 0.15s ease;
  box-shadow: 0 2px 4px rgba(37, 99, 235, 0.15);
}
.drawer-btn:hover {
  opacity: 0.9;
}
</style>
"""


def _esc(s: Any) -> str:
    """HTML-escape a string for safe insertion into text/attribute content."""
    return html.escape(str(s) if s is not None else "")


def _snippet(content: Any, limit: int = 200) -> str:
    raw = (str(content) if content is not None else "").strip()
    if len(raw) > limit:
        raw = raw[:limit] + "…"
    return _esc(raw)


def render_answer_html(answer: str, sources: List[Dict[str, Any]]) -> str:
    """Render an answer with inline ``[n]`` citation superscript-links pointing to #source-n.

    - Answer text is HTML-escaped (LLM output -> untrusted); ``[n]`` markers
      survive (brackets are not escaped) and are replaced with
      ``<sup class="cite"><a href="#source-n">[n]</a></sup>``.
    - A marker whose number has no matching source is left as literal ``[n]``
      (never fabricate a citation / link).
    - ``\\n`` -> ``<br>`` so multi-paragraph legal answers keep line breaks.
    """
    if answer is None:
        answer = ""
    by_id: Dict[int, Dict[str, Any]] = {
        int(s["id"]): s for s in (sources or []) if s.get("id") is not None
    }
    text = _esc(answer).replace("\n", "<br>")

    def _repl(m: "re.Match[str]") -> str:
        n = int(m.group(1))
        src = by_id.get(n)
        if src is None:
            return m.group(0)  # literal [n]
        return f'<sup class="cite"><a href="#source-{n}">[{n}]</a></sup>'

    return _MARKER_RE.sub(_repl, text)


def _item_html(src: Dict[str, Any]) -> str:
    n = src.get("id", "?")
    kind = src.get("kind") or "corpus"
    title = _esc(src.get("title") or "Tài liệu tham khảo")
    url = src.get("url") or ""
    origin = src.get("source") or ""
    if kind == "web":
        # For web, "source" is usually empty; derive a host hint from the url.
        if not origin:
            try:
                from urllib.parse import urlparse
                origin = urlparse(url).netloc or "web"
            except Exception:
                origin = "web"
        kind_label = "Web"
    else:
        kind_label = "Kho luật"
    origin_line = (
        f'<div class="src-origin">Nguồn: {_esc(kind_label)}'
        + (f" · {_esc(origin)}" if origin and origin != kind_label else "")
        + "</div>"
    )
    snippet_html = f'<div class="src-snippet">{_snippet(src.get("content"))}</div>'
    if url:
        title_html = (
            f'<span class="src-title"><a href="{_esc(url)}" target="_blank" '
            f'rel="noopener noreferrer">{title}</a></span>'
        )
        link_html = f'<a class="src-link" href="{_esc(url)}" target="_blank" rel="noopener noreferrer">{_esc(url)}</a>'
    else:
        title_html = f'<span class="src-title">{title}</span>'
        link_html = ""
    return (
        f'<div class="src-item {kind}">'
        f'<span class="src-n">[{n}]</span>'
        f'{title_html}'
        f'{origin_line}'
        f'{snippet_html}'
        f'{link_html}'
        f'</div>'
    )


def render_sources_panel(sources: List[Dict[str, Any]]) -> str:
    """Grouped sources panel HTML, split by ``kind`` (corpus vs web).

    Each group has a colored header with count; each source is a card keyed by
    its ``[n]`` marker (matching the inline markers in the answer) with title,
    origin, snippet, and link. Empty groups are omitted; an empty sources list
    yields a muted "no sources" note.
    """
    sources = sources or []
    if not sources:
        return '<div class="src-panel"><div class="src-empty">Không có dẫn chứng cho câu trả lời này.</div></div>'

    corpus = [s for s in sources if (s.get("kind") or "corpus") == "corpus"]
    web = [s for s in sources if s.get("kind") == "web"]

    parts: List[str] = ['<div class="src-panel">']
    if corpus:
        parts.append(
            f'<div class="src-group"><div class="src-group-h corpus">'
            f'📚 Nguồn tài liệu ({len(corpus)})</div>'
            + "".join(_item_html(s) for s in corpus)
            + "</div>"
        )
    if web:
        parts.append(
            f'<div class="src-group"><div class="src-group-h web">'
            f'🌐 Kết quả Web search ({len(web)})</div>'
            + "".join(_item_html(s) for s in web)
            + "</div>"
        )
    parts.append("</div>")
    return "".join(parts)


def render_sources_footnote(sources: List[Dict[str, Any]]) -> str:
    """Backward-compatible compact footnote (kept for callers that want a
    one-line summary instead of the full panel). Grouped by kind, keyed by [n]."""
    sources = sources or []
    if not sources:
        return ""
    items: List[str] = []
    for src in sources:
        n = src.get("id", "?")
        title = _esc(src.get("title") or "Tài liệu")
        url = src.get("url") or ""
        if url:
            items.append(
                f'<span>[{n}] <a href="{_esc(url)}" target="_blank" rel="noopener noreferrer">{title}</a></span>'
            )
        else:
            items.append(f"<span>[{n}] {title}</span>")
    return '<div class="cite-footnote">' + " · ".join(items) + "</div>"


def render_sources_drawers_html(sources: List[Dict[str, Any]]) -> str:
    """Generate the hidden HTML overlay drawers for each source.
    
    Clicking a citation link like `[n]` will jump to `#source-{n}`, which makes
    the drawer visible via CSS target. Clicking outside or clicking the close button
    jumps to `#` which hides it.
    """
    sources = sources or []
    if not sources:
        return ""
        
    drawers = []
    for src in sources:
        n = src.get("id", "?")
        kind = src.get("kind") or "corpus"
        title = _esc(src.get("title") or "Dẫn chứng pháp lý")
        url = src.get("url") or ""
        origin = src.get("source") or ""
        content = src.get("content") or ""
        
        # Determine badges and headers
        if kind == "web":
            badge_html = '<span class="drawer-badge web">🌐 Tìm kiếm Web</span>'
            origin_label = "Web search result"
            if not origin and url:
                try:
                    from urllib.parse import urlparse
                    origin = urlparse(url).netloc or "web"
                except Exception:
                    origin = "web"
        else:
            badge_html = '<span class="drawer-badge corpus">📚 Kho Luật Việt Nam</span>'
            origin_label = "Tài liệu Corpus"
            
        origin_line = f"Nguồn: <strong>{_esc(origin or origin_label)}</strong>"
        
        # Build the Action Button (only for real web URLs, ignore Google Search fallback for corpus)
        action_btn_html = ""
        # Check if the URL is a real web URL (doesn't contain Google search query)
        is_real_url = url and ("google.com/search" not in url)
        if kind == "web" and is_real_url:
            action_btn_html = (
                f'<div style="margin-top: 24px;">'
                f'<a class="drawer-btn" href="{_esc(url)}" target="_blank" rel="noopener noreferrer">'
                f'🌐 Xem trang gốc'
                f'</a>'
                f'</div>'
            )
            
        drawer_html = (
            f'<div id="source-{n}" class="drawer-overlay">'
            f'  <a href="#" class="drawer-backdrop"></a>'
            f'  <div class="drawer-content">'
            f'    <div class="drawer-header">'
            f'      {badge_html}'
            f'      <a href="#" class="drawer-close">&times;</a>'
            f'    </div>'
            f'    <h3 class="drawer-title">{title}</h3>'
            f'    <div class="drawer-origin">{origin_line}</div>'
            f'    <div class="drawer-body">'
            f'      <div class="drawer-section-title">Nội dung trích dẫn:</div>'
            f'      <div class="drawer-snippet {kind}">{_esc(content)}</div>'
            f'    </div>'
            f'    {action_btn_html}'
            f'  </div>'
            f'</div>'
        )
        drawers.append(drawer_html)
        
    return "\n".join(drawers)