"""Inline-citation helpers for the legal chatbot.

Goal: every legal answer must carry inline ``[n]`` citation markers tied to
specific source chunks, so the frontend can render a hover popup (source card
+ link) per claim instead of dumping an unrelated source list below the answer.

This module owns the *backend* side of the contract:
- ``CITATION_RULE``: prompt instruction appended to every RAG/web system prompt
  so the LLM emits ``[n]`` markers matching the numbered docs it was given.
- ``build_search_url``: corpus chunks have no URL — synthesize a Google search
  link from law_name + article_number (fallback question / content) so every
  citation is still clickable ("link search cũng cần").
- ``normalize_sources``: stamp a stable 1-based ``id`` (matching the numbering
  in ``gen_doc_prompt``), a human ``title``, a ``url``, and a ``kind`` tag.
  Preserves all existing fields (content/doc_id/score/law_name/...).

The *frontend* side (HTML popover rendering) lives in
``frontend/citation_render.py``; the contract between them is the source dict
schema documented on ``normalize_sources``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from urllib.parse import quote_plus

logger = logging.getLogger(__name__)

# Appended to RAG / web system prompts. The docs in the user message are
# numbered ``[Tài liệu n]`` by ``gen_doc_prompt``; this rule tells the LLM to
# echo that number inline at each legal claim. Multiple sources -> ``[1][3]``.
CITATION_RULE: str = (
    "\n\n6. DẪN CHỨNG BẮT BUỘC: Mỗi khẳng định pháp luật / số điều / khoản / quy "
    "định phải kết thúc bằng marker `[n]` với n là số thứ tự tài liệu trong "
    "'Tài liệu tham khảo' bên dưới. Ví dụ: '...phải bồi thường thiệt hại [1]'. "
    "Nếu một câu khẳng định dựa trên nhiều nguồn, ghi liền: '[1][3]'. CHỈ dùng "
    "số của tài liệu thực sự được cung cấp, KHÔNG được bịa số, KHÔNG được bịa "
    "số điều / nội dung không có trong tài liệu. Nếu không có tài liệu liên "
    "quan, nói rõ và KHÔNG ghi marker."
)

_GOOGLE_SEARCH_URL = "https://www.google.com/search?q="


def build_search_url(
    law_name: Optional[str],
    article_number: Optional[int],
    question: Optional[str],
    content: Optional[str],
) -> Optional[str]:
    """Build a Google search URL for a corpus chunk that has no real URL.

    Query priority (first non-empty wins):
      1. ``"Điều {article_number} {law_name}"`` — when both present, the most
         precise legal lookup.
      2. ``law_name`` alone.
      3. ``question`` (the chunk's original Q).
      4. first 60 chars of ``content``.

    Returns ``None`` when nothing usable is available so the caller can render
    a no-link popup instead of a broken search.
    """
    parts: List[str] = []
    law_name = (law_name or "").strip()
    question = (question or "").strip()
    content = (content or "").strip()

    if article_number and law_name:
        parts.append(f"Điều {article_number} {law_name}")
    elif law_name:
        parts.append(law_name)
    elif question:
        parts.append(question)
    elif content:
        parts.append(content[:60])

    if not parts:
        return None
    return _GOOGLE_SEARCH_URL + quote_plus(parts[0])


def _source_title(doc: Dict[str, Any]) -> str:
    """Human-readable title for a source card.

    Priority: ``law_name + article_number`` > ``question`` (truncated) >
    ``source`` (filename) > generic fallback.
    """
    law_name = (doc.get("law_name") or "").strip()
    article_number = doc.get("article_number")
    if law_name and article_number is not None:
        return f"Điều {article_number} {law_name}".strip()
    if law_name:
        return law_name
    question = (doc.get("question") or "").strip()
    if question:
        return question[:80]
    source = (doc.get("source") or "").strip()
    if source and source != "unknown":
        return source
    return "Tài liệu tham khảo"


def normalize_sources(docs: List[Dict[str, Any]], kind: str = "corpus") -> List[Dict[str, Any]]:
    """Stamp citation metadata onto a list of retrieved source chunks.

    Adds (does not remove) these fields per doc:
      - ``id``: 1-based index in ``docs``. MUST match the ``[Tài liệu n]``
        numbering produced by ``gen_doc_prompt`` — both consume the same list
        in the same order, so the LLM's ``[n]`` markers resolve to the right
        source on the frontend.
      - ``title``: from ``_source_title``.
      - ``url``: for ``kind="web"`` the chunk's real URL is kept; for
        ``kind="corpus"`` a Google search link is synthesized via
        ``build_search_url``.
      - ``kind``: ``"corpus"`` | ``"web"``.

    Existing fields (content/source/doc_id/score/law_name/article_number/
    content_type) are preserved. Returns a new list of new dicts (no mutation).
    """
    out: List[Dict[str, Any]] = []
    for idx, doc in enumerate(docs, start=1):
        item = dict(doc)  # copy — never mutate caller's dicts
        item["id"] = idx
        item["title"] = item.get("title") or _source_title(doc)
        item["kind"] = kind
        if kind == "web":
            # Real URL from Tavily; only synthesize if missing.
            if not item.get("url"):
                item["url"] = build_search_url(
                    doc.get("law_name"), doc.get("article_number"),
                    doc.get("question"), doc.get("content"),
                )
        else:
            item["url"] = item.get("url") or build_search_url(
                doc.get("law_name"), doc.get("article_number"),
                doc.get("question"), doc.get("content"),
            )
        out.append(item)
    return out