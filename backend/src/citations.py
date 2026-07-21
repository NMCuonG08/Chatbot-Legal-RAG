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
    "\n\n6. DẪN CHỨNG BẮT BUỘC & QUY TẮC TRÌNH BÀY:\n"
    "- Mỗi khẳng định pháp luật / số điều / khoản / quy định phải kết thúc bằng marker `[n]` tương ứng với số thứ tự tài liệu được cung cấp. Ví dụ: '...phải bồi thường thiệt hại [1]'. Nếu một câu dựa trên nhiều nguồn, ghi liền: '[1][3]'. CHỈ dùng số tài liệu thực sự được cung cấp, KHÔNG được bịa số.\n"
    "- TUYỆT ĐỐI KHÔNG tự viết hay liệt kê phần/mục 'Tài liệu tham khảo' ở cuối câu trả lời. Hệ thống UI sẽ tự động hiển thị danh sách dẫn chứng và liên kết. Bạn CHỈ chèn các marker `[n]` trong câu trả lời."
)


_GOOGLE_SEARCH_URL = "https://www.google.com/search?q="


def _citation_label(
    law_name: str,
    article_number: Optional[int],
    clause_number: Optional[int] = None,
    point_letter: Optional[str] = None,
) -> Optional[str]:
    """Build the most precise Vietnamese legal citation label available.

    Composes ``"Điều {a} khoản {k} điểm {p} {law}"`` omitting any segment whose
    value is missing, so a chunk that only cites an article still yields
    ``"Điều 418 Bộ luật Dân sự 2015"``. Returns ``None`` when neither article
    nor law_name is present.
    """
    if article_number is None and not law_name:
        return None
    segs: List[str] = []
    if article_number is not None:
        segs.append(f"Điều {article_number}")
    if clause_number is not None:
        segs.append(f"khoản {clause_number}")
    if point_letter:
        segs.append(f"điểm {point_letter}")
    if law_name:
        segs.append(law_name)
    return " ".join(segs).strip() if segs else None


def build_search_url(
    law_name: Optional[str],
    article_number: Optional[int],
    question: Optional[str],
    content: Optional[str],
    clause_number: Optional[int] = None,
    point_letter: Optional[str] = None,
) -> Optional[str]:
    """Build a Google search URL for a corpus chunk that has no real URL.

    Query priority (first non-empty wins):
      1. ``"Điều {article_number} khoản {k} điểm {p} {law_name}"`` — most
         precise legal lookup; clause/point included only when present.
      2. ``law_name`` alone.
      3. ``question`` (the chunk's original Q).
      4. first 60 chars of ``content``.

    Returns ``None`` when nothing usable is available so the caller can render
    a no-link popup instead of a broken search.
    """
    law_name = (law_name or "").strip()
    question = (question or "").strip()
    content = (content or "").strip()

    label = _citation_label(law_name, article_number, clause_number, point_letter)
    if label:
        return _GOOGLE_SEARCH_URL + quote_plus(label)
    if law_name:
        return _GOOGLE_SEARCH_URL + quote_plus(law_name)
    if question:
        return _GOOGLE_SEARCH_URL + quote_plus(question)
    if content:
        return _GOOGLE_SEARCH_URL + quote_plus(content[:60])
    return None


def _source_title(doc: Dict[str, Any]) -> str:
    """Human-readable title for a source card.

    Priority: full citation label (Điều/khoản/điểm + law_name) > ``law_name``
    alone > ``question`` (truncated) > ``source`` (filename) > generic fallback.
    """
    law_name = (doc.get("law_name") or "").strip()
    article_number = doc.get("article_number")
    clause_number = doc.get("clause_number")
    point_letter = doc.get("point_letter")
    label = _citation_label(law_name, article_number, clause_number, point_letter)
    if label:
        return label
    if law_name:
        return law_name
    question = (doc.get("question") or "").strip()
    if question:
        return question[:80]
    source = (doc.get("source") or "").strip()
    if source and source != "unknown":
        return source
    return "Tài liệu tham khảo"


import re


def strip_trailing_references(text: str) -> str:
    """Strip any trailing 'Tài liệu tham khảo:' list generated by LLMs in markdown.

    The frontend system panel / drawer handles rendering the full source cards,
    so an LLM-emitted reference list at the bottom of the answer creates ugly
    duplication and non-clickable text.
    """
    if not text:
        return ""

    # Match trailing reference list section headers (e.g. **Tài liệu tham khảo:**)
    pattern = r"(?i)\n+ *(?:\*\*|###*|##*)? *(?:Tài liệu tham khảo|Nguồn tham khảo|Danh mục (?:tài liệu )?tham khảo) *(?:\*\*|:)? *[\s\S]*$"

    # Preserve any legal disclaimer (*Lưu ý: ...*) if present at the very end
    disclaimer = ""
    disclaimer_match = re.search(r"(\n+ *\*?Lưu ý:[\s\S]*\*?)$", text, re.IGNORECASE)
    if disclaimer_match:
        disclaimer = disclaimer_match.group(1)

    cleaned = re.sub(pattern, "", text)
    if disclaimer and disclaimer.strip() not in cleaned:
        cleaned = cleaned.strip() + "\n\n" + disclaimer.strip()

    return cleaned.strip()


def normalize_sources(docs: List[Dict[str, Any]], kind: str = "corpus") -> List[Dict[str, Any]]:
    """Stamp citation metadata onto a list of retrieved source chunks.

    Adds (does not remove) these fields per doc:
      - ``id``: 1-based index in ``docs``. MUST match the ``[Tài liệu n]``
        numbering produced by ``gen_doc_prompt`` — both consume the same list
        in the same order, so the LLM's ``[n]`` markers resolve to the right
        source on the frontend.
      - ``title``: from ``_source_title``.
      - ``url``: priority: exact ``url`` / ``origin_url`` / ``link`` if present;
        otherwise fallback Google search link synthesized via ``build_search_url``.
      - ``kind``: ``"corpus"`` | ``"web"``.

    Existing fields (content/source/doc_id/score/law_name/article_number/
    content_type) are preserved. Returns a new list of new dicts (no mutation).
    """
    out: List[Dict[str, Any]] = []
    for idx, doc in enumerate(docs, start=1):
        item = dict(doc)  # copy — never mutate caller's dicts
        item["id"] = idx
        item["title"] = item.get("title") or _source_title(doc)
        item["kind"] = item.get("kind") or kind

        existing_url = item.get("url") or item.get("origin_url") or item.get("link")
        if existing_url:
            item["url"] = existing_url
        else:
            item["url"] = build_search_url(
                doc.get("law_name"), doc.get("article_number"),
                doc.get("question"), doc.get("content"),
                doc.get("clause_number"), doc.get("point_letter"),
            )
        out.append(item)
    return out