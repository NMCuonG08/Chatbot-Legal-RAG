"""
Retrieval-backed legal tools for the Vietnamese law chatbot agent.

These tools query the Qdrant vector store (via ``hybrid_search`` / ``search_vector``)
to fetch real legal text. They are "best-effort": the corpus payload only carries
``question``, ``content``, ``source``, and ``content_type`` fields — there is no
structured ``law_name`` / ``article_number`` metadata yet — so article lookup is
semantic with an optional ``content_type`` filter, not an exact index lookup.

Keeping retrieval tools in a separate module from ``legal_tools.py`` avoids
import-time coupling to ``search`` / ``brain`` / ``vectorize`` for the pure
calc tools, and respects the 800-line file-size guideline.
"""

import logging
import re
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Punctuation-aware tokenizer: ``\w+`` with re.UNICODE captures Vietnamese
# diacritic letters + digits, stripping trailing punctuation ("phạm." → "phạm").
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

# Common Vietnamese function words that carry little legal signal. Removing
# them raises the signal/noise of the Jaccard overlap so paraphrases of the
# same article compare closer, and unrelated text compares further.
_VI_STOPWORDS = frozenset({
    "và", "của", "là", "hoặc", "thì", "được", "có", "không", "với", "cho",
    "trong", "đến", "từ", "một", "các", "những", "khi", "nếu", "sẽ", "đã",
    "đang", "này", "đó", "về", "bị", "để", "mà", "cũng", "như", "nhưng",
    "theo", "tại", "trên", "dưới", "về", "lại", "còn", "phải", "vừa",
})


def _tokens(s: str) -> set:
    """Lowercase, split on non-word chars, drop stopwords + tokens ≤2 chars."""
    return set(
        t for t in _TOKEN_RE.findall(s.lower())
        if len(t) > 2 and t not in _VI_STOPWORDS
    )

# Lazily imported so importing this module never triggers Qdrant/LLM setup.
# Tests and cold import paths stay light.


def _hybrid_search(query: str, limit: int = 5) -> List[Dict]:
    """Thin lazy wrapper around ``search.hybrid_search``."""
    try:
        from search import hybrid_search  # local import avoids circular at module load
        return hybrid_search(query, limit=limit)
    except Exception as e:  # pragma: no cover - environment-dependent
        logger.error(f"[RETRIEVAL] hybrid_search failed: {e}")
        return []


def _vector_search(query: str, limit: int = 5, filters: Optional[Dict] = None) -> List[Dict]:
    """Lazy vector-only search with optional payload filter (e.g. content_type)."""
    try:
        from brain import get_embedding
        from config import DEFAULT_COLLECTION_NAME
        from vectorize import search_vector
        vec = get_embedding(query)
        if vec is None:
            return []
        return search_vector(DEFAULT_COLLECTION_NAME, vec, limit=limit, filters=filters, score_threshold=0.2)
    except Exception as e:  # pragma: no cover - environment-dependent
        logger.error(f"[RETRIEVAL] vector search failed: {e}")
        return []


def _format_hits(hits: List[Dict], max_chars: int = 600) -> List[Dict]:
    """Normalize hit dicts into a compact, LLM-friendly shape."""
    out = []
    for h in hits:
        content = (h.get("content") or "").strip()
        if not content:
            continue
        if len(content) > max_chars:
            content = content[:max_chars] + "…"
        item = {
            "question": (h.get("question") or "").strip(),
            "content": content,
            "source": h.get("source", "unknown"),
            "doc_id": h.get("doc_id"),
            "score": h.get("hybrid_score") or h.get("similarity_score") or 0.0,
        }
        # Surface structured metadata when the chunk carries it (post-enrichment).
        if h.get("law_name"):
            item["law_name"] = h.get("law_name")
        if h.get("article_number") is not None:
            item["article_number"] = h.get("article_number")
        out.append(item)
    return out


def lookup_article(law_name: str, article_number: Optional[int] = None, limit: int = 5) -> Dict:
    """
    Tra cứu nội dung một điều/chương của văn bản luật trong kho dữ liệu.

    Lưu ý: kho dữ liệu hiện CHƯA có metadata cấu trúc (law_name / article_number),
    nên tra cứu là semantic + lọc ``content_type='law'``. Kết quả là đoạn văn bản
    khớp nhất, KHÔNG đảm bảo đúng điều nếu corpus không chứa văn bản đó.

    Args:
        law_name: Tên luật/bộ luật, ví dụ "Bộ luật Dân sự 2015", "Luật Đất đai 2024".
        article_number: Số điều cần tra, ví dụ 418. Có thể bỏ trống để tra cả luật.
        limit: Số kết quả tối đa (mặc định 5).

    Returns:
        Dict với danh sách đoạn văn bản luật khớp + nguồn, hoặc thông báo không tìm thấy.
    """
    law_name = (law_name or "").strip()
    if not law_name:
        return {"error": "Cần truyền tên luật (law_name) để tra cứu."}

    query = f"Điều {article_number} {law_name}" if article_number else law_name

    # Exact-field path: when article_number is known, filter Qdrant by the
    # structured metadata enriched at ingest (see legal_metadata.extract_legal_metadata).
    # Falls back to semantic content_type='law' search, then to hybrid.
    hits: List[Dict] = []
    if article_number:
        exact_filters: Dict = {"content_type": "law", "article_number": article_number}
        if law_name:
            exact_filters["law_name"] = law_name
        hits = _vector_search(query, limit=limit, filters=exact_filters)

    if not hits:
        hits = _vector_search(query, limit=limit, filters={"content_type": "law"})
    if not hits:
        hits = _hybrid_search(query, limit=limit)

    formatted = _format_hits(hits)
    if not formatted:
        return {
            "found": False,
            "law_name": law_name,
            "article_number": article_number,
            "message": (
                "Không tìm thấy văn bản phù hợp trong kho. Có thể corpus chưa chứa "
                "luật/điều này — nên dùng web_search để tra luật mới."
            ),
        }

    # Exact-match = first hit carries the requested article_number (enriched metadata).
    exact_match = bool(article_number) and any(
        m.get("article_number") == article_number for m in formatted
    )

    return {
        "found": True,
        "law_name": law_name,
        "article_number": article_number,
        "exact_match": exact_match,
        "matches": formatted,
        "count": len(formatted),
        "note": (
            "Khớp chính xác theo metadata law_name/article_number khi exact_match=true; "
            "nếu false, kết quả là semantic + lọc content_type='law' — xác nhận lại số điều "
            "với nguồn trước khi trích dẫn cho user."
        ),
    }


def precedent_lookup(fact_pattern: str, limit: int = 5) -> Dict:
    """
    Tra cứu án lệ (case law) liên quan đến một tình huống.

    Án lệ VN (từ 2015) được lưu trong corpus với ``content_type='case_law'``.
    Dùng khi user hỏi "trường hợp của tôi thì bị xử sao" / "có án lệ nào…".

    Args:
        fact_pattern: Tình huống thực tế, ví dụ "vay tiền không trả có giấy vay".
        limit: Số kết quả tối đa.

    Returns:
        Dict danh sách án lệ khớp + nguồn.
    """
    fact_pattern = (fact_pattern or "").strip()
    if not fact_pattern:
        return {"error": "Cần mô tả tình huống (fact_pattern) để tra án lệ."}

    hits = _vector_search(fact_pattern, limit=limit, filters={"content_type": "case_law"})
    if not hits:
        hits = _hybrid_search(fact_pattern, limit=limit)

    formatted = _format_hits(hits)
    if not formatted:
        return {
            "found": False,
            "message": "Không tìm thấy án lệ phù hợp trong kho. Có thể dùng web_search để tìm thêm.",
        }

    return {
        "found": True,
        "fact_pattern": fact_pattern,
        "precedents": formatted,
        "count": len(formatted),
        "note": "Án lệ chỉ mang tính tham khảo, mỗi vụ việc tùy tình tiết cụ thể.",
    }


def cross_reference(law_name: str, article_number: int, limit: int = 5) -> Dict:
    """
    Tìm các điều/văn bản dẫn chiếu ĐẾN một điều luật cụ thể.

    Ví dụ: "Điều 418 BLDS 2015 dẫn chiếu điều nào?" → tìm corpus chứa
    chuỗi "Điều 418" + tên luật để phát hiện dẫn chiếu.

    Args:
        law_name: Tên luật chứa điều đó, ví dụ "Bộ luật Dân sự 2015".
        article_number: Số điều, ví dụ 418.
        limit: Số kết quả.

    Returns:
        Dict các đoạn văn bản dẫn chiếu điều này.
    """
    law_name = (law_name or "").strip()
    if not law_name or not article_number:
        return {"error": "Cần law_name và article_number để tra dẫn chiếu."}

    query = f"Điều {article_number} {law_name}"

    # Use broader hybrid (no content_type filter) since references appear in any doc type.
    hits = _hybrid_search(query, limit=limit)
    formatted = _format_hits(hits)

    # Heuristic: keep only hits whose content actually mentions "Điều {article_number}".
    needle = f"Điều {article_number}"
    referencing = [h for h in formatted if needle in h.get("content", "")]

    return {
        "law_name": law_name,
        "article_number": article_number,
        "referencing_texts": referencing,
        "count": len(referencing),
        "note": (
            f"Best-effort: lọc các đoạn thực sự nhắc '{needle}'. Corpus thiếu đồ thị "
            "dẫn chiếu cấu trúc nên có thể bỏ sót dẫn chiếu gián tiếp."
        ),
    }


def verify_citation(law_name: str, article_number: int, claimed_text: str) -> Dict:
    """
    Xác minh một trích dẫn pháp luật có khớp với văn bản thật trong corpus không.

    Anti-hallucination guard cho domain pháp luật: trước khi agent khẳng định
    "Điều X Luật Y nói Z", gọi tool này để so Z với nội dung thật.

    Args:
        law_name: Tên luật, ví dụ "Bộ luật Dân sự 2015".
        article_number: Số điều, ví dụ 418.
        claimed_text: Nội dung agent/user claim điều đó nói.

    Returns:
        Dict: verified (bool), overlap_ratio, best_match, verdict.
    """
    law_name = (law_name or "").strip()
    claimed_text = (claimed_text or "").strip()
    if not law_name or not article_number or not claimed_text:
        return {"error": "Cần law_name, article_number, claimed_text."}

    result = lookup_article(law_name, article_number, limit=3)
    if not result.get("found"):
        return {
            "verified": False,
            "verdict": "cannot_verify",
            "reason": "Corpus không chứa văn bản luật/điều này — không thể xác minh. Đề nghị web_search.",
            "law_name": law_name,
            "article_number": article_number,
        }

    matches = result.get("matches", [])

    claimed_tokens = _tokens(claimed_text)
    best = None
    best_ratio = 0.0
    for m in matches:
        toks = _tokens(m.get("content", ""))
        if not claimed_tokens or not toks:
            continue
        inter = len(claimed_tokens & toks)
        union = len(claimed_tokens | toks)
        ratio = inter / union if union else 0.0
        if ratio > best_ratio:
            best_ratio = ratio
            best = m

    # Heuristic thresholds; legal text often paraphrased so keep lenient-ish.
    if best_ratio >= 0.45:
        verdict = "consistent"
    elif best_ratio >= 0.2:
        verdict = "partial"
    else:
        verdict = "contradicts_or_unsupported"

    return {
        "verified": verdict == "consistent",
        "verdict": verdict,
        "overlap_ratio": round(best_ratio, 3),
        "best_match": best,
        "law_name": law_name,
        "article_number": article_number,
        "claimed_text_preview": claimed_text[:200],
        "note": (
            "So sánh token-overlap (regex \\w+ loại dấu câu + bỏ stopword Việt). "
            "'contradicts_or_unsupported' không có nghĩa điều luật sai — chỉ là claim "
            "không khớp corpus; agent nên web_search xác minh."
        ),
    }