"""Agent tool — multi-hop recall over the Neo4j legal knowledge graph.

``recall_legal_graph_tool`` lets the ReAct agent traverse the
``(:Statute)-[:HAS_ARTICLE]->(:Article)`` graph for queries that vector search
handles poorly: citation chains ("điều nào dẫn chiếu", "còn hiệu lực không"),
article enumeration within a statute, and multi-hop lookups.

Best-effort: when Neo4j is down/absent, returns an empty result so the agent
falls back to vector retrieval (graph is additive, never blocks).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict

from graph_db import execute_read

logger = logging.getLogger(__name__)

# Parse "Điều 35" / "điều 35" out of a free-form query.
_ARTICLE_RE = re.compile(r"điều\s+(\d{1,4})", re.IGNORECASE)
# Parse a statute name loosely — greedy capture of the statute type plus its
# proper-noun phrase, stopping at sentence punctuation. Trailing year and
# filler/question words are stripped in _parse_law.
_LAW_RE = re.compile(
    r"((?:bộ\s+luật|luật|nghị\s+định|pháp\s+lệnh|thông\s+tư)\s+[^,.;?!]*)",
    re.IGNORECASE,
)

# Trailing filler/question words to strip from a parsed statute name so
# "Bộ luật Dân sự nói gì?" -> "Bộ luật Dân sự".
_FILLER_TAIL_RE = re.compile(
    r"\s+(?:nói\s+gì|thế\s+nào|bao\s+giờ|bao\s+lâu|như\s+thế\s+nào|"
    r"nói|quy\s+định|được|có|là|không|đã|về|của|nào|gì|khi|ở|tại|"
    r"thì|và|hoặc|hay|phải|nên|cần|muốn|bị|đang|sẽ|đã|chưa).*$",
    re.IGNORECASE,
)
_YEAR_TAIL_RE = re.compile(r"\s+\d{4}$")

# Multi-hop terms that signal the graph tool is a better fit than vector search.
MULTI_HOP_KEYWORDS = ["dẫn chiếu", "dẫn chứng", "bác bỏ", "còn hiệu lực",
                      "điều nào", "tham chiếu", "chuỗi dẫn chiếu"]

_LOOKUP_BY_LAW_CYPHER = """
MATCH (st:Statute)-[:HAS_ARTICLE]->(a:Article)
WHERE toLower(st.name) CONTAINS toLower($law)
RETURN a.number AS number, a.text AS text, st.name AS law
ORDER BY a.number
LIMIT $limit
"""

_LOOKUP_BY_LAW_AND_ARTICLE_CYPHER = """
MATCH (st:Statute)-[:HAS_ARTICLE]->(a:Article {number: $art})
WHERE toLower(st.name) CONTAINS toLower($law)
RETURN a.number AS number, a.text AS text, st.name AS law
LIMIT $limit
"""


def _parse_law(query: str) -> str:
    m = _LAW_RE.search(query or "")
    if not m:
        return ""
    name = m.group(1).strip()
    # Strip trailing 4-digit year, then trailing filler/question words, so the
    # parsed name is just the statute proper noun (e.g. "Bộ luật Dân sự").
    name = _YEAR_TAIL_RE.sub("", name)
    name = _FILLER_TAIL_RE.sub("", name).strip()
    return name


def _parse_article(query: str) -> int:
    m = _ARTICLE_RE.search(query or "")
    return int(m.group(1)) if m else 0


def recall_legal_graph(query: str, limit: int = 5) -> Dict[str, Any]:
    """Traverse the legal graph for a multi-hop query.

    Args:
        query: natural-language query, e.g. "Điều 35 Bộ luật Dân sự nói gì,
            bị án nào dẫn chiếu chưa?".
        limit: max articles to return.

    Returns:
        ``{"query": str, "law": str, "article": int, "articles": List[dict]}``.
        Each article dict has ``number``, ``text``, ``law``. Empty list when
        the graph is unavailable or no match found (never raises).
    """
    law = _parse_law(query)
    article = _parse_article(query)
    if not law:
        return {"query": query, "law": "", "article": article, "articles": []}

    if article and article > 0:
        rows = execute_read(
            _LOOKUP_BY_LAW_AND_ARTICLE_CYPHER,
            law=law, art=article, limit=limit,
        )
    else:
        rows = execute_read(_LOOKUP_BY_LAW_CYPHER, law=law, limit=limit)

    return {
        "query": query,
        "law": law,
        "article": article,
        "articles": rows,
    }


def recall_legal_graph_tool(query: str, limit: int = 5) -> str:
    """Agent-facing wrapper around ``recall_legal_graph`` (JSON-serialized).

    Dùng khi câu hỏi cần duyệt đồ thị luật: "điều nào dẫn chiếu", "còn hiệu lực
    không", "bác bỏ bởi luật nào", hoặc liệt kê các điều của một văn bản luật.
    Ưu tiên cho truy vấn multi-hop mà vector search xử lý kém.

    Args:
        query: Câu hỏi tự nhiên, ví dụ "Điều 35 Bộ luật Dân sự nói gì, bị án
            nào dẫn chiếu chưa?".
        limit: Số điều luật trả về tối đa.
    """
    try:
        result = recall_legal_graph(query, limit=limit)
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as exc:
        logger.error("recall_legal_graph_tool error: %s", exc)
        return json.dumps({"error": str(exc), "articles": []}, ensure_ascii=False)