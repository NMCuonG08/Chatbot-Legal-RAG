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
# Citation/article boundary: a statute name never contains a citation verb or
# article reference, so cut the captured name at the first one. Fixes queries
# like "Nghị định 126/2014 dẫn chiếu điều nào?" -> "Nghị định 126/2014"
# (without this, _LAW_RE's greedy `[^,.;?!]*` captures "Nghị định 126/2014 dẫn
# chiếu điều nào" and the exact-statute CITES Cypher match fails).
_LAW_BOUNDARY_RE = re.compile(
    r"\s+(?:dẫn\s+chiếu|tham\s+chiếu|dẫn\s+chứng|bác\s+bỏ|còn\s+hiệu\s+lực|"
    r"chuỗi\s+dẫn\s+chiếu|điều|khoản|điểm|theo|tại|về|của).*$",
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

# Phase 4 cross-reference traversal. Outbound CITES/AMENDS from a source
# article (the article the user asked about) -> the articles it references or
# amends. ``relation`` labels the edge type so the caller can render "dẫn chiếu
# tới" / "sửa đổi" distinctly.
_LOOKUP_CITES_CYPHER = """
MATCH (src:Article {number: $art, statute: $law})-[r:CITES|AMENDS]->(dst:Article)
RETURN dst.number AS number, dst.text AS text, dst.statute AS law,
       type(r) AS relation
LIMIT $limit
"""

# Inbound cross-refs — "điều nào dẫn chiếu tới Điều X" / "bị sửa đổi bởi điều nào".
_LOOKUP_CITED_BY_CYPHER = """
MATCH (src:Article)-[r:CITES|AMENDS]->(dst:Article {number: $art, statute: $law})
RETURN src.number AS number, src.text AS text, src.statute AS law,
       type(r) AS relation
LIMIT $limit
"""


def _parse_law(query: str) -> str:
    m = _LAW_RE.search(query or "")
    if not m:
        return ""
    name = m.group(1).strip()
    # Cut at the first citation/article boundary (e.g. " dẫn chiếu ...") so the
    # statute name does not absorb query intent words. Then strip trailing
    # 4-digit year + filler/question words, leaving the proper noun (e.g.
    # "Bộ luật Dân sự").
    name = _LAW_BOUNDARY_RE.sub("", name)
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


def recall_legal_graph_relations(
    query: str,
    limit: int = 5,
    direction: str = "out",
) -> Dict[str, Any]:
    """Traverse Phase 4 cross-reference edges (CITES / AMENDS) for a query.

    Parses the source statute + article from the query (same regex as
    ``recall_legal_graph``) and returns the articles connected by a
    cross-reference edge — outbound (the article cites/amends others) or
    inbound (others cite/amend it).

    Args:
        query:     natural-language query naming an article, e.g.
                   "Điều 35 Bộ luật Dân sự dẫn chiếu điều nào?".
        limit:     max related articles to return.
        direction: ``"out"`` (outbound CITES/AMENDS) or ``"in"`` (inbound).

    Returns:
        ``{"query", "law", "article", "relations": List[dict]}`` where each
        relation dict carries ``number``, ``text``, ``law``, ``relation``.
        Empty when the graph is unavailable or no source article parsed.
    """
    law = _parse_law(query)
    article = _parse_article(query)
    if not law or not article:
        return {"query": query, "law": law, "article": article, "relations": []}

    cypher = _LOOKUP_CITED_BY_CYPHER if direction == "in" else _LOOKUP_CITES_CYPHER
    rows = execute_read(cypher, law=law, art=article, limit=limit)
    return {
        "query": query,
        "law": law,
        "article": article,
        "relations": rows,
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