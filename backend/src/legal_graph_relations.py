"""Rule + regex cross-reference extractor for the legal knowledge graph.

Companion to ``legal_graph_ingest``. The base ingest only writes
``(:Statute)-[:HAS_ARTICLE]->(:Article)``; this module mines the chunk text for
cross-reference edges between articles/statutes and lets the graph capture the
citation, amendment, repeal, and replacement relationships that pure
``HAS_ARTICLE`` cannot represent.

Edge types produced (all between ``:Article`` nodes unless noted):

- ``CITES``        — source article references target article
                    (verbs: dẫn chiếu, tham chiếu, theo, tại, quy định tại,
                    được quy định tại, hướng dẫn).
- ``AMENDS``       — source article amends target (sửa đổi, bổ sung).
- ``REPEALS``       — source article repeals target (bãi bỏ).
- ``REPLACED_BY``  — source article is replaced by target (thay thế bằng).

Case-law ``CITES``/``OVERTURNS`` relations are deliberately out of scope here
(they need an LLM extractor and ``content_type='case_law'`` chunks — see the
note in ``legal_graph_ingest``).

The extractor is conservative: it only emits a relation when it can resolve a
target article number (and ideally a target statute name) from the same chunk.
Unresolved targets are dropped rather than written as dangling edges.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Public edge-type enum — matched 1:1 to Neo4j relationship types.
RELATION_CITES = "CITES"
RELATION_AMENDS = "AMENDS"
RELATION_REPEALS = "REPEALS"
RELATION_REPLACED_BY = "REPLACED_BY"
RELATION_TYPES = (RELATION_CITES, RELATION_AMENDS, RELATION_REPEALS, RELATION_REPLACED_BY)

# Article reference: "Điều 35", "điều 35", optionally followed by a statute name
# fragment. Captures the article number; the statute name (if any) is parsed
# separately by ``_parse_target_law`` over the same clause.
_ARTICLE_REF_RE = re.compile(r"Điều\s+(\d{1,4})", re.IGNORECASE | re.UNICODE)

# A statute name fragment following a target article reference, e.g.
# "Điều 35 Bộ luật Dân sự" or "Điều 10 Luật Đất đai 2024". Non-greedy up to the
# next clause boundary.
_TARGET_LAW_RE = re.compile(
    r"Điều\s+\d{1,4}\s+((?:Bộ\s+luật|Luật|Nghị\s+định|Thông\s+tư|Nghị\s+quyết)"
    r"[\wÀ-ỹ\s]+?\d{4})",
    re.UNICODE,
)

# Each verb phrase maps to an edge type. Order matters: scan longer/stronger
# verbs first so "bãi bỏ" wins over a generic "theo" in the same clause.
_VERB_PATTERNS: List[tuple] = [
    (re.compile(r"thay\s+thế\s+(?:bằng|cho)", re.IGNORECASE | re.UNICODE), RELATION_REPLACED_BY),
    (re.compile(r"bãi\s+bỏ", re.IGNORECASE | re.UNICODE), RELATION_REPEALS),
    (re.compile(r"sửa\s+đổi|bổ\s+sung", re.IGNORECASE | re.UNICODE), RELATION_AMENDS),
    # Generic citation verbs — lowest precedence.
    (re.compile(
        r"dẫn\s+chiếu|tham\s+chiếu|được\s+quy\s+định\s+tại|quy\s+định\s+tại|"
        r"hướng\s+dẫn|theo|tại",
        re.IGNORECASE | re.UNICODE,
    ), RELATION_CITES),
]


def _parse_target_law(clause: str) -> Optional[str]:
    """Extract a target statute name from a clause like 'Điều 35 Bộ luật Dân sự'."""
    m = _TARGET_LAW_RE.search(clause)
    if not m:
        return None
    return m.group(1).strip()


def _clause_around(text: str, verb_match: re.Match) -> str:
    """Return a window of text around a verb match for target resolution.

    Looks forward up to ~160 chars from the verb — Vietnamese legal clauses
    place the target ('Điều X Luật Y') after the referencing verb.
    """
    start = verb_match.start()
    return text[start:start + 160]


def extract_relations(
    text: str,
    source_law: Optional[str] = None,
    source_article: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Extract cross-reference relations from ``text``.

    Args:
        text:           chunk text to mine.
        source_law:     the citing statute name (from metadata). When absent the
                         relation is dropped (no source endpoint).
        source_article: the citing article number. When absent the relation is
                         dropped.

    Returns:
        List of relation dicts:
        ``{source_law, source_art, target_law, target_art, relation}``.
        ``target_law`` may fall back to ``source_law`` when the clause cites only
        an article number without a statute name (the common "khoản 2 Điều 35
        [of this law]" case). De-duplicated by ``(target_art, target_law, relation)``.
    """
    if not text or not source_law or source_article is None:
        return []

    seen: set = set()
    out: List[Dict[str, Any]] = []
    for verb_re, relation in _VERB_PATTERNS:
        for vm in verb_re.finditer(text):
            clause = _clause_around(text, vm)
            # Find the first article reference in the forward window.
            art_m = _ARTICLE_REF_RE.search(clause)
            if not art_m:
                continue
            try:
                target_art = int(art_m.group(1))
            except ValueError:
                continue
            target_law = _parse_target_law(clause)
            # If no target statute named, assume same statute as the source.
            if not target_law:
                target_law = source_law

            # Drop self-references ("Theo Điều 35 ...", where the clause cites
            # the source article itself) — not a real graph edge.
            if target_art == int(source_article) and str(target_law) == str(source_law):
                continue

            key = (target_art, target_law, relation)
            if key in seen:
                continue
            seen.add(key)

            out.append({
                "source_law": str(source_law),
                "source_art": int(source_article),
                "target_law": str(target_law),
                "target_art": target_art,
                "relation": relation,
            })

    return out