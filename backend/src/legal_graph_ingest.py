"""Legal knowledge-graph ingest — write Statute/Article nodes from chunks.

Reuses ``extract_legal_metadata`` (regex) to pull ``law_name`` +
``article_number`` from each chunk, then MERGEs the
``(:Statute {name})-[:HAS_ARTICLE]->(:Article {number, text})`` subgraph in
Neo4j. Best-effort + idempotent (MERGE): a down/absent Neo4j is logged and
swallowed so the vector ingest path is never blocked (graph is additive).

Phase 3b (case-law ``CITES``/``OVERTURNS`` relations) is intentionally out of
scope here — it needs an LLM relation extractor and is gated on
``content_type == "case_law"`` chunks existing in the corpus.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Optional

from graph_db import execute_write

logger = logging.getLogger(__name__)

# Year suffix on a law name, e.g. "Bộ luật Dân sự 2015" -> year 2015.
_YEAR_RE = re.compile(r"(19|20)\d{2}\b")

# Truncate article text stored on the node — full chunk already lives in Qdrant.
_MAX_ARTICLE_TEXT = 2000

_MERGE_CYPHER = """
MERGE (st:Statute {name: $law})
  ON CREATE SET st.created_at = timestamp()
  SET st.year = $year
MERGE (a:Article {number: $art, statute: $law})
  ON CREATE SET a.created_at = timestamp()
  SET a.text = $text
MERGE (st)-[:HAS_ARTICLE]->(a)
"""


def _extract_year(law_name: str) -> Optional[int]:
    m = _YEAR_RE.search(law_name or "")
    if not m:
        return None
    try:
        return int(m.group(0))
    except ValueError:
        return None


def add_to_graph(chunk_id: str, text: str, meta: Dict[str, Any]) -> bool:
    """MERGE a Statute->Article subgraph from a chunk's extracted metadata.

    Args:
        chunk_id: the chunk's deterministic id (unused in MERGE key but kept
            for traceability/future ``(:Chunk)`` node linkage).
        text: the chunk text to store on the Article node (truncated).
        meta: extracted metadata dict; must contain ``law_name`` AND a
            non-null ``article_number`` (both produced by
            ``extract_legal_metadata``). Missing either -> no-op.

    Returns:
        True if a write was attempted and succeeded, False if skipped or
        the graph was unavailable (never raises).
    """
    if not meta:
        return False
    law_name = meta.get("law_name")
    article_number = meta.get("article_number")
    if not law_name or article_number is None:
        return False
    try:
        art = int(article_number)
    except (TypeError, ValueError):
        return False

    return execute_write(
        _MERGE_CYPHER,
        law=str(law_name),
        year=_extract_year(str(law_name)),
        art=art,
        text=(text or "")[:_MAX_ARTICLE_TEXT],
    )


def add_batch_to_graph(items):
    """Ingest an iterable of ``(chunk_id, text, meta)`` tuples.

    Best-effort: a single failing write is logged and skipped; the rest
    proceed. Idempotent via MERGE — safe to re-ingest the same corpus.
    """
    written = 0
    for chunk_id, text, meta in items:
        if add_to_graph(chunk_id, text, meta):
            written += 1
    return written