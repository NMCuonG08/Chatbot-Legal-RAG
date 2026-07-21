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
from legal_graph_relations import extract_relations, RELATION_TYPES

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

# Cross-reference edges between articles/statutes. Cypher cannot parameterize
# the relationship TYPE, so one template per edge type is kept here and looked
# up by relation name. Each template MERGEs both endpoints (so a relation is
# written even if the target article was never ingested as its own chunk) and
# the typed edge with a created_at stamp + ``chunk_id`` for traceability.
# Endpoints: source/target (:Article) for CITES and AMENDS; (:Statute)
# endpoints for REPEALS / REPLACED_BY (with the article numbers carried as edge
# props since the Statute node itself has no single article).
_MERGE_RELATION_CYPHERS = {
    "CITES": """
MERGE (src:Article {number: $source_art, statute: $source_law})
MERGE (dst:Article {number: $target_art, statute: $target_law})
MERGE (src)-[r:CITES]->(dst)
  ON CREATE SET r.created_at = timestamp(), r.chunk_id = $chunk_id
""",
    "AMENDS": """
MERGE (src:Article {number: $source_art, statute: $source_law})
MERGE (dst:Article {number: $target_art, statute: $target_law})
MERGE (src)-[r:AMENDS]->(dst)
  ON CREATE SET r.created_at = timestamp(), r.chunk_id = $chunk_id
""",
    "REPEALS": """
MERGE (src:Statute {name: $source_law})
MERGE (dst:Statute {name: $target_law})
MERGE (src)-[r:REPEALS]->(dst)
  ON CREATE SET r.created_at = timestamp(), r.chunk_id = $chunk_id,
                 r.source_article = $source_art, r.target_article = $target_art
""",
    "REPLACED_BY": """
MERGE (src:Statute {name: $source_law})
MERGE (dst:Statute {name: $target_law})
MERGE (src)-[r:REPLACED_BY]->(dst)
  ON CREATE SET r.created_at = timestamp(), r.chunk_id = $chunk_id,
                 r.source_article = $source_art, r.target_article = $target_art
""",
}


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

    written = execute_write(
        _MERGE_CYPHER,
        law=str(law_name),
        year=_extract_year(str(law_name)),
        art=art,
        text=(text or "")[:_MAX_ARTICLE_TEXT],
    )
    # Mine cross-reference edges from the same chunk text and MERGE them.
    # Best-effort: a relation write failure is logged and swallowed so the
    # HAS_ARTICLE write above is never rolled back (graph is additive).
    if written:
        try:
            add_relations_to_graph(chunk_id, text or "", law_name, art)
        except Exception as exc:  # noqa: BLE001 — graph is additive
            logger.warning("[GRAPH] relation extract/merge failed for %s: %s", chunk_id, exc)
    return written


def add_relations_to_graph(
    chunk_id: str,
    text: str,
    source_law: str,
    source_article: int,
) -> int:
    """Extract cross-reference relations from ``text`` and MERGE them.

    Called from ``add_to_graph`` after the Statute->Article node write so the
    endpoints already exist (or are created on demand by the edge MERGE).
    Idempotent via MERGE — safe to re-ingest the same corpus.

    Returns the number of edge writes attempted (not necessarily distinct — a
    re-ingest re-MERGES the same edges). Returns 0 when no relations were
    extracted or the graph is unavailable.
    """
    relations = extract_relations(text, source_law=source_law, source_article=source_article)
    if not relations:
        return 0
    written = 0
    for rel in relations:
        cypher = _MERGE_RELATION_CYPHERS.get(rel["relation"])
        if not cypher:
            continue
        ok = execute_write(
            cypher,
            source_law=str(rel["source_law"]),
            source_art=int(rel["source_art"]),
            target_law=str(rel["target_law"]),
            target_art=int(rel["target_art"]),
            chunk_id=str(chunk_id),
        )
        if ok:
            written += 1
    return written


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