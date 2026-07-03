"""Neo4j driver singleton + test seam for the legal knowledge graph.

Mirrors the ``vectorize.py`` Qdrant-client pattern: a module-level driver
singleton, a ``set_graph_client`` test seam, and best-effort ``execute_*``
helpers that swallow connection errors so a down/absent Neo4j never breaks
the vector ingest path (graph is additive, not blocking).

Schema (Phase 3 core — see plan):
    (:Statute {name, year})-[:HAS_ARTICLE]->(:Article {number, text})
Optional Phase 3b (case-law relations) is gated elsewhere.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_driver: Optional[Any] = None
_initialized = False


def _init_driver() -> None:
    """Lazily create the Neo4j driver from env. No-op if NEO4J_URI unset."""
    global _driver, _initialized
    if _initialized:
        return
    _initialized = True
    uri = os.getenv("NEO4J_URI", "").strip()
    if not uri:
        logger.info("NEO4J_URI unset — graph memory disabled (additive).")
        return
    try:
        from neo4j import GraphDatabase

        _driver = GraphDatabase.driver(
            uri,
            auth=(
                os.getenv("NEO4J_USER", "neo4j"),
                os.getenv("NEO4J_PASSWORD", "neo4j"),
            ),
        )
        logger.info("Neo4j driver initialized for %s", uri)
    except Exception as exc:  # missing dep / unreachable — never fatal
        logger.warning("Neo4j driver init failed, graph disabled: %s", exc)
        _driver = None


def get_graph_client() -> Optional[Any]:
    """Return the lazy Neo4j driver (or None if disabled/unavailable)."""
    if not _initialized:
        _init_driver()
    return _driver


def set_graph_client(driver: Optional[Any]) -> None:
    """Test seam: inject a fake/real driver (None disables graph writes)."""
    global _driver, _initialized
    _driver = driver
    _initialized = True


def execute_write(cypher: str, **params: Any) -> bool:
    """Run a write query. Best-effort: swallows errors when graph is down.

    Returns:
        True if the query ran, False if the graph is disabled or errored.
    """
    driver = get_graph_client()
    if driver is None:
        return False
    try:
        with driver.session() as session:
            session.run(cypher, **params)
        return True
    except Exception as exc:
        logger.warning("graph write failed (additive, non-blocking): %s", exc)
        return False


def execute_read(cypher: str, **params: Any) -> List[Dict[str, Any]]:
    """Run a read query, returning records as dicts. Best-effort.

    Returns:
        List of record dicts (empty if graph disabled/errored).
    """
    driver = get_graph_client()
    if driver is None:
        return []
    try:
        with driver.session() as session:
            result = session.run(cypher, **params)
            return [dict(r) for r in result]
    except Exception as exc:
        logger.warning("graph read failed (additive, non-blocking): %s", exc)
        return []