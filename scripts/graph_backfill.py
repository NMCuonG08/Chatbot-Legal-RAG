"""One-time backfill — populate Neo4j graph from existing Qdrant chunks.

Re-running ``import_data`` skips already-embedded chunks (hash match), so the
Phase 3 graph-ingest hook never fires for the existing corpus. This script
scrolls the live Qdrant collection and MERGEs Statute->Article nodes for every
chunk whose metadata yields a ``law_name`` + ``article_number``.

Idempotent (MERGE) — safe to re-run. Best-effort — Neo4j down = no-op.

Env: NEO4J_URI/USER/PASSWORD (+ Qdrant defaults via vectorize).
Run:  python scripts/graph_backfill.py [--collection llm] [--limit 0] [--batch 500]
"""
import argparse
import os
import sys
from typing import Any, Dict

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, "..", "backend", "src"))
sys.path.insert(0, SRC)

from qdrant_client import QdrantClient  # noqa: E402

from legal_metadata import extract_legal_metadata            # noqa: E402
from legal_graph_ingest import _extract_year, _MAX_ARTICLE_TEXT  # noqa: E402
from graph_db import get_graph_client, execute_read           # noqa: E402
from config import DEFAULT_COLLECTION_NAME                    # noqa: E402
from vectorize import get_client                                # noqa: E402

# Batched MERGE — one UNWIND transaction per flush instead of one session per
# chunk. Drastically cuts Bolt auth handshakes (the prior per-chunk session
# spam tripped Neo4j's auth rate-limiter on large corpora).
_FLUSH_CYPHER = """
UNWIND $rows AS row
MERGE (st:Statute {name: row.law})
  ON CREATE SET st.created_at = timestamp()
  SET st.year = row.year
MERGE (a:Article {number: row.art, statute: row.law})
  ON CREATE SET a.created_at = timestamp()
  SET a.text = row.text
MERGE (st)-[:HAS_ARTICLE]->(a)
"""


def _flush(rows) -> int:
    """MERGE a batch of {law, year, art, text} rows in ONE driver session.

    Returns the number of rows written (0 if the graph is down or the write
    errored — never raises, mirroring the additive best-effort contract).
    """
    driver = get_graph_client()
    if driver is None or not rows:
        return 0
    try:
        with driver.session() as session:
            session.run(_FLUSH_CYPHER, rows=rows)
        return len(rows)
    except Exception as exc:
        print(f"  flush failed (non-blocking): {exc}")
        return 0


def _scroll_all(client: QdrantClient, collection: str, batch: int):
    """Yield (point_id, payload) for every point in the collection.

    Retries each scroll page a few times — long scrolls over 100k+ points can
    hit transient HTTP connection drops (WinError 10054); the opaque scroll
    offset is reused so no page is lost or duplicated on retry.
    """
    import time

    offset = None
    while True:
        points, next_offset = None, None
        last_exc = None
        for attempt in range(5):
            try:
                points, next_offset = client.scroll(
                    collection_name=collection,
                    limit=batch,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
                break
            except Exception as exc:  # noqa: BLE001 — transient, retry same page
                last_exc = exc
                time.sleep(2 * (attempt + 1))
        if points is None:
            raise RuntimeError(f"scroll failed after retries at offset={offset}: {last_exc}")
        for p in points:
            yield p.id, (p.payload or {})
        if next_offset is None:
            break
        offset = next_offset


def main() -> None:
    ap = argparse.ArgumentParser(description="Backfill Neo4j graph from Qdrant chunks.")
    ap.add_argument("--collection", default=DEFAULT_COLLECTION_NAME,
                    help=f"Qdrant collection (default {DEFAULT_COLLECTION_NAME})")
    ap.add_argument("--limit", type=int, default=0,
                    help="Max chunks to process (0 = all)")
    ap.add_argument("--batch", type=int, default=500,
                    help="Scroll page size")
    ap.add_argument("--flush", type=int, default=500,
                    help="Rows per Neo4j UNWIND transaction (cuts auth handshakes)")
    args = ap.parse_args()

    print("== Neo4j graph backfill ==")
    print("driver:", get_graph_client())
    qclient = get_client()

    seen = 0
    written = 0
    skipped_no_meta = 0
    rows: list = []
    for point_id, payload in _scroll_all(qclient, args.collection, args.batch):
        if args.limit and seen >= args.limit:
            break
        seen += 1

        # Prefer payload-stored law_name/article_number (enriched at ingest);
        # re-extract defensively for older chunks that predate the enrichment.
        content = payload.get("content") or payload.get("text") or ""
        question = payload.get("question") or ""
        meta: Dict[str, Any] = {}
        if payload.get("law_name"):
            meta["law_name"] = payload["law_name"]
        if payload.get("article_number") is not None:
            meta["article_number"] = payload["article_number"]
        if "law_name" not in meta or "article_number" not in meta:
            extracted = extract_legal_metadata(f"{question} {content}")
            meta.update(extracted)

        if not meta.get("law_name") or meta.get("article_number") is None:
            skipped_no_meta += 1
            continue

        rows.append({
            "law": str(meta["law_name"]),
            "year": _extract_year(str(meta["law_name"])),
            "art": int(meta["article_number"]),
            "text": (content or "")[:_MAX_ARTICLE_TEXT],
        })

        if len(rows) >= args.flush:
            written += _flush(rows)
            print(f"  ... {written} rows MERGEd ({seen} scanned, {len(rows)} in txn)")
            rows = []

    if rows:
        written += _flush(rows)

    print(f"\nscanned={seen}  written={written}  skipped_no_meta={skipped_no_meta}")

    # Verify.
    n_statutes = execute_read("MATCH (s:Statute) RETURN count(s) AS c")
    n_articles = execute_read("MATCH (a:Article) RETURN count(a) AS c")
    n_edges = execute_read("MATCH ()-[r:HAS_ARTICLE]->() RETURN count(r) AS c")
    ns = n_statutes[0]["c"] if n_statutes else 0
    na = n_articles[0]["c"] if n_articles else 0
    ne = n_edges[0]["c"] if n_edges else 0
    print(f"graph now: Statutes={ns}  Articles={na}  HAS_ARTICLE edges={ne}")
    print("== BACKFILL DONE ==")


if __name__ == "__main__":
    main()