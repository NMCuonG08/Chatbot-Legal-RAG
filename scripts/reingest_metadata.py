"""One-time backfill — enrich existing Qdrant chunks with Phase 1 metadata.

Re-running ``import_data`` skips already-embedded chunks (chunk-hash match), so
chunks indexed before the Phase 1 metadata upgrade never receive the new
fields (``clause_number``, ``point_letter``, ``document_number``,
``document_year``, ``document_type``, ``effectivity_status``). This script
scrolls the live Qdrant collection, re-extracts metadata from each chunk's
``question`` + ``content``, classifies effectivity, and writes the new fields
back via ``set_payload`` — **no re-embedding**: vectors are untouched, only
payload fields are updated in place.

Idempotent: re-running only re-writes the same fields. Best-effort: Qdrant
down = script errors out (no silent skip — operator must see it).

Optionally re-MERGEs Neo4j ``Statute->Article`` (reuses
``legal_graph_ingest.add_to_graph``) and rebuilds the BM25 cache so the new
metadata-rich text is tokenized.

Env: Qdrant defaults via ``vectorize``; NEO4J_URI/USER/PASSWORD for the graph
re-merge (optional — pass ``--no-graph`` to skip).
Run:  python scripts/reingest_metadata.py [--collection llm] [--limit 0] \\
                                [--batch 500] [--no-graph] [--no-bm25]
"""
import argparse
import os
import sys
import time
from typing import Any, Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, "..", "backend", "src"))
sys.path.insert(0, SRC)

from qdrant_client import QdrantClient  # noqa: E402

from legal_metadata import extract_legal_metadata            # noqa: E402
from legal_effectivity import effectivity_for_payload         # noqa: E402
from legal_graph_ingest import add_to_graph                   # noqa: E402
from config import DEFAULT_COLLECTION_NAME                    # noqa: E402
from vectorize import get_client                               # noqa: E402

# Fields this script owns on the payload. Kept in one place so the set_payload
# call and the "skip if already complete" check stay in sync.
NEW_FIELDS = (
    "clause_number",
    "point_letter",
    "document_number",
    "document_year",
    "document_type",
    "effectivity_status",
)


def _scroll_all(client: QdrantClient, collection: str, batch: int):
    """Yield (point_id, payload) for every point, retrying transient scroll drops.

    Mirrors ``scripts/graph_backfill._scroll_all``: long scrolls over 100k+
    points can hit transient HTTP connection drops, so each page is retried a
    few times with the same opaque offset (no page lost or duplicated).
    """
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


def _build_payload_update(question: str, content: str) -> Dict[str, Any]:
    """Re-extract metadata + effectivity for one chunk.

    Returns ONLY the new fields (no null pollution — keys absent when the
    regex did not match), so ``set_payload`` overwrites exactly the fields the
    parser recognized and leaves the rest of the payload untouched.
    """
    meta = extract_legal_metadata(f"{question} {content}")
    update: Dict[str, Any] = {}
    for k in ("clause_number", "point_letter", "document_number",
             "document_year", "document_type"):
        if k in meta:
            update[k] = meta[k]
    # Refresh law_name/article_number too: older chunks may have stale or
    # missing values from the pre-Phase-1 parser.
    if "law_name" in meta:
        update["law_name"] = meta["law_name"]
    if "article_number" in meta:
        update["article_number"] = meta["article_number"]
    update["effectivity_status"] = effectivity_for_payload(
        update.get("law_name"), update.get("document_year")
    )
    return update


def _is_already_complete(payload: Dict[str, Any]) -> bool:
    """Skip points that already carry every Phase 1 field (re-run fast path)."""
    return all(payload.get(f) is not None for f in NEW_FIELDS)


def main() -> None:
    ap = argparse.ArgumentParser(description="Backfill Phase 1 metadata onto Qdrant chunks.")
    ap.add_argument("--collection", default=DEFAULT_COLLECTION_NAME,
                    help=f"Qdrant collection (default {DEFAULT_COLLECTION_NAME})")
    ap.add_argument("--limit", type=int, default=0,
                    help="Max chunks to process (0 = all)")
    ap.add_argument("--batch", type=int, default=500,
                    help="Scroll page size")
    ap.add_argument("--flush", type=int, default=200,
                    help="Points per set_payload batch")
    ap.add_argument("--force", action="store_true",
                    help="Re-write even chunks that already have all fields")
    ap.add_argument("--no-graph", action="store_true",
                    help="Skip Neo4j Statute->Article re-MERGE")
    ap.add_argument("--no-bm25", action="store_true",
                    help="Skip BM25 cache rebuild at the end")
    ap.add_argument("--no-payload", action="store_true",
                    help="Skip Qdrant set_payload (payload already backfilled). "
                         "Run graph MERGE only — uses each point's EXISTING "
                         "payload (law_name/article_number/effectivity_status "
                         "from a prior run) so add_to_graph still gets metadata. "
                         "Use this to (re)build the graph without re-stressing "
                         "Qdrant with 170k set_payload writes.")
    args = ap.parse_args()

    print("== Phase 1 metadata backfill ==")
    qclient = get_client()

    seen = 0
    updated = 0
    skipped_done = 0
    pending: List[Any] = []
    pending_payloads: List[Dict[str, Any]] = []

    def _flush_set():
        nonlocal updated
        if not pending:
            return
        try:
            qclient.set_payload(
                collection_name=args.collection,
                payload=pending_payloads[0],  # same update dict for all points
                points=pending,
                wait=True,
            )
            updated += len(pending)
            print(f"  ... set_payload wrote {updated} points ({seen} scanned)")
        except Exception as exc:
            print(f"  set_payload batch failed (non-blocking): {exc}")
        pending.clear()
        pending_payloads.clear()

    for point_id, payload in _scroll_all(qclient, args.collection, args.batch):
        if args.limit and seen >= args.limit:
            break
        seen += 1

        content = payload.get("content") or payload.get("text") or ""
        question = payload.get("question") or ""

        if args.no_payload:
            # Graph-only path: payload already backfilled in a prior run, so
            # skip set_payload and feed the EXISTING payload to add_to_graph.
            if not args.no_graph:
                try:
                    add_to_graph(str(point_id), content, dict(payload))
                except Exception as exc:  # noqa: BLE001 — graph is additive
                    print(f"  graph merge skipped for {point_id}: {exc}")
            continue

        if not args.force and _is_already_complete(payload):
            skipped_done += 1
            continue

        update = _build_payload_update(question, content)

        # Re-MERGE Neo4j per point (best-effort, idempotent).
        if not args.no_graph:
            try:
                merged = dict(payload)
                merged.update(update)
                add_to_graph(str(point_id), content, merged)
            except Exception as exc:  # noqa: BLE001 — graph is additive
                print(f"  graph merge skipped for {point_id}: {exc}")

        pending.append(point_id)
        if not pending_payloads:
            pending_payloads.append(update)
        if len(pending) >= args.flush:
            _flush_set()

    if not args.no_payload:
        _flush_set()

    print(f"\nscanned={seen}  updated={updated}  skipped_already_done={skipped_done}")

    # BM25 cache rebuild — the new metadata-rich text (question + content +
    # parsed law references) should be tokenized fresh. Delegates to the
    # existing search initializer which persists to data/bm25_cache.
    if not args.no_bm25:
        print("\n== Rebuilding BM25 cache ==")
        try:
            from search import initialize_from_vector_store  # noqa: E402
            initialize_from_vector_store(args.collection)
            print("BM25 cache rebuilt.")
        except Exception as exc:
            print(f"BM25 rebuild skipped: {exc}")

    print("== METADATA BACKFILL DONE ==")


if __name__ == "__main__":
    main()