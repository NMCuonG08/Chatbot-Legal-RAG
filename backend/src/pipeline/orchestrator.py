"""Pipeline orchestrator — the single core loop.

One loop, no per-source clones. Adding a source = append a connector to the
``connectors`` list passed in. Each document is processed independently: one
failure marks that doc ``failed`` and the loop continues — a bad PDF never
halts the JSONL batch.

Idempotency: before processing, ``state.is_done(doc_id, "embedded")`` is
checked; an already-embedded doc is skipped. Re-runs are cheap and safe.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Iterable

from config import DEFAULT_COLLECTION_NAME
from pipeline import state, storage
from pipeline.chunker import chunk
from pipeline.connectors.base import BaseConnector
from pipeline.embedder import embed_chunks
from pipeline.parsers import parse
from pipeline.schema import RawDocument

logger = logging.getLogger(__name__)


def run_pipeline(
    connectors: Iterable[BaseConnector],
    collection_name: str = DEFAULT_COLLECTION_NAME,
    use_semantic: bool = True,
    limit: int | None = None,
) -> dict:
    """Run fetch → parse → chunk → embed for every doc from every connector.

    Args:
        connectors: iterable of :class:`BaseConnector`. Add a source = add a
            connector here.
        collection_name: Qdrant collection to write chunks into.
        use_semantic: semantic (True) vs token (False) chunking.
        limit: hard cap on docs processed across all connectors (testing).

    Returns:
        stats dict: ``{fetched, parsed, chunked, embedded, skipped, failed}``.
    """
    state.ensure_pipeline_schema()

    stats = {"fetched": 0, "parsed": 0, "chunked": 0, "embedded": 0, "skipped": 0, "failed": 0}

    for connector in connectors:
        logger.info("== Connector %s (%s) ==", connector.source_id, connector.source_type)
        try:
            raw_docs: list[RawDocument] = connector.fetch()
        except Exception as exc:  # noqa: BLE001 — a broken source must not kill the run
            logger.error("Connector %s fetch failed: %s", connector.source_id, exc)
            continue

        for doc in raw_docs:
            if limit is not None and sum(stats.values()) >= limit:
                logger.info("Reached limit %d, stopping", limit)
                return stats

            if state.is_done(doc.doc_id, state.STATUS_EMBEDDED):
                stats["skipped"] += 1
                logger.info("Skip %s (already embedded)", doc.doc_id)
                continue

            try:
                storage.persist_raw(doc)
                state.mark_status(doc.doc_id, doc.source_id, state.STATUS_FETCHED)
                stats["fetched"] += 1

                parsed = parse(doc)
                storage.persist_parsed(parsed)
                state.mark_status(doc.doc_id, doc.source_id, state.STATUS_PARSED)
                stats["parsed"] += 1

                chunked = chunk(parsed, use_semantic=use_semantic)
                chunked = replace(
                    chunked, meta={**chunked.meta, "question": parsed.meta.get("question", "")}
                )
                storage.persist_serving(chunked)
                state.mark_status(doc.doc_id, doc.source_id, state.STATUS_CHUNKED)
                stats["chunked"] += 1

                embedded = embed_chunks(chunked, collection_name)
                # Re-persist serving with embed_model filled in (final lineage).
                storage.persist_serving(embedded)
                state.mark_status(doc.doc_id, doc.source_id, state.STATUS_EMBEDDED)
                stats["embedded"] += 1
            except Exception as exc:  # noqa: BLE001 — isolate per-doc failures
                logger.error("Doc %s failed: %s", doc.doc_id, exc)
                state.mark_status(doc.doc_id, doc.source_id, state.STATUS_FAILED, error=str(exc))
                stats["failed"] += 1

    logger.info("Pipeline done: %s", stats)
    return stats