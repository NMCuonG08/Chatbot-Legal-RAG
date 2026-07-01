"""Three-tier immutable storage lake.

    raw/         — original bytes/content. NEVER modified. Re-run any
                  experiment (swap OCR, swap chunker) from here without
                  re-fetching.
    processed/   — ParsedDocument JSON, lineage: raw_doc_id + parser_used.
    serving/     — ChunkedDocument JSON, lineage: parsed_doc_id + chunk_config.

Raw is write-once: persisting a doc that already exists on disk is a no-op,
which is what makes re-runs cheap and the raw tier a true immutable source of
truth.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from pipeline.schema import ChunkedDocument, ParsedDocument, RawDocument

logger = logging.getLogger(__name__)

# Default lake root: <repo>/data/pipeline_lake. Override via PIPELINE_LAKE_DIR.
_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LAKE_ROOT = _REPO_ROOT / "data" / "pipeline_lake"


def _lake_root() -> Path:
    import os

    return Path(os.getenv("PIPELINE_LAKE_DIR", str(DEFAULT_LAKE_ROOT)))


def raw_dir(source_id: str) -> Path:
    return _lake_root() / "raw" / source_id


def processed_dir() -> Path:
    return _lake_root() / "processed"


def serving_dir() -> Path:
    return _lake_root() / "serving"


def _safe_name(doc_id: str) -> str:
    # doc_id already connector-controlled; strip anything path-nasty defensively.
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in doc_id)


def persist_raw(doc: RawDocument) -> Path:
    """Write raw content once. Re-writes are no-ops (immutable tier)."""
    target_dir = raw_dir(doc.source_id)
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"{_safe_name(doc.doc_id)}.txt"
    if path.exists():
        logger.debug("Raw already persisted for %s, skipping", doc.doc_id)
        return path
    path.write_text(doc.content, encoding="utf-8")
    return path


def persist_parsed(parsed: ParsedDocument) -> Path:
    """Write the parsed document JSON with lineage back to raw."""
    processed_dir().mkdir(parents=True, exist_ok=True)
    path = processed_dir() / f"{_safe_name(parsed.doc_id)}.json"
    payload = {
        "doc_id": parsed.doc_id,
        "source_id": parsed.source_id,
        "source_type": parsed.source_type,
        "text": parsed.text,
        "parser_used": parsed.parser_used,
        "parser_version": parsed.parser_version,
        "raw_doc_id": parsed.raw_doc_id,
        "fetched_at": parsed.fetched_at.isoformat(),
        "meta": parsed.meta,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def persist_serving(chunked: ChunkedDocument) -> Path:
    """Write the chunked document JSON with lineage back to parsed."""
    serving_dir().mkdir(parents=True, exist_ok=True)
    path = serving_dir() / f"{_safe_name(chunked.doc_id)}.json"
    payload = {
        "doc_id": chunked.doc_id,
        "source_id": chunked.source_id,
        "chunks": list(chunked.chunks),
        "chunk_config": chunked.chunk_config,
        "embed_model": chunked.embed_model,
        "parsed_doc_id": chunked.parsed_doc_id,
        "meta": chunked.meta,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path