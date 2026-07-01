"""Normalized intermediate schema for the multi-source pipeline.

Every stage past ingestion operates exclusively on these immutable
dataclasses. Each carries lineage (``doc_id`` + the version/config of the
stage that produced it) so any chunk can be traced back to its raw origin.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class RawDocument:
    """Tier-1 raw document — exactly what a connector fetched.

    ``doc_id`` is a content hash so the same document fetched twice yields the
    same id (idempotent ingestion — never reprocessed once ``embedded``).
    """

    doc_id: str
    source_id: str  # e.g. "jsonl_qa", "markdown_dir", "pdf_crawler"
    source_type: str  # "json" | "markdown" | "html" | "pdf"
    content: str
    origin_url: str
    fetched_at: datetime
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ParsedDocument:
    """Tier-2 parsed document — plain text extracted from raw content.

    Lineage: ``raw_doc_id`` points back to the :class:`RawDocument` that
    produced this. ``parser_used`` / ``parser_version`` record *how* it was
    extracted so re-runs after a parser change are detectable.
    """

    doc_id: str
    source_id: str
    source_type: str
    text: str
    parser_used: str
    parser_version: str
    raw_doc_id: str
    fetched_at: datetime
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ChunkedDocument:
    """Tier-3 serving document — chunks ready for embedding.

    Lineage: ``parsed_doc_id`` points back to the :class:`ParsedDocument`;
    ``chunk_config`` / ``embed_model`` record the chunking + embedding config
    so a chunk produced under a different config is not confused with an
    older one.
    """

    doc_id: str
    source_id: str
    chunks: tuple[str, ...]
    chunk_config: str
    embed_model: str
    parsed_doc_id: str
    meta: dict[str, Any] = field(default_factory=dict)