"""Chunker — thin wrapper over the existing LlamaIndex splitter.

Single place to change chunking strategy for *every* source. The orchestrator
calls :func:`chunk`; nothing else knows about LlamaIndex nodes.
"""

from __future__ import annotations

import logging

from pipeline.schema import ChunkedDocument, ParsedDocument

logger = logging.getLogger(__name__)

CHUNK_CONFIG_VERSION = "semantic_v1"


def chunk(parsed: ParsedDocument, use_semantic: bool = True) -> ChunkedDocument:
    """Split a :class:`ParsedDocument` into chunks.

    Delegates to :func:`splitter.split_document` so the whole pipeline shares
    one chunking implementation. Returns an immutable :class:`ChunkedDocument`
    (tuple of chunks) carrying lineage back to ``parsed``.
    """
    # Local import avoids loading LlamaIndex at module import time (tests).
    from splitter import split_document

    nodes = split_document(parsed.text, use_semantic=use_semantic)
    chunks = tuple(node.text for node in nodes)
    logger.info("Chunked doc %s into %d chunks", parsed.doc_id, len(chunks))
    return ChunkedDocument(
        doc_id=parsed.doc_id,
        source_id=parsed.source_id,
        chunks=chunks,
        chunk_config=CHUNK_CONFIG_VERSION + ("+semantic" if use_semantic else "+token"),
        embed_model="",  # filled by embedder at embed time
        parsed_doc_id=parsed.doc_id,
        meta={
            "parser_used": parsed.parser_used,
            "origin_url": parsed.meta.get("origin_url", ""),
        },
    )