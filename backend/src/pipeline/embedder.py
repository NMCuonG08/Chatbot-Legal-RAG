"""Embedder — embed chunks and upsert into Qdrant + MySQL chunk metadata.

Single place to swap the embedding model for *every* source. Reuses the
existing ``custom_embedding`` / ``vectorize`` / ``models`` building blocks so
the new pipeline writes into the same Qdrant collection the RAG engine reads
from — no second serving store.

Lineage: each chunk's ``document_chunks`` row stores ``doc_id`` (= the
pipeline ``doc_id``) so a re-run can detect unchanged chunks via
``chunk_hash`` and skip them (incremental, idempotent).
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from dataclasses import replace

from pipeline.schema import ChunkedDocument

logger = logging.getLogger(__name__)

EMBED_MODEL_TAG = "custom_embedding_v1"


def embed_chunks(chunked: ChunkedDocument, collection_name: str) -> ChunkedDocument:
    """Embed every chunk, upsert into Qdrant, record chunk hashes in MySQL.

    Returns a new :class:`ChunkedDocument` with ``embed_model`` filled in
    (immutable update via ``dataclasses.replace``).
    """
    if not chunked.chunks:
        logger.warning("No chunks to embed for %s, skipping", chunked.doc_id)
        return replace(chunked, embed_model=EMBED_MODEL_TAG)

    # Local imports keep LlamaIndex / Qdrant / DB out of module import time.
    from custom_embedding import get_custom_embedding
    from models import get_doc_chunks, save_doc_chunk, delete_doc_chunks_by_ids
    from vectorize import add_vector, delete_vectors_by_ids

    doc_id_str = chunked.doc_id
    old_chunks = {c.chunk_id: c.chunk_hash for c in get_doc_chunks(doc_id_str)}

    new_chunk_ids: set[str] = set()
    to_upsert: list[tuple[str, str, str]] = []  # (chunk_id, text, hash)
    to_delete: list[str] = []

    for idx, text in enumerate(chunked.chunks):
        cid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"doc_{doc_id_str}_chunk_{idx}"))
        new_chunk_ids.add(cid)
        chash = hashlib.md5(text.encode("utf-8")).hexdigest()
        if cid not in old_chunks or old_chunks[cid] != chash:
            to_upsert.append((cid, text, chash))

    for old_cid in old_chunks:
        if old_cid not in new_chunk_ids:
            to_delete.append(old_cid)

    # 1. Delete orphaned chunks (Qdrant + MySQL).
    if to_delete:
        delete_vectors_by_ids(collection_name, to_delete)
        delete_doc_chunks_by_ids(to_delete)

    # 2. Embed + upsert new/changed chunks.
    if to_upsert:
        texts = [item[1] for item in to_upsert]
        embeddings = get_custom_embedding(texts)
        if not isinstance(embeddings, list) or (
            embeddings and not isinstance(embeddings[0], list)
        ):
            embeddings = [embeddings]

        vectors_payload: dict[str, dict] = {}
        for (cid, text_val, chash), vector in zip(to_upsert, embeddings):
            vectors_payload[cid] = {
                "vector": vector,
                "payload": {
                    "question": chunked.meta.get("question", ""),
                    "content": text_val,
                    "source": chunked.source_id,
                    "doc_id": doc_id_str,
                },
            }
            save_doc_chunk(doc_id_str, cid, chash)

        add_vector(collection_name=collection_name, vectors=vectors_payload)
        logger.info(
            "Embedded %d chunks for doc %s (deleted %d orphans)",
            len(to_upsert),
            doc_id_str,
            len(to_delete),
        )
    else:
        logger.info("No chunk changes for doc %s, embed skipped", doc_id_str)

    return replace(chunked, embed_model=EMBED_MODEL_TAG)