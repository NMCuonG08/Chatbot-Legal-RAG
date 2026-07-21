"""Document reranker — Cohere API or local BGE CrossEncoder, switchable via env.

Entry point: ``rerank_documents(docs, query, top_n, rank_model)``.

Backend selected by ``RERANKER_TYPE`` env (``cohere`` default | ``bge`` |
anything else = disabled). ``RERANK_TOP_N`` env overrides the default top_n
when the caller does not pass one explicitly.

Hardening contract (Phase 3 upgrade):
- A missing Cohere key, a failed API call, or a disabled/unknown reranker type
  no longer silently returns the input docs as if they were reranked. The
  fallback still returns the docs (the pipeline must not crash) but:
    * logs a ``WARNING`` so the operator sees the degradation, and
    * stamps ``rerank_failed=True`` + ``relevance_score=None`` on every
      returned doc so downstream nodes (``grade_documents_node``) can tell a
      real rerank score from a passthrough and choose to skip LLM-judge cost.
- Uses ``logging`` (not ``print``) so output respects the log config.
"""
import logging
import os
from time import time
from typing import Any, Dict, List, Optional

import cohere
import numpy as np
import requests

logger = logging.getLogger(__name__)

# Global Cohere client instance for lazy loading
co = None
DEFAULT_COHERE_MODEL = "rerank-multilingual-v3.0"

# Global BGE model instance for lazy loading
bge_model = None


def _default_top_n() -> int:
    """Resolve the default top_n from env (RERANK_TOP_N), default 3."""
    try:
        return int(os.environ.get("RERANK_TOP_N", "3"))
    except ValueError:
        return 3


def _passthrough(docs: List[Dict[str, Any]], top_n: int, reason: str) -> List[Dict[str, Any]]:
    """Return docs unchanged (truncated to top_n) but flagged as rerank-failed.

    The pipeline keeps working, but downstream nodes can detect the lack of a
    real ``relevance_score`` via the ``rerank_failed`` flag instead of mistaking
    the passthrough for a successful rerank.
    """
    logger.warning("[RERANK] rerank unavailable — passthrough (%s)", reason)
    out = []
    for doc in docs[:top_n]:
        d = doc.copy()
        d["rerank_failed"] = True
        d["relevance_score"] = None
        out.append(d)
    return out


def rerank_documents(
    docs: List[Dict[str, Any]],
    query: str,
    top_n: Optional[int] = None,
    rank_model: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Rerank documents based on the query using Cohere or BGE CrossEncoder.

    Args:
        docs:       list of retrieved chunk dicts.
        query:      the user query (segmented/normalized by the caller).
        top_n:      number of docs to return; ``None`` reads ``RERANK_TOP_N`` env
                    (default 3).
        rank_model: override the rerank model id; ``None`` uses the backend default.

    Returns:
        Ranked docs each carrying ``relevance_score``. On any rerank failure
        returns the input docs (truncated) with ``rerank_failed=True`` and
        ``relevance_score=None`` so downstream can distinguish a passthrough.
    """
    global co

    if top_n is None:
        top_n = _default_top_n()

    # Check if docs list or query is empty
    if not docs:
        logger.info("[RERANK] Docs list is empty, skipping rerank")
        return []

    if not query or not query.strip():
        logger.info("[RERANK] Query is empty, returning original docs without rerank")
        return _passthrough(docs, top_n, "empty query")

    # Get reranker configuration from environment
    reranker_type = os.environ.get("RERANKER_TYPE", "cohere").lower().strip()

    # Create process_docs and filter empty ones
    process_docs: List[str] = []
    valid_doc_indices: List[int] = []

    for idx, doc in enumerate(docs):
        title = doc.get("title", "") or ""
        content = doc.get("content", "") or ""
        combined = f"{title} {content}".strip()

        if combined:
            process_docs.append(combined)
            valid_doc_indices.append(idx)

    # If no valid documents to rerank
    if not process_docs:
        logger.info("[RERANK] No valid documents to rerank")
        return _passthrough(docs, top_n, "no non-empty docs")

    # --- Cohere Rerank ---
    if reranker_type == "cohere":
        cohere_api_key = os.environ.get("COHERE_API_KEY") or os.environ.get("CO_API_KEY")
        if not cohere_api_key:
            return _passthrough(docs, top_n, "COHERE_API_KEY missing")

        if co is None:
            try:
                co = cohere.Client(cohere_api_key)
            except Exception as e:
                logger.warning("[RERANK] Failed to initialize Cohere client: %s", e)
                return _passthrough(docs, top_n, "Cohere client init failed")

        cohere_model = rank_model or DEFAULT_COHERE_MODEL
        logger.info("[RERANK] Reranking %d docs with Cohere model: %s", len(process_docs), cohere_model)

        try:
            results = co.rerank(
                query=query,
                documents=process_docs,
                top_n=min(top_n, len(process_docs)),
                model=cohere_model,
            )

            # Map results back to original documents
            ranked_docs = []
            for item in results.results:
                original_idx = valid_doc_indices[item.index]
                doc = docs[original_idx].copy()
                doc["relevance_score"] = float(item.relevance_score)
                doc["rerank_failed"] = False
                ranked_docs.append(doc)
                logger.info(
                    "[RERANK] Doc %d: %s - Score: %.5f",
                    original_idx,
                    str(doc.get("title", "No title"))[:50],
                    item.relevance_score,
                )

            return ranked_docs

        except Exception as e:
            logger.warning("[RERANK] Cohere rerank error: %s — passthrough", e)
            return _passthrough(docs, top_n, f"Cohere API error: {e}")

    # --- BGE Rerank (Local CrossEncoder) ---
    elif reranker_type == "bge":
        global bge_model
        bge_model_name = rank_model or os.environ.get("BGE_RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
        device = os.environ.get("BGE_RERANK_DEVICE", "cpu")

        if bge_model is None:
            logger.info("[RERANK] Loading local BGE model: %s on %s", bge_model_name, device)
            t0 = time()
            try:
                from sentence_transformers import CrossEncoder
                bge_model = CrossEncoder(bge_model_name, device=device)
                logger.info("[RERANK] Local BGE model loaded in %.2fs", time() - t0)
            except Exception as e:
                logger.warning(
                    "[RERANK] Failed to load BGE model: %s — ensure sentence-transformers+torch", e
                )
                return _passthrough(docs, top_n, "BGE model load failed")

        logger.info("[RERANK] Reranking %d docs with BGE CrossEncoder: %s", len(process_docs), bge_model_name)

        try:
            t0 = time()
            pairs = [[query, doc_text] for doc_text in process_docs]
            scores = bge_model.predict(pairs)
            logger.info("[RERANK] BGE rerank finished in %.2fs", time() - t0)

            # Construct ranked documents with score
            scored_docs = []
            for i, score in enumerate(scores):
                original_idx = valid_doc_indices[i]
                doc = docs[original_idx].copy()
                doc["relevance_score"] = float(score)
                doc["rerank_failed"] = False
                scored_docs.append(doc)

            # Sort by score descending
            scored_docs.sort(key=lambda x: x["relevance_score"], reverse=True)

            for i, doc in enumerate(scored_docs[:top_n]):
                logger.info(
                    "[RERANK] BGE Top %d: %s - Score: %.5f",
                    i + 1,
                    str(doc.get("title", "No title"))[:50],
                    doc["relevance_score"],
                )

            return scored_docs[:top_n]

        except Exception as e:
            logger.warning("[RERANK] BGE rerank error: %s — passthrough", e)
            return _passthrough(docs, top_n, f"BGE error: {e}")

    # --- No Reranker / Unknown Reranker ---
    else:
        return _passthrough(docs, top_n, f"RERANKER_TYPE='{reranker_type}' disabled/unknown")