import os
from time import time
from typing import List

import cohere
import numpy as np
import requests

# Set up Cohere client
COHERE_API_KEY = os.environ.get("COHERE_API_KEY") or os.environ.get("CO_API_KEY")
DEFAULT_COHERE_MODEL = "rerank-multilingual-v3.0"
co = cohere.Client(COHERE_API_KEY) if COHERE_API_KEY else None

# Global BGE model instance for lazy loading
bge_model = None


def rerank_documents(docs, query, top_n=3, rank_model=None):
    """
    Rerank documents based on the query using either Cohere or BGE CrossEncoder.
    """
    # Check if docs list or query is empty
    if not docs:
        print("[RERANK] Docs list is empty, skipping rerank")
        return []

    if not query or not query.strip():
        print("[RERANK] Query is empty, returning original docs without rerank")
        return docs[:top_n]

    # Get reranker configuration from environment
    reranker_type = os.environ.get("RERANKER_TYPE", "cohere").lower().strip()

    # Create process_docs and filter empty ones
    process_docs = []
    valid_doc_indices = []

    for idx, doc in enumerate(docs):
        title = doc.get("title", "") or ""
        content = doc.get("content", "") or ""
        combined = f"{title} {content}".strip()

        if combined:
            process_docs.append(combined)
            valid_doc_indices.append(idx)

    # If no valid documents to rerank
    if not process_docs:
        print("[RERANK] No valid documents to rerank")
        return docs[:top_n]

    # --- Cohere Rerank ---
    if reranker_type == "cohere":
        if not COHERE_API_KEY or co is None:
            print("[RERANK] COHERE_API_KEY not found, returning original docs")
            return docs[:top_n]

        cohere_model = rank_model or DEFAULT_COHERE_MODEL
        print(
            f"[RERANK] Reranking {len(process_docs)} documents with Cohere model: {cohere_model}..."
        )

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
                ranked_docs.append(doc)
                print(
                    f"[RERANK] Doc {original_idx}: {doc.get('title', 'No title')[:50]} - Score: {item.relevance_score:.5f}"
                )

            return ranked_docs

        except Exception as e:
            print(f"[RERANK] Error during Cohere reranking: {e}")
            print(f"[RERANK] Returning original docs without rerank")
            return docs[:top_n]

    # --- BGE Rerank (Local CrossEncoder) ---
    elif reranker_type == "bge":
        global bge_model
        bge_model_name = rank_model or os.environ.get("BGE_RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
        device = os.environ.get("BGE_RERANK_DEVICE", "cpu")

        if bge_model is None:
            print(f"[RERANK] Loading local BGE model: {bge_model_name} on {device}...")
            t0 = time()
            try:
                from sentence_transformers import CrossEncoder
                bge_model = CrossEncoder(bge_model_name, device=device)
                print(f"[RERANK] Local BGE model loaded successfully in {time() - t0:.2f}s")
            except Exception as e:
                print(f"[RERANK] Failed to load local BGE model: {e}")
                print("[RERANK] Please ensure 'sentence-transformers' and 'torch' are installed.")
                print("[RERANK] Returning original docs without rerank")
                return docs[:top_n]

        print(
            f"[RERANK] Reranking {len(process_docs)} documents with BGE CrossEncoder: {bge_model_name}..."
        )

        try:
            t0 = time()
            pairs = [[query, doc_text] for doc_text in process_docs]
            scores = bge_model.predict(pairs)
            print(f"[RERANK] BGE Reranking finished in {time() - t0:.2f}s")

            # Construct ranked documents with score
            scored_docs = []
            for i, score in enumerate(scores):
                original_idx = valid_doc_indices[i]
                doc = docs[original_idx].copy()
                doc["relevance_score"] = float(score)
                scored_docs.append(doc)

            # Sort by score descending
            scored_docs.sort(key=lambda x: x["relevance_score"], reverse=True)

            # Print top results
            for i, doc in enumerate(scored_docs[:top_n]):
                print(
                    f"[RERANK] BGE Top {i+1}: {doc.get('title', 'No title')[:50]} - Score: {doc['relevance_score']:.5f}"
                )

            return scored_docs[:top_n]

        except Exception as e:
            print(f"[RERANK] Error during BGE reranking: {e}")
            print(f"[RERANK] Returning original docs without rerank")
            return docs[:top_n]

    # --- No Reranker / Unknown Reranker ---
    else:
        print(f"[RERANK] Reranker type '{reranker_type}' is disabled or not recognized. Returning original docs.")
        return docs[:top_n]