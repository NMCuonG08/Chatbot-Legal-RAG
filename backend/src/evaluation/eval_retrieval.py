"""Retrieval-only evaluation: ablate the components of the retrieval pipeline.

Configurations under test:
- vector            : pure dense vector search (Qdrant)
- hybrid            : BM25 + vector (existing ``hybrid_search``)
- hybrid_rerank     : hybrid + Cohere reranker
- multi_query       : multi-query rewriting + hybrid (full pipeline minus rerank)
- full              : multi-query + hybrid + rerank (production path)

Each configuration is run on the same sampled eval set and the gold context's
rank in the returned list is recorded.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List

logger = logging.getLogger(__name__)


@dataclass
class RetrievalRunResult:
    """Per-configuration retrieval run."""

    config: str
    ranks: List[int] = field(default_factory=list)
    latencies_ms: List[float] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)


def _retrieved_contents(docs: List[dict]) -> List[str]:
    return [str(d.get("content", "")) for d in docs]


def _config_vector(query: str, top_k: int) -> List[dict]:
    from brain import get_embedding
    from config import DEFAULT_COLLECTION_NAME
    from vectorize import search_vector

    vec = get_embedding(query)
    return search_vector(DEFAULT_COLLECTION_NAME, vec, top_k)


def _config_hybrid(query: str, top_k: int) -> List[dict]:
    from search import hybrid_search

    return hybrid_search(query, limit=top_k)


def _config_hybrid_rerank(query: str, top_k: int) -> List[dict]:
    from rerank import rerank_documents
    from search import hybrid_search

    docs = hybrid_search(query, limit=max(top_k * 2, 10))
    return rerank_documents(docs, query, top_n=top_k)


def _config_multi_query(query: str, top_k: int) -> List[dict]:
    from query_rewriter import rewrite_query_to_multi_queries
    from search import hybrid_search

    queries = rewrite_query_to_multi_queries(query, num_queries=3)
    seen, merged = set(), []
    for q in queries:
        for doc in hybrid_search(q, limit=top_k):
            key = hash(doc.get("content", ""))
            if key in seen:
                continue
            seen.add(key)
            merged.append(doc)
    merged.sort(key=lambda d: d.get("hybrid_score", d.get("similarity_score", 0)), reverse=True)
    return merged[:top_k]


def _config_full(query: str, top_k: int) -> List[dict]:
    from query_rewriter import rewrite_query_to_multi_queries
    from rerank import rerank_documents
    from search import hybrid_search

    queries = rewrite_query_to_multi_queries(query, num_queries=3)
    seen, merged = set(), []
    for q in queries:
        for doc in hybrid_search(q, limit=top_k):
            key = hash(doc.get("content", ""))
            if key in seen:
                continue
            seen.add(key)
            merged.append(doc)
    return rerank_documents(merged, query, top_n=top_k)


CONFIGS: Dict[str, Callable[[str, int], List[dict]]] = {
    "vector": _config_vector,
    "hybrid": _config_hybrid,
    "hybrid_rerank": _config_hybrid_rerank,
    "multi_query": _config_multi_query,
    "full": _config_full,
}


def run_retrieval_eval(
    samples,
    configs: List[str] = None,
    top_k: int = 10,
    progress_every: int = 10,
) -> Dict[str, RetrievalRunResult]:
    """Run each retrieval configuration on the sample set.

    Parameters
    ----------
    samples
        List of ``EvalSample`` objects.
    configs
        Subset of CONFIGS keys to run. Defaults to all.
    top_k
        How many docs to retrieve per query.
    progress_every
        Log every N samples.
    """
    from .dataset import gold_in_retrieved
    from .metrics_retrieval import aggregate_retrieval_metrics

    configs = configs or list(CONFIGS.keys())
    results: Dict[str, RetrievalRunResult] = {
        cfg: RetrievalRunResult(config=cfg) for cfg in configs
    }

    for i, sample in enumerate(samples, 1):
        for cfg in configs:
            fn = CONFIGS[cfg]
            t0 = time.perf_counter()
            try:
                docs = fn(sample.question, top_k)
            except Exception as exc:  # pragma: no cover - per-query failure
                logger.warning("Config %s failed on %s: %s", cfg, sample.sample_id, exc)
                docs = []
            elapsed_ms = (time.perf_counter() - t0) * 1000
            rank = gold_in_retrieved(sample, _retrieved_contents(docs))
            results[cfg].ranks.append(rank)
            results[cfg].latencies_ms.append(elapsed_ms)

        if i % progress_every == 0:
            logger.info("Retrieval eval: processed %d/%d", i, len(samples))

    # Aggregate
    for cfg, run in results.items():
        run.metrics = aggregate_retrieval_metrics(run.ranks, ks=(1, 3, 5, 10))
        run.metrics["latency_ms_mean"] = (
            sum(run.latencies_ms) / len(run.latencies_ms) if run.latencies_ms else 0.0
        )
        run.metrics["latency_ms_p95"] = _percentile(run.latencies_ms, 95)

    return results


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    k = max(0, min(len(sorted_vals) - 1, int(round(pct / 100 * (len(sorted_vals) - 1)))))
    return sorted_vals[k]
