"""Retrieval metrics: Hit@K, Recall@K, MRR, nDCG@K.

Each per-query function takes the 1-indexed rank of the gold document in the
retrieved list (``0`` means the gold doc was not retrieved). The
``aggregate_retrieval_metrics`` helper turns a list of ranks into a metric
dictionary suitable for reporting.
"""

from __future__ import annotations

import math
from typing import Dict, List


def hit_at_k(rank: int, k: int) -> float:
    """1.0 if gold appears in top-k, else 0.0."""
    return 1.0 if 0 < rank <= k else 0.0


def recall_at_k(rank: int, k: int, total_relevant: int = 1) -> float:
    """Recall@K assuming a single relevant doc per query (typical for QA).

    With one gold doc, recall@k is identical to hit@k. We keep this function
    separate so the API stays generic if multi-relevant ground truth is added
    later.
    """
    return hit_at_k(rank, k) / max(total_relevant, 1)


def mrr(rank: int) -> float:
    """Reciprocal rank: 1/rank if found, else 0."""
    return 1.0 / rank if rank > 0 else 0.0


def ndcg_at_k(rank: int, k: int) -> float:
    """nDCG@K with a single binary-relevant doc.

    DCG = 1 / log2(rank + 1) if rank <= k else 0.
    IDCG = 1 / log2(2) = 1.0 (gold at rank 1).
    """
    if not (0 < rank <= k):
        return 0.0
    return 1.0 / math.log2(rank + 1)


def aggregate_retrieval_metrics(
    ranks: List[int],
    ks: tuple = (1, 3, 5, 10),
) -> Dict[str, float]:
    """Average retrieval metrics across a list of per-query ranks.

    Parameters
    ----------
    ranks
        Per-query 1-indexed rank of the gold document (0 if not retrieved).
    ks
        Cutoffs to report.

    Returns
    -------
    Dict with keys: ``hit@k``, ``recall@k``, ``ndcg@k`` for each k, plus
    ``mrr`` and ``n_queries``.
    """
    n = len(ranks)
    if n == 0:
        return {"n_queries": 0}

    out: Dict[str, float] = {"n_queries": n}
    for k in ks:
        out[f"hit@{k}"] = sum(hit_at_k(r, k) for r in ranks) / n
        out[f"recall@{k}"] = sum(recall_at_k(r, k) for r in ranks) / n
        out[f"ndcg@{k}"] = sum(ndcg_at_k(r, k) for r in ranks) / n
    out["mrr"] = sum(mrr(r) for r in ranks) / n
    return out
