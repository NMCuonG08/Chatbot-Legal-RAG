"""RAG evaluation package.

Modules
-------
- dataset           : Load and sample evaluation set from train.jsonl
- metrics_retrieval : Hit@K, Recall@K, MRR, nDCG@K
- metrics_generation: LLM-as-judge faithfulness / answer relevance / context precision
- eval_retrieval    : Run retrieval ablation (vector vs hybrid vs hybrid+rerank)
- eval_generation   : Run generation evaluation
- eval_e2e          : Run end-to-end evaluation with latency
- run_eval          : CLI orchestrator that produces JSON + Markdown report
"""

from .dataset import EvalSample, load_eval_dataset
from .metrics_retrieval import (
    hit_at_k,
    recall_at_k,
    mrr,
    ndcg_at_k,
    aggregate_retrieval_metrics,
)

__all__ = [
    "EvalSample",
    "load_eval_dataset",
    "hit_at_k",
    "recall_at_k",
    "mrr",
    "ndcg_at_k",
    "aggregate_retrieval_metrics",
]
