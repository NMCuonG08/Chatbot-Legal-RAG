"""End-to-end evaluation: route through the LangGraph chat graph used in production.

Unlike ``eval_generation`` which calls the RAG path directly, this runs the
real ``run_chat_graph`` so the routing decision (legal_rag / agent_tools /
web_search / general_chat) is included in the measurement.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class E2ERunResult:
    sample_id: str
    question: str
    answer: str
    latency_ms: float
    error: Optional[str] = None


def run_e2e_eval(samples, history: Optional[List[dict]] = None) -> List[E2ERunResult]:
    """Run each question through the production chat graph."""
    from tasks import run_chat_graph

    history = history or []
    results: List[E2ERunResult] = []
    for i, sample in enumerate(samples, 1):
        t0 = time.perf_counter()
        try:
            answer = run_chat_graph(history, sample.question)
            err = None
        except Exception as exc:
            answer = ""
            err = f"{type(exc).__name__}: {exc}"
            logger.warning("E2E failed for %s: %s", sample.sample_id, exc)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        results.append(
            E2ERunResult(
                sample_id=sample.sample_id,
                question=sample.question,
                answer=answer,
                latency_ms=elapsed_ms,
                error=err,
            )
        )
        if i % 5 == 0:
            logger.info("E2E eval: processed %d/%d", i, len(samples))
    return results


def summarize_e2e_results(results: List[E2ERunResult]) -> Dict[str, float]:
    n = len(results)
    if n == 0:
        return {"n_queries": 0}
    latencies = [r.latency_ms for r in results]
    success = [r for r in results if not r.error and r.answer]
    sorted_lat = sorted(latencies)
    p95_idx = max(0, int(round(0.95 * (n - 1))))
    return {
        "n_queries": n,
        "success_rate": len(success) / n,
        "latency_ms_mean": sum(latencies) / n,
        "latency_ms_p95": sorted_lat[p95_idx],
    }
