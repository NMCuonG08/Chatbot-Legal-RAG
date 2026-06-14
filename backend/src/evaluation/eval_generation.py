"""Generation evaluation: produce answers with the full RAG pipeline and judge them."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List

logger = logging.getLogger(__name__)


@dataclass
class GenerationRunResult:
    """Per-sample generation evaluation record."""

    sample_id: str
    question: str
    answer: str
    contexts: List[str]
    scores: Dict[str, float]
    rationales: Dict[str, str]
    latency_ms: float


def _retrieve_for_generation(question: str, top_k: int) -> List[dict]:
    """Use the production retrieval pipeline (multi-query + hybrid + rerank)."""
    from query_rewriter import rewrite_query_to_multi_queries
    from rerank import rerank_documents
    from search import hybrid_search

    queries = rewrite_query_to_multi_queries(question, num_queries=3)
    seen, merged = set(), []
    for q in queries:
        for doc in hybrid_search(q, limit=top_k):
            key = hash(doc.get("content", ""))
            if key in seen:
                continue
            seen.add(key)
            merged.append(doc)
    return rerank_documents(merged, question, top_n=top_k)


def _generate_answer(question: str, contexts: List[str]) -> str:
    """Mirror ``tasks.generate_rag_answer`` but stripped down for eval."""
    from brain import gen_doc_prompt, vietnamese_llm_chat_complete

    docs = [{"question": question, "content": ctx} for ctx in contexts]
    doc_context = gen_doc_prompt(docs)

    system_prompt = (
        "Bạn là trợ lý AI chuyên về tư vấn pháp luật Việt Nam. "
        "Chỉ sử dụng thông tin từ tài liệu được cung cấp. "
        "Trích dẫn căn cứ pháp lý khi có thể."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"{doc_context}\n\nCâu hỏi: {question}\n\nHãy trả lời dựa trên các tài liệu trên.",
        },
    ]
    return vietnamese_llm_chat_complete(messages)


def run_generation_eval(
    samples,
    top_k: int = 5,
    judge_fn=None,
    progress_every: int = 5,
) -> List[GenerationRunResult]:
    """Run RAG generation + LLM-as-judge for each sample."""
    from brain import groq_chat_complete

    from .metrics_generation import evaluate_generation_sample

    judge_fn = judge_fn or groq_chat_complete

    results: List[GenerationRunResult] = []
    for i, sample in enumerate(samples, 1):
        t0 = time.perf_counter()
        try:
            docs = _retrieve_for_generation(sample.question, top_k)
            contexts = [str(d.get("content", "")) for d in docs if d.get("content")]
            answer = _generate_answer(sample.question, contexts)
        except Exception as exc:
            logger.warning("Generation failed for %s: %s", sample.sample_id, exc)
            contexts, answer = [], ""
        latency_ms = (time.perf_counter() - t0) * 1000

        scores_obj = evaluate_generation_sample(
            sample.question, answer, contexts, judge_fn
        )

        results.append(
            GenerationRunResult(
                sample_id=sample.sample_id,
                question=sample.question,
                answer=answer,
                contexts=contexts,
                scores=scores_obj.as_dict(),
                rationales=scores_obj.rationales,
                latency_ms=latency_ms,
            )
        )

        if i % progress_every == 0:
            logger.info("Generation eval: processed %d/%d", i, len(samples))

    return results


def summarize_generation_results(results: List[GenerationRunResult]) -> Dict[str, float]:
    """Aggregate per-sample generation results into mean scores + latency."""
    n = len(results)
    if n == 0:
        return {"n_queries": 0}

    def _mean_metric(key: str) -> float:
        values = [r.scores.get(key) for r in results if r.scores.get(key) is not None]
        return sum(values) / len(values) if values else 0.0

    latencies = [r.latency_ms for r in results]
    summary = {
        "n_queries": n,
        "faithfulness_mean": _mean_metric("faithfulness"),
        "answer_relevance_mean": _mean_metric("answer_relevance"),
        "context_precision_mean": _mean_metric("context_precision"),
        "latency_ms_mean": sum(latencies) / n,
    }
    sorted_lat = sorted(latencies)
    p95_idx = max(0, int(round(0.95 * (n - 1))))
    summary["latency_ms_p95"] = sorted_lat[p95_idx]
    return summary
