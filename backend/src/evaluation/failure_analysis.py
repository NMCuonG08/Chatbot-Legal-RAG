"""Failure analysis and classification module for RAG evaluation.

Categorizes errors into Retrieval Failure, Routing Failure, Hallucination,
Answer Irrelevance, or Execution Error.
"""

from __future__ import annotations

from typing import Dict, List, Optional


class FailureCategory:
    SUCCESS = "success"
    ERROR = "execution_error"
    ROUTING_FAIL = "routing_failure"
    RETRIEVER_FAIL = "retrieval_failure"
    HALLUCINATION = "hallucination_failure"
    IRRELEVANCE = "irrelevance_failure"


def classify_sample_failure(
    *,
    error: Optional[str] = None,
    actual_route: Optional[str] = None,
    expected_route: Optional[str] = None,
    retrieval_hit: Optional[bool] = None,  # True if gold context was retrieved
    faithfulness: Optional[float] = None,
    answer_relevance: Optional[float] = None,
    faithfulness_threshold: float = 0.7,
    relevance_threshold: float = 0.7,
) -> str:
    """Classify the primary failure mode of a single evaluation sample.

    Prioritizes root causes:
    Execution Error > Routing Failure > Retrieval Failure > Hallucination > Irrelevance.
    """
    if error:
        return FailureCategory.ERROR

    # 1. Check Routing Failure (for E2E)
    if actual_route and expected_route and actual_route != expected_route:
        # Special allowance: if it went to agent_tools or legal_rag, both are legal,
        # so don't penalize unless they mismatch specifically on a calculation question
        legal_routes = {"legal_rag", "agent_tools"}
        if not (actual_route in legal_routes and expected_route in legal_routes):
            return FailureCategory.ROUTING_FAIL

    # 2. Check Retrieval Failure
    if retrieval_hit is False:
        return FailureCategory.RETRIEVER_FAIL

    # 3. Check Generation Failures
    if faithfulness is not None and faithfulness < faithfulness_threshold:
        return FailureCategory.HALLUCINATION

    if answer_relevance is not None and answer_relevance < relevance_threshold:
        return FailureCategory.IRRELEVANCE

    return FailureCategory.SUCCESS


def summarize_failures(categories: List[str]) -> Dict[str, float]:
    """Calculate percentages of each failure mode in the evaluation run."""
    total = len(categories)
    if total == 0:
        return {}

    counts = {
        FailureCategory.SUCCESS: 0,
        FailureCategory.ERROR: 0,
        FailureCategory.ROUTING_FAIL: 0,
        FailureCategory.RETRIEVER_FAIL: 0,
        FailureCategory.HALLUCINATION: 0,
        FailureCategory.IRRELEVANCE: 0,
    }

    for cat in categories:
        if cat in counts:
            counts[cat] += 1
        else:
            counts[FailureCategory.SUCCESS] += 1  # default fallback

    return {cat: count / total for cat, count in counts.items()}
