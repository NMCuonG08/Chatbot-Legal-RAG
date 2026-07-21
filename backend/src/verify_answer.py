"""PEV verify_answer — judge the FINAL agent/RAG answer for citation
groundedness + hallucination, before it leaves the graph.

Reuses ``evaluate_faithfulness`` (Ragas-style claim decomposition with a
server-side score) so a hallucinated aggregate score cannot pass. Only the
RAG route carries real ``sources``; the agent/web/general routes return
``sources: []`` and are short-circuited (no citation material to check).

Verdict vocabulary mirrors ``legal_retrieval_tools.verify_citation``:
``supported`` | ``partial`` | ``unsupported``.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional

from langsmith import traceable

from brain import build_judge_fn
from config import JUDGE_MODEL, JUDGE_PROVIDER, JUDGE_TEMPERATURE, VERIFY_ANSWER_THRESHOLD, VERIFY_PARTIAL_THRESHOLD
from evaluation.metrics_generation import evaluate_faithfulness

logger = logging.getLogger(__name__)

# Minimum answer length to bother judging — shorter is treated as unsupported.
_MIN_ANSWER_LEN = 15


@traceable(name="judge_answer", run_type="chain")
def judge_answer(question: str, answer: str, sources: List[Dict],
                 judge_fn: Optional[Callable] = None) -> Dict:
    """Score an answer against its retrieval sources.

    Args:
        question: standalone user question.
        answer: candidate final response text.
        sources: retrieved source chunks (RAG route only). Each entry may
            carry a ``content`` field used as grounding context.
        judge_fn: optional override judge callable (messages -> str). Defaults
            to a pinned judge built from JUDGE_PROVIDER/JUDGE_MODEL so a judge
            swap is auditable instead of hardcoded to the agent's Groq call.

    Returns:
        ``{"score": float, "rationale": str, "verdict": str}`` where verdict is
        one of ``supported`` / ``partial`` / ``unsupported``.
    """
    if not answer or len(answer.strip()) < _MIN_ANSWER_LEN:
        return {"score": 0.0, "rationale": "empty_answer", "verdict": "unsupported"}

    # Phase 5 — agent_tools / web_search now carry source chunks (collected
    # from retrieval-tool calls / Tavily). Only general_chat legitimately has
    # none. An empty ``sources`` here is still short-circuited (no citation
    # material to check) but now LOGGED so a regression on agent/web surfacing
    # empty sources is visible instead of silently passing.
    if not sources:
        logger.info("[VERIFY] no sources — citation groundedness skipped (general_chat or agent/web regression)")
        return {
            "score": 1.0,
            "rationale": "no_sources_citation_check_skipped",
            "verdict": "supported",
        }

    contexts = [s.get("content", "") for s in sources if s.get("content")]
    if not contexts:
        logger.info("[VERIFY] sources present but no content — citation groundedness skipped")
        return {
            "score": 1.0,
            "rationale": "sources_present_no_content_citation_check_skipped",
            "verdict": "supported",
        }

    j_fn = judge_fn or build_judge_fn(JUDGE_PROVIDER, JUDGE_MODEL, JUDGE_TEMPERATURE)
    try:
        jr = evaluate_faithfulness(question, answer, contexts, judge_fn=j_fn)
    except Exception as exc:  # judge failure must not break the graph
        logger.warning("verify_answer judge failed: %s", exc)
        return {"score": 0.0, "rationale": f"judge_error: {exc}", "verdict": "unsupported"}

    score = float(jr.score)
    if score >= VERIFY_ANSWER_THRESHOLD:
        verdict = "supported"
    elif score >= VERIFY_PARTIAL_THRESHOLD:
        verdict = "partial"
    else:
        verdict = "unsupported"
    return {"score": score, "rationale": jr.rationale, "verdict": verdict}