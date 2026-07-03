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
from typing import Dict, List

from brain import groq_chat_complete
from config import VERIFY_ANSWER_THRESHOLD, VERIFY_PARTIAL_THRESHOLD
from evaluation.metrics_generation import evaluate_faithfulness

logger = logging.getLogger(__name__)

# Minimum answer length to bother judging — shorter is treated as unsupported.
_MIN_ANSWER_LEN = 15


def judge_answer(question: str, answer: str, sources: List[Dict]) -> Dict:
    """Score an answer against its retrieval sources.

    Args:
        question: standalone user question.
        answer: candidate final response text.
        sources: retrieved source chunks (RAG route only). Each entry may
            carry a ``content`` field used as grounding context.

    Returns:
        ``{"score": float, "rationale": str, "verdict": str}`` where verdict is
        one of ``supported`` / ``partial`` / ``unsupported``.
    """
    if not answer or len(answer.strip()) < _MIN_ANSWER_LEN:
        return {"score": 0.0, "rationale": "empty_answer", "verdict": "unsupported"}

    # Non-RAG routes (agent_tools / web_search / general_chat) carry no
    # source chunks, so citation groundedness is not checkable here. Mark
    # supported by default to avoid blocking those routes — the metacognitive
    # node still gates on stakes, and the agent route already calls
    # verify_citation per-assertion inside ReAct.
    if not sources:
        return {
            "score": 1.0,
            "rationale": "non-RAG route, citation check skipped",
            "verdict": "supported",
        }

    contexts = [s.get("content", "") for s in sources if s.get("content")]
    if not contexts:
        return {
            "score": 1.0,
            "rationale": "sources present but no content, citation check skipped",
            "verdict": "supported",
        }

    try:
        jr = evaluate_faithfulness(question, answer, contexts, judge_fn=groq_chat_complete)
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