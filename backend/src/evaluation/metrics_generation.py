"""LLM-as-judge generation metrics for Vietnamese legal RAG.

Three Ragas-style metrics, implemented directly so we can run on Vietnamese
content without depending on Ragas/OpenAI:

- faithfulness        : answer claims are grounded in retrieved context
- answer_relevance    : answer addresses the user question
- context_precision   : how many of the top-K retrieved chunks are relevant

The judge is the project's existing Groq-backed LLM (``brain.groq_chat_complete``).
Each metric returns a float in [0, 1] plus a short rationale for auditing.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class JudgeResult:
    """Output of an LLM judge call."""

    score: float
    rationale: str = ""
    raw: str = ""


@dataclass
class GenerationScores:
    """Aggregate per-sample generation scores."""

    faithfulness: Optional[float] = None
    answer_relevance: Optional[float] = None
    context_precision: Optional[float] = None
    rationales: Dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Optional[float]]:
        return {
            "faithfulness": self.faithfulness,
            "answer_relevance": self.answer_relevance,
            "context_precision": self.context_precision,
        }


# ---------------------------------------------------------------------------
# Judge plumbing
# ---------------------------------------------------------------------------


_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_judge_json(raw: str) -> dict:
    """Best-effort extraction of a JSON object from an LLM response."""
    if not raw:
        return {}
    match = _JSON_BLOCK_RE.search(raw)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        # Try a lenient fallback: replace single quotes with double.
        try:
            return json.loads(match.group(0).replace("'", '"'))
        except Exception:
            return {}


def _clamp_unit(value: float) -> float:
    if value < 0:
        return 0.0
    if value > 1:
        return 1.0
    return float(value)


def _call_judge(prompt: str, judge_fn) -> JudgeResult:
    """Send a prompt to the judge LLM and parse a {score, reason} JSON reply."""
    messages = [
        {
            "role": "system",
            "content": (
                "Bạn là chuyên gia đánh giá chất lượng hệ thống RAG tiếng Việt. "
                "Bạn LUÔN trả lời bằng JSON hợp lệ duy nhất, không thêm văn bản khác."
            ),
        },
        {"role": "user", "content": prompt},
    ]
    try:
        raw = judge_fn(messages)
    except Exception as exc:  # pragma: no cover - judge errors are non-fatal
        logger.warning("Judge call failed: %s", exc)
        return JudgeResult(score=0.0, rationale=f"judge_error: {exc}", raw="")

    raw = (raw or "").strip()
    parsed = _parse_judge_json(raw)
    score = parsed.get("score", parsed.get("rating", 0))
    try:
        score = float(score)
    except (TypeError, ValueError):
        score = 0.0
    rationale = str(parsed.get("reason", parsed.get("rationale", "")))
    return JudgeResult(score=_clamp_unit(score), rationale=rationale, raw=raw)


# ---------------------------------------------------------------------------
# Metric prompts
# ---------------------------------------------------------------------------


def _faithfulness_prompt(question: str, answer: str, context: str) -> str:
    return f"""Hãy đánh giá độ TRUNG THỰC (faithfulness) của câu trả lời so với tài liệu được cung cấp.

Định nghĩa:
- Score = 1.0 khi mọi nhận định trong câu trả lời đều có thể truy ngược được vào tài liệu.
- Score = 0.5 khi một phần nhận định có trong tài liệu, một phần bị suy diễn hoặc bịa.
- Score = 0.0 khi câu trả lời mâu thuẫn hoặc bịa hoàn toàn so với tài liệu.

Câu hỏi:
{question}

Tài liệu được cung cấp:
{context}

Câu trả lời cần đánh giá:
{answer}

Trả về DUY NHẤT một JSON dạng:
{{"score": <0.0..1.0>, "reason": "<lý do ngắn 1-2 câu>"}}"""


def _answer_relevance_prompt(question: str, answer: str) -> str:
    return f"""Hãy đánh giá độ LIÊN QUAN (answer relevance) của câu trả lời so với câu hỏi.

Định nghĩa:
- Score = 1.0 khi câu trả lời trực tiếp giải đáp đúng trọng tâm câu hỏi.
- Score = 0.5 khi câu trả lời chạm vào chủ đề nhưng lạc đề một phần.
- Score = 0.0 khi câu trả lời lạc đề hoàn toàn hoặc né tránh.

Câu hỏi:
{question}

Câu trả lời:
{answer}

Trả về DUY NHẤT một JSON dạng:
{{"score": <0.0..1.0>, "reason": "<lý do ngắn 1-2 câu>"}}"""


def _context_precision_prompt(question: str, contexts: List[str]) -> str:
    joined = "\n\n".join(
        f"[Tài liệu {i+1}]\n{ctx}" for i, ctx in enumerate(contexts)
    )
    return f"""Đánh giá ĐỘ CHÍNH XÁC NGỮ CẢNH (context precision): trong các tài liệu được truy xuất, có bao nhiêu phần trăm thực sự liên quan tới câu hỏi.

Hướng dẫn:
- Đếm số tài liệu RELEVANT (cung cấp thông tin có thể trả lời câu hỏi, dù chỉ một phần).
- Score = số tài liệu relevant / tổng số tài liệu.

Câu hỏi:
{question}

Các tài liệu được truy xuất:
{joined}

Trả về DUY NHẤT một JSON dạng:
{{"score": <0.0..1.0>, "relevant_indices": [<các chỉ số 1-based>], "reason": "<lý do ngắn>"}}"""


# ---------------------------------------------------------------------------
# Public metric functions
# ---------------------------------------------------------------------------


def evaluate_faithfulness(
    question: str,
    answer: str,
    contexts: List[str],
    judge_fn,
) -> JudgeResult:
    """Score how grounded ``answer`` is in ``contexts`` for ``question``."""
    if not answer or not answer.strip():
        return JudgeResult(score=0.0, rationale="empty_answer")
    if not contexts:
        return JudgeResult(score=0.0, rationale="no_context")
    joined_ctx = "\n\n".join(contexts)
    return _call_judge(
        _faithfulness_prompt(question, answer, joined_ctx),
        judge_fn,
    )


def evaluate_answer_relevance(
    question: str,
    answer: str,
    judge_fn,
) -> JudgeResult:
    """Score how well ``answer`` addresses ``question``."""
    if not answer or not answer.strip():
        return JudgeResult(score=0.0, rationale="empty_answer")
    return _call_judge(
        _answer_relevance_prompt(question, answer),
        judge_fn,
    )


def evaluate_context_precision(
    question: str,
    contexts: List[str],
    judge_fn,
) -> JudgeResult:
    """Score the proportion of retrieved contexts that are actually relevant."""
    if not contexts:
        return JudgeResult(score=0.0, rationale="no_context")
    return _call_judge(
        _context_precision_prompt(question, contexts),
        judge_fn,
    )


def evaluate_generation_sample(
    question: str,
    answer: str,
    contexts: List[str],
    judge_fn,
) -> GenerationScores:
    """Run all three generation metrics for one (question, answer, contexts) tuple."""
    faith = evaluate_faithfulness(question, answer, contexts, judge_fn)
    rel = evaluate_answer_relevance(question, answer, judge_fn)
    prec = evaluate_context_precision(question, contexts, judge_fn)
    return GenerationScores(
        faithfulness=faith.score,
        answer_relevance=rel.score,
        context_precision=prec.score,
        rationales={
            "faithfulness": faith.rationale,
            "answer_relevance": rel.rationale,
            "context_precision": prec.rationale,
        },
    )


def aggregate_generation_scores(
    samples: List[GenerationScores],
) -> Dict[str, float]:
    """Average non-None scores across samples."""
    n = len(samples)
    if n == 0:
        return {"n_queries": 0}

    def _mean(key: str) -> float:
        values = [getattr(s, key) for s in samples if getattr(s, key) is not None]
        return sum(values) / len(values) if values else 0.0

    return {
        "n_queries": n,
        "faithfulness_mean": _mean("faithfulness"),
        "answer_relevance_mean": _mean("answer_relevance"),
        "context_precision_mean": _mean("context_precision"),
    }
