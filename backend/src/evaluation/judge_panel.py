"""Judge hardening: swap augmentation, CoT-before-score, multi-judge panel,
inter-judge agreement (Cohen's kappa), calibration.

Mitigates the known LLM-as-judge biases:
- **position bias** -> ``swap_augment_pairwise`` runs both orderings and flips
  the sign; disagreement becomes a tie.
- **rubric verbosity / shallow scoring** -> ``cot_score`` asks the judge to
  reason before emitting a score (G-Eval style).
- **self-preference / single-family bias** -> ``multi_judge_panel`` aggregates
  several judges (Groq-Llama + Ollama-glm here); ``cohen_kappa`` measures
  agreement so a calibrated panel is auditable.

Groq + Ollama only. No OpenAI/Anthropic key in this project, so cross-family
diversity is partial (two Llama-family hosts) — documented as a limitation.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

from evaluation.metrics_generation import _parse_judge_json

logger = logging.getLogger(__name__)

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass(frozen=True)
class JudgeVote:
    judge_id: str
    score: float
    rationale: str = ""
    raw: str = ""


@dataclass(frozen=True)
class PanelVerdict:
    metric: str
    score: float
    votes: List[JudgeVote] = field(default_factory=list)
    aggregation: str = "mean"  # "mean" | "majority" | "swap"


@dataclass(frozen=True)
class CalibrationItem:
    question: str
    answer: str
    contexts: List[str]
    reference_score: float  # human gold score in [0,1]


# ---------------------------------------------------------------------------
# Swap augmentation (pairwise position-bias mitigation)
# ---------------------------------------------------------------------------


def _extract_preference(raw: str) -> Optional[str]:
    """Return 'A' | 'B' | 'tie' from a judge pairwise reply, else None."""
    parsed = _parse_judge_json(raw)
    if parsed:
        v = str(parsed.get("winner", parsed.get("preferred", ""))).strip().lower()
        if v in ("a", "answer_a", "1"):
            return "A"
        if v in ("b", "answer_b", "2"):
            return "B"
        if v in ("tie", "draw", "equal", "0", ""):
            return "tie"
    low = (raw or "").lower()
    if "answer_a" in low or "winner: a" in low or "a is better" in low:
        return "A"
    if "answer_b" in low or "winner: b" in low or "b is better" in low:
        return "B"
    if "tie" in low or "equal" in low:
        return "tie"
    return None


def _pairwise_prompt(question: str, a: str, b: str, a_label: str, b_label: str) -> str:
    return f"""So sánh hai câu trả lời cho cùng một câu hỏi pháp lý. Chọn câu trả lời TỐT HƠN.

Câu hỏi:
{question}

Câu trả lời {a_label}:
{a}

Câu trả lời {b_label}:
{b}

Tiêu chí: đúng trọng tâm, chính xác pháp lý, trích dẫn căn cứ, rõ ràng.
Trả về DUY NHẤT một JSON: {{"winner": "A" hoặc "B" hoặc "tie", "reason": "<ngắn>"}}"""


def _pref_to_score(p: Optional[str]) -> float:
    return {"A": 1.0, "B": -1.0, "tie": 0.0, None: 0.0}[p]


def swap_augment_pairwise(
    question: str, a: str, b: str, judge_fn: Callable,
    *, a_name: str = "A", b_name: str = "B",
) -> PanelVerdict:
    """Pairwise judge with position-bias mitigation via swap augmentation.

    Runs the judge twice (A-first, then B-first) and flips the second verdict.
    If the two runs disagree -> tie. Returns a PanelVerdict whose ``score`` is
    +1 (A wins), -1 (B wins), 0 (tie).
    """
    # Labels travel with the text (A=a, B=b in BOTH runs); only the physical
    # presentation order is swapped. So r1 and r2 are already in the same label
    # frame — NO flip needed. A content-faithful judge returns the same label
    # both runs (agree -> win); a position-biased judge picks the first-presented
    # label (run1 "A", run2 "B") -> disagree -> tie.
    r1 = _extract_preference(judge_fn([
        {"role": "system", "content": "Bạn là giám khảo công bằng. Trả lời JSON."},
        {"role": "user", "content": _pairwise_prompt(question, a, b, a_name, b_name)},
    ]))
    r2 = _extract_preference(judge_fn([
        {"role": "system", "content": "Bạn là giám khảo công bằng. Trả lời JSON."},
        {"role": "user", "content": _pairwise_prompt(question, b, a, b_name, a_name)},
    ]))

    if r1 is None and r2 is None:
        score, verdict = 0.0, "tie"
    elif r1 == r2:
        verdict = r1 or "tie"
        score = _pref_to_score(verdict)
    else:
        score, verdict = 0.0, "tie"
    return PanelVerdict(
        metric="pairwise", score=score, votes=[
            JudgeVote(judge_id="swap_run1", score=_pref_to_score(r1)),
            JudgeVote(judge_id="swap_run2", score=_pref_to_score(r2)),
        ], aggregation="swap",
    )


# ---------------------------------------------------------------------------
# CoT-before-score (G-Eval style) for single-score metrics
# ---------------------------------------------------------------------------


def _cot_prompt(metric: str, question: str, answer: str, contexts: str) -> str:
    return f"""Hãy đánh giá {metric} của câu trả lời. Suy nghĩ từng bước trước khi chấm điểm.

Câu hỏi: {question}
Ngữ cảnh: {contexts}
Câu trả lời: {answer}

Bước 1: Liệt kê 2-3 nhận xét ngắn về {metric} của câu trả lời.
Bước 2: Dựa trên nhận xét, chấm điểm từ 0.0 đến 1.0.

Trả về DUY NHẤT một JSON: {{"reasoning": "<nhận xét>", "score": <0.0..1.0>}}"""


def cot_score(metric: str, question: str, answer: str, contexts: str,
              judge_fn: Callable) -> JudgeVote:
    """CoT-before-score single judge (G-Eval style). Returns a JudgeVote."""
    raw = judge_fn([
        {"role": "system", "content": "Bạn là giám khảo chuyên gia. Luôn trả lời JSON hợp lệ."},
        {"role": "user", "content": _cot_prompt(metric, question, answer, contexts)},
    ])
    parsed = _parse_judge_json(raw)
    try:
        score = float(parsed.get("score", 0))
    except (TypeError, ValueError):
        score = 0.0
    score = max(0.0, min(1.0, score))
    return JudgeVote(judge_id="cot", score=score,
                     rationale=str(parsed.get("reasoning", "")), raw=raw)


# ---------------------------------------------------------------------------
# Multi-judge panel aggregation
# ---------------------------------------------------------------------------


def multi_judge_panel(
    judges: Sequence[Callable],
    prompt_builder: Callable[[Callable, str], JudgeVote],
    *,
    metric: str = "metric",
    aggregation: str = "mean",
) -> PanelVerdict:
    """Run ``prompt_builder(judge, judge_id)`` over each judge and aggregate.

    ``aggregation``: "mean" averages scores; "majority" takes the median (robust
    to a single outlier judge). Votes from judges that raise are dropped (not
    fatal) — a panel that fully fails returns score 0.0.
    """
    votes: List[JudgeVote] = []
    for i, j in enumerate(judges):
        try:
            votes.append(prompt_builder(j, f"judge_{i}"))
        except Exception as exc:
            logger.warning("panel judge %d failed: %s", i, exc)
    if not votes:
        return PanelVerdict(metric=metric, score=0.0, votes=[], aggregation=aggregation)
    scores = sorted(v.score for v in votes)
    if aggregation == "majority":
        score = scores[len(scores) // 2]
    else:
        score = sum(scores) / len(scores)
    return PanelVerdict(metric=metric, score=score, votes=votes, aggregation=aggregation)


def build_default_judge_panel() -> List[Callable]:
    """Build the project default panel: Groq-Llama + Ollama (if available).

    Ollama is included only when its base URL is configured, so offline/CI runs
    without Ollama get a single-judge panel instead of crashing.
    """
    import os
    from brain import build_judge_fn
    panel = [build_judge_fn("groq", os.environ.get("JUDGE_MODEL", "llama-3.1-8b-instant"))]
    if os.environ.get("OLLAMA_BASE_URL") or os.environ.get("OLLAMA_API_KEY"):
        ollama_model = (os.environ.get("OLLAMA_LLM_MODEL")
                        or os.environ.get("OLLAMA_MODEL") or "llama3.1:latest")
        panel.append(build_judge_fn("ollama", ollama_model))
    return panel


# ---------------------------------------------------------------------------
# Inter-judge agreement: Cohen's kappa
# ---------------------------------------------------------------------------


def cohen_kappa(scores_a: Sequence, scores_b: Sequence,
                levels: Optional[Sequence] = None) -> float:
    """Cohen's kappa over two raters' categorical labels (aligned).

    ``levels`` enumerates the category set; defaults to the union of observed
    labels. Returns 1.0 for perfect agreement, 0.0 for chance/no agreement.
    """
    a = list(scores_a)
    b = list(scores_b)
    if len(a) != len(b) or not a:
        return 0.0
    cats = list(levels) if levels is not None else sorted(set(a) | set(b))
    n = len(a)
    obs = sum(1 for x, y in zip(a, b) if x == y) / n
    marg_a = {c: a.count(c) / n for c in cats}
    marg_b = {c: b.count(c) / n for c in cats}
    exp = sum(marg_a[c] * marg_b[c] for c in cats)
    if exp == 1.0:
        return 1.0  # both constant on the same label
    if exp == 0.0:
        return 0.0
    return (obs - exp) / (1.0 - exp)


def _bin(score: float) -> str:
    if score < 0.34:
        return "low"
    if score < 0.67:
        return "mid"
    return "high"


def calibration_kappa(
    calibration_set: Sequence[CalibrationItem],
    judge_a: Callable,
    judge_b: Callable,
) -> float:
    """Cohen's kappa between two judges over a calibration set (binned scores)."""
    bins_a: List[str] = []
    bins_b: List[str] = []
    for item in calibration_set:
        ctx = "\n\n".join(item.contexts)
        va = cot_score("faithfulness", item.question, item.answer, ctx, judge_a)
        vb = cot_score("faithfulness", item.question, item.answer, ctx, judge_b)
        bins_a.append(_bin(va.score))
        bins_b.append(_bin(vb.score))
    return cohen_kappa(bins_a, bins_b, levels=["low", "mid", "high"])


__all__ = [
    "JudgeVote",
    "PanelVerdict",
    "CalibrationItem",
    "swap_augment_pairwise",
    "cot_score",
    "multi_judge_panel",
    "build_default_judge_panel",
    "cohen_kappa",
    "calibration_kappa",
]