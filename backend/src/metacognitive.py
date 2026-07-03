"""Metacognitive escalation — graph-level safety gate.

Combines two signals into a 3-way decision (answer / recover / escalate):
- **confidence**: the verify_answer groundedness score (0.0–1.0).
- **stakes**: a tiered classifier (high / medium / low) over the user question.

Replaces the prior post-hoc disclaimer append (which only mutated text after
the fact) with a real graph node that runs before END, so an uncertain answer
on a high-stakes question is flagged explicitly and auditably in the trace.

Stakes tiers:
- **high**: criminal-defense / license-required topics. Always escalate,
  regardless of confidence — reuses ``ESCALATION_TOPICS`` from
  ``legal_knowledge_tools`` (the canonical tool-side list) as the high tier.
- **medium**: civil disputes with money/litigation, contracts, family /
  child custody, appeals. Escalate only when confidence is low.
- **low**: everything else. Never escalate.

The high-stakes list is exported as ``HIGH_STAKES_KEYWORDS`` so
``guardrails_manager`` can reuse it instead of maintaining a duplicate copy.
"""

from __future__ import annotations

import logging
from typing import Dict

from config import ESCALATION_CONFIDENCE_THRESHOLD
from legal_knowledge_tools import ESCALATION_TOPICS

logger = logging.getLogger(__name__)

# Canonical high-stakes list — single source of truth (was duplicated in
# guardrails_manager._ESCALATION_TOPICS). Criminal-defense / license-required
# topics always escalate to "consult a licensed lawyer".
HIGH_STAKES_KEYWORDS: list = list(ESCALATION_TOPICS)

# Medium-stakes: civil disputes, contracts, family/child custody, appeals.
# These escalate only when the answer confidence is low (below threshold).
# Note: avoid the bare token "kiện" — it substrings-matches "điều kiện"
# (condition), a common neutral legal word. Litigation is covered by
# "tranh chấp" / "đòi nợ" / "tòa án" instead.
MEDIUM_STAKES_KEYWORDS = [
    "tranh chấp", "hợp đồng", "ly hôn", "nuôi con",
    "đòi nợ", "bồi thường", "thiệt hại", "phúc thẩm dân sự",
    "chia tài sản", "gây ô nhiễm", "sao kê", "chuyển nhượng",
    "tòa án", "khởi kiện",
]

# Prefix prepended (never silently mutated) when escalating — keeps the
# original answer intact below it for auditability in the trace.
ESCALATION_PREFIX = (
    "⚠️ Đây là vấn đề pháp lý có hậu quả nặng. "
    "Tôi khuyến nghị bạn nên tham vấn với **luật sư hành nghề** "
    "trước khi quyết định.\n\n"
)


def classify_stakes(question: str) -> str:
    """Classify a question's stakes into ``high`` / ``medium`` / ``low``.

    Args:
        question: the standalone user question (case-insensitive match).

    Returns:
        One of ``"high"`` / ``"medium"`` / ``"low"``. High is checked first so
        a criminal topic always wins over a medium keyword match.
    """
    q = (question or "").lower()
    if any(k in q for k in HIGH_STAKES_KEYWORDS):
        return "high"
    if any(k in q for k in MEDIUM_STAKES_KEYWORDS):
        return "medium"
    return "low"


def should_escalate(stakes: str, confidence: float) -> bool:
    """Decide whether to surface the lawyer-escalation prefix.

    Args:
        stakes: result of ``classify_stakes``.
        confidence: verify_answer groundedness score (0.0–1.0). Treats a
            missing/zero confidence as low.

    Returns:
        True if the answer should be flagged for lawyer consultation.
    """
    if stakes == "high":
        return True
    if stakes == "medium":
        try:
            return float(confidence) < ESCALATION_CONFIDENCE_THRESHOLD
        except (TypeError, ValueError):
            return True
    return False


def build_escalation(question: str, confidence: float) -> Dict:
    """Compute the metacognitive decision payload for a graph step.

    Args:
        question: standalone user question.
        confidence: verify_answer groundedness score.

    Returns:
        ``{"stakes": str, "confidence": float, "escalate": bool}`` — the
        payload traced via ``_trace_node_end`` and used by the node to decide
        whether to prepend ``ESCALATION_PREFIX``.
    """
    stakes = classify_stakes(question)
    try:
        conf = float(confidence)
    except (TypeError, ValueError):
        conf = 0.0
    escalate = should_escalate(stakes, conf)
    return {"stakes": stakes, "confidence": conf, "escalate": escalate}