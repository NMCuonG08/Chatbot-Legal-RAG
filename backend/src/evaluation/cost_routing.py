"""Cost-aware model routing.

Pick a cheaper model for low-stakes routes (general_chat, simple agent_tools) and
reserve the big model for legal_rag. Reuses ``metrics_generation.PRICING_MAP``
for savings estimation so routing decisions are auditable against the same price
table used by the cost metric.

Public surface:
- ``CostRouteRule`` frozen.
- ``select_model_for_route``, ``estimate_route_savings``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# Default model assignment per route. Big model only for legal_rag (high-stakes,
# citation-grounded). Small/cheap for everything else.
SMALL_MODEL = "llama-3.1-8b-instant"
BIG_MODEL = "llama-3.3-70b-versatile"

_ROUTE_MODEL = {
    "legal_rag": BIG_MODEL,
    "agent_tools": SMALL_MODEL,   # calculation/lookup tools -> small
    "web_search": SMALL_MODEL,
    "general_chat": SMALL_MODEL,
}


@dataclass(frozen=True)
class CostRouteRule:
    route: str
    model: str
    reason: str


def select_model_for_route(route: Optional[str], difficulty: str = "medium") -> CostRouteRule:
    """Return the model to use for a given route + difficulty.

    Hard legal_rag questions always get the big model; medium legal_rag keeps
    the big model too (citation grounding needs capability). Other routes get
    the small model regardless of difficulty.
    """
    r = (route or "general_chat").lower()
    if r not in _ROUTE_MODEL:
        r = "general_chat"
    model = _ROUTE_MODEL[r]
    reason = f"route={r}"
    if r == "legal_rag" and difficulty == "hard":
        reason = f"route={r} difficulty=hard -> big model"
    return CostRouteRule(route=r, model=model, reason=reason)


def estimate_route_savings(route: Optional[str]) -> float:
    """Estimate per-1M-tokens savings (USD) vs always using the big model.

    Savings = big_model_avg_price - routed_model_avg_price, where avg_price is
    the mean of prompt + completion price from PRICING_MAP. Negative means the
    routed model is MORE expensive than the big baseline (should not happen).
    """
    from evaluation.metrics_generation import PRICING_MAP

    def _avg(model: str) -> float:
        p = PRICING_MAP.get(model, PRICING_MAP[SMALL_MODEL])
        return (p["prompt"] + p["completion"]) / 2.0

    rule = select_model_for_route(route)
    big_avg = _avg(BIG_MODEL)
    routed_avg = _avg(rule.model)
    return round(big_avg - routed_avg, 6)


__all__ = [
    "CostRouteRule",
    "select_model_for_route",
    "estimate_route_savings",
    "SMALL_MODEL",
    "BIG_MODEL",
]