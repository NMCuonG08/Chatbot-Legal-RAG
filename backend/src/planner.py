"""LLM planner for multi-step legal queries (Phase 3).

Given a user question + recent history, the planner produces an ordered list
of steps, each assigning a specialist (``rag`` | ``tool`` | ``web`` | ``chat``)
and a goal. Simple queries yield a 1-step plan; multi-step queries (e.g.
"tính trợ cấp thôi việc rồi dẫn điều luật áp dụng") yield 2+ steps that the
supervisor (``supervisor.py``) walks through.

Public surface:
- ``build_plan_prompt(question, history_summary)`` — Vietnamese prompt asking
  the LLM for a ``<plan>`` block.
- ``parse_plan(text)`` — extract the step list from the LLM response.
- ``validate_plan(plan)`` — clamp to allowed specialists + MAX_PLAN_STEPS.

The LLM call itself is NOT made here — ``planner_node`` in tasks.py calls the
broker/LLM and feeds the response to ``parse_plan``. This keeps the module
import-light (no langchain/llama-index) and unit-testable.
"""
from __future__ import annotations

import logging
import re
from typing import Dict, List

from llm_json import extract_json

logger = logging.getLogger(__name__)

# Specialists the planner may assign. "rag" = vector retrieve + CRAG;
# "tool" = ReAct agent_tools; "web" = web_search; "chat" = general_chat.
SPECIALISTS = ("rag", "tool", "web", "chat")

MAX_PLAN_STEPS = 5

_PLAN_PROMPT_TMPL = """\
Bạn là bộ lập kế hoạch (planner) cho hệ thống pháp luật Việt Nam.
Phân tích câu hỏi và lịch sử, rồi đưa ra kế hoạch thực hiện dạng JSON:

{{"steps": [{{"specialist": "rag|tool|web|chat", "goal": "mô tả mục tiêu bước này bằng tiếng Việt"}}, ...]}}

Quy ước specialist:
- rag: tra cứu tài liệu luật / điều luật / án lệ (truy xuất vector).
- tool: tính toán pháp lý (phạt vi phạm, trợ cấp, thuế, phí, thừa kế...) hoặc gọi công cụ agent.
- web: tìm kiếm web khi cần thông tin mới ngoài kho luật.
- chat: trả lời hội thoại chung / giải thích khái niệm.

Quy tắc:
- Câu hỏi đơn giản -> đúng 1 bước.
- Câu hỏi nhiều ý -> nhiều bước, mỗi bước một specialist, tối đa {max_steps} bước.
- Chỉ trả về JSON, không giải thích, không markdown, không thẻ <plan>.

Câu hỏi: {question}
Lịch sử tóm tắt: {history}
"""


def build_plan_prompt(question: str, history_summary: str = "") -> str:
    return _PLAN_PROMPT_TMPL.format(
        question=question.strip(),
        history=(history_summary or "(trống)").strip(),
        max_steps=MAX_PLAN_STEPS,
    )


_STEP_RE = re.compile(
    r'<step\s+specialist\s*=\s*"([^"]+)"\s+goal\s*=\s*"([^"]*)"\s*/>',
    re.IGNORECASE | re.DOTALL,
)
_STEP_RE_LOOSE = re.compile(
    r'specialist\s*=\s*["\']?(\w+)["\']?[^>]*?goal\s*=\s*["\']([^"\']*)["\']',
    re.IGNORECASE | re.DOTALL,
)

_ALIAS = {
    "agent": "tool", "tools": "tool", "agent_tools": "tool",
    "retrieve": "rag", "legal_rag": "rag", "document": "rag",
    "web_search": "web", "search": "web",
    "general": "chat", "general_chat": "chat", "generalchat": "chat",
}


def _extract(pattern: re.Pattern, text: str) -> List[Dict[str, str]]:
    out = []
    for m in pattern.finditer(text):
        specialist = m.group(1).strip().lower()
        goal = m.group(2).strip()
        if specialist not in SPECIALISTS:
            specialist = _ALIAS.get(specialist, specialist)
        if specialist not in SPECIALISTS:
            logger.warning(f"[PLANNER] dropping step with unknown specialist={m.group(1)!r}")
            continue
        out.append({"specialist": specialist, "goal": goal})
    return out


def _json_steps(obj: object) -> List[Dict[str, str]]:
    """Schema-validate a parsed JSON object into plan steps. Returns [] if
    the object is not ``{"steps": [{"specialist","goal"}, ...]}`` or any step
    is malformed."""
    if not isinstance(obj, dict):
        return []
    raw_steps = obj.get("steps")
    if not isinstance(raw_steps, list):
        return []
    out: List[Dict[str, str]] = []
    for s in raw_steps:
        if not isinstance(s, dict):
            continue
        specialist = str(s.get("specialist", "")).strip().lower()
        goal = str(s.get("goal", "") or "").strip()
        if specialist not in SPECIALISTS:
            specialist = _ALIAS.get(specialist, specialist)
        if specialist not in SPECIALISTS or not goal:
            continue
        out.append({"specialist": specialist, "goal": goal})
    return out


def parse_plan(text: str) -> List[Dict[str, str]]:
    """Extract steps from the LLM plan response. Returns [] on no match.

    Primary path (structured stop condition): parse a JSON object
    ``{"steps": [{"specialist","goal"}, ...]}`` and schema-validate each step.
    Fallback path: the legacy ``<step .../>`` tag regex (kept so an LLM that
    emits the old format still works + the existing test suite stays green).

    Robust to: missing <plan> wrapper, attribute order, single vs double
    quotes, extra prose around the block, ```json``` fences.
    """
    if not text:
        return []
    steps = _json_steps(extract_json(text))
    if steps:
        return steps
    steps = _extract(_STEP_RE, text)
    if not steps:
        steps = _extract(_STEP_RE_LOOSE, text)
    return steps


def validate_plan(plan: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Clamp a parsed plan: drop empty plans, cap at MAX_PLAN_STEPS, dedupe
    trivially repeated consecutive identical steps."""
    if not plan:
        return []
    cleaned = []
    prev = None
    for step in plan[:MAX_PLAN_STEPS]:
        if step.get("specialist") not in SPECIALISTS:
            continue
        if prev and prev["specialist"] == step["specialist"] and prev["goal"] == step["goal"]:
            continue
        cleaned.append(step)
        prev = step
    return cleaned


def fallback_plan(route: str) -> List[Dict[str, str]]:
    """1-step plan derived from the existing router's classification, used when
    the LLM planner fails / is unavailable. Maps route -> specialist."""
    mapping = {
        "legal_rag": "rag",
        "agent_tools": "tool",
        "web_search": "web",
        "general_chat": "chat",
    }
    spec = mapping.get(route, "chat")
    return [{"specialist": spec, "goal": "xử lý theo route phân loại"}]


__all__ = [
    "SPECIALISTS",
    "MAX_PLAN_STEPS",
    "build_plan_prompt",
    "parse_plan",
    "validate_plan",
    "fallback_plan",
]