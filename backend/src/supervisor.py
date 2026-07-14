"""LLM supervisor for inter-specialist handoff (Phase 3).

After a specialist (rag / tool / web / chat) produces an answer, the
supervisor decides whether to hand off to another specialist or finish
(END). It prefers an LLM decision; on any LLM failure / unavailability it
falls back to the same Vietnamese keyword heuristics the graph already used
(``_HANDOFF_*_MARKERS``), so behavior never regresses.

Public surface:
- ``build_supervisor_prompt(question, current_specialist, answer, plan)`` —
  Vietnamese prompt asking the LLM for a ``<handoff>`` decision.
- ``parse_supervisor_decision(text)`` -> ``{next, rationale}``.
- ``heuristic_handoff(current_specialist, answer)`` -> next specialist or
  ``"END"`` (the deterministic fallback).
- ``supervisor_decide(..., llm_call=None)`` -> ``{next, rationale, source}``
  where source is ``"llm"`` or ``"heuristic"``. ``next`` is one of
  SPECIALISTS or ``"END"``. Enforces ``MAX_HANDOFF_STEPS``: once the budget
  is exhausted, forces END regardless of the decision.

Import-light (no langchain/llama-index) so it is unit-testable in isolation.
"""
from __future__ import annotations

import logging
import re
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

SPECIALISTS = ("rag", "tool", "web", "chat")
END = "end"
MAX_HANDOFF_STEPS = 5

# Mirrors tasks._HANDOFF_NOT_FOUND_MARKERS / _HANDOFF_NEEDS_LOOKUP_MARKERS.
# Duplicated here to avoid a circular import (tasks imports supervisor).
_NOT_FOUND_MARKERS = (
    "không tìm thấy", "không tìm thấy thông tin", "không có đủ thông tin",
    "tôi không có thông tin", "không có trong tài liệu", "vượt quá khả năng",
)
_NEEDS_LOOKUP_MARKERS = (
    "cần tra cứu", "tra cứu văn bản", "tôi sẽ tra cứu", "hãy tham khảo văn bản",
)

_SUPERVISOR_PROMPT_TMPL = """\
Bạn là bộ giám sát (supervisor) của hệ thống pháp luật Việt Nam.
Một specialist vừa xử lý xong một bước. Quyết định bước tiếp theo.

Specialist hiện tại: {current}
Câu hỏi gốc: {question}
Kế hoạch: {plan}
Kết quả specialist vừa làm: {answer}

Trả lời đúng một dòng dạng:
<handoff next="rag|tool|web|chat|END" rationale="lý do ngắn bằng tiếng Việt" />

- next="END" khi câu hỏi đã được trả lời đầy đủ.
- next là specialist khác khi cần bổ sung thông tin/tính toán.
- Chỉ trả về thẻ <handoff>, không thêm gì khác.
"""


def build_supervisor_prompt(
    question: str, current_specialist: str, answer: str, plan: List[Dict[str, str]]
) -> str:
    plan_str = "; ".join(f'{s["specialist"]}: {s["goal"]}' for s in plan) or "(1 bước)"
    return _SUPERVISOR_PROMPT_TMPL.format(
        current=current_specialist or "?",
        question=(question or "").strip(),
        plan=plan_str,
        answer=(answer or "")[:2000],
    )


_HANDOFF_RE = re.compile(
    r'<handoff\s+next\s*=\s*"([^"]+)"\s+rationale\s*=\s*"([^"]*)"\s*/>',
    re.IGNORECASE | re.DOTALL,
)
_HANDOFF_RE_LOOSE = re.compile(
    r'next\s*=\s*["\']?(\w+)["\']?[^>]*?rationale\s*=\s*["\']([^"\']*)["\']',
    re.IGNORECASE | re.DOTALL,
)

_ALIAS = {
    "agent": "tool", "tools": "tool", "agent_tools": "tool",
    "retrieve": "rag", "legal_rag": "rag",
    "web_search": "web", "search": "web",
    "general": "chat", "general_chat": "chat",
    "stop": "end", "done": "end", "finish": "end",
}


def parse_supervisor_decision(text: str) -> Dict[str, str]:
    """Return {next, rationale} from the LLM handoff response, or
    ``{"next": "", "rationale": ""}`` when no tag is found."""
    if not text:
        return {"next": "", "rationale": ""}
    for pattern in (_HANDOFF_RE, _HANDOFF_RE_LOOSE):
        m = pattern.search(text)
        if m:
            nxt = m.group(1).strip().lower()
            if nxt not in SPECIALISTS and nxt != "end":
                nxt = _ALIAS.get(nxt, "")
            return {"next": nxt, "rationale": m.group(2).strip()}
    return {"next": "", "rationale": ""}


def heuristic_handoff(current_specialist: str, answer: str) -> str:
    """Deterministic fallback. Returns next specialist or END."""
    if not answer:
        return END
    low = answer.lower()
    needs_lookup = any(m in low for m in _NEEDS_LOOKUP_MARKERS)
    not_found = any(m in low for m in _NOT_FOUND_MARKERS)

    if current_specialist == "tool":
        # agent_tools -> retrieve when the agent says it needs legal-doc lookup.
        return "rag" if needs_lookup else END
    if current_specialist == "rag":
        # RAG canned not-found -> try web; explicit needs-lookup (rare) -> tool.
        if not_found:
            return "web"
        if needs_lookup:
            return "tool"
        return END
    if current_specialist == "web":
        # web result looks like a question needing tool use.
        return "tool" if needs_lookup else END
    if current_specialist == "chat":
        return END
    return END


def supervisor_decide(
    question: str,
    current_specialist: str,
    answer: str,
    plan: List[Dict[str, str]],
    steps_taken: int = 0,
    llm_call: Optional[Callable[[str], str]] = None,
) -> Dict[str, str]:
    """Decide the next specialist or END.

    ``llm_call`` is a ``prompt -> response text`` callable (the broker/LLM).
    When None or when it raises / returns an unparseable response, fall back to
    ``heuristic_handoff``. ``steps_taken`` is the number of handoffs already
    executed this run; once it reaches ``MAX_HANDOFF_STEPS`` the supervisor
    forces END (loop guard).
    """
    if steps_taken >= MAX_HANDOFF_STEPS:
        logger.info(f"[SUPERVISOR] step budget exhausted ({steps_taken}) -> END")
        return {"next": END, "rationale": "giới hạn số bước đạt tối đa", "source": "guard"}

    if llm_call is not None:
        try:
            prompt = build_supervisor_prompt(question, current_specialist, answer, plan)
            raw = llm_call(prompt)
            decision = parse_supervisor_decision(raw)
            if decision["next"] in (*SPECIALISTS, END) and decision["next"]:
                decision["source"] = "llm"
                return decision
            logger.warning(f"[SUPERVISOR] LLM returned unparseable handoff: {raw!r}")
        except Exception as exc:
            logger.warning(f"[SUPERVISOR] LLM call failed, using heuristic: {exc}")

    nxt = heuristic_handoff(current_specialist, answer)
    return {"next": nxt, "rationale": "heuristic fallback", "source": "heuristic"}


__all__ = [
    "SPECIALISTS",
    "END",
    "MAX_HANDOFF_STEPS",
    "build_supervisor_prompt",
    "parse_supervisor_decision",
    "heuristic_handoff",
    "supervisor_decide",
]