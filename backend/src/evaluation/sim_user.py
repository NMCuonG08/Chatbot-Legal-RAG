"""LLM-as-user simulator for multi-turn scenario eval (tau-bench style).

A ``UserPersona`` drives a simulated user that converses with the agent until
the scenario goal is met or ``max_turns`` is hit. The persona LLM call uses
``brain.groq_chat_complete`` at temperature 0.9 for variability; tests inject a
stub callable.

Public surface:
- ``UserPersona``, ``Turn``, ``ConversationLog`` — frozen.
- ``build_persona``, ``persona_message``, ``simulate_conversation``.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class UserPersona:
    name: str
    role: str  # layperson | business | junior_lawyer
    traits: tuple = ()  # vague | terse | multilingual | precise
    language: str = "vi"
    system_prompt: str = ""


@dataclass(frozen=True)
class Turn:
    turn_idx: int
    user_msg: str
    agent_reply: str
    route: Optional[str] = None
    tool_calls: tuple = ()
    latency_ms: float = 0.0


@dataclass(frozen=True)
class ConversationLog:
    scenario_id: str
    persona: UserPersona
    turns: tuple = ()
    reached_goal: bool = False
    truncated: bool = False


_PERSONA_TEMPLATES = {
    "layperson": (
        "Bạn là người dùng phổ thông, không rành luật. Hỏi ngắn gọn, đôi khi mơ hồ, "
        "có thể dùng tiếng Việt phổ thông. Khi đã đủ thông tin hãy nói 'Cảm ơn, tôi đã hiểu rõ.'"
    ),
    "business": (
        "Bạn là chủ doanh nghiệp nhỏ. Hỏi thực tế, tập trung vào rủi ro và thủ tục. "
        "Khi đủ thông tin hãy nói 'Cảm ơn, tôi đã hiểu rõ.'"
    ),
    "junior_lawyer": (
        "Bạn là luật sư junior. Hỏi chính xác, dùng thuật ngữ pháp lý, yêu cầu căn cứ "
        "điều luật. Khi đủ thông tin hãy nói 'Cảm ơn, tôi đã hiểu rõ.'"
    ),
}


def build_persona(role: str, *, name: Optional[str] = None,
                  traits: tuple = (), language: str = "vi") -> UserPersona:
    role = role if role in _PERSONA_TEMPLATES else "layperson"
    return UserPersona(
        name=name or f"persona-{role}",
        role=role,
        traits=tuple(traits),
        language=language,
        system_prompt=_PERSONA_TEMPLATES[role],
    )


def persona_message(persona: UserPersona, scenario_goal: str,
                    last_agent_reply: str, llm_fn: Callable) -> str:
    """Generate the next user utterance given the agent's last reply.

    ``llm_fn(messages) -> str``. Injects a termination cue: the persona is told
    to say the done-phrase when its goal is satisfied.
    """
    prompt = (
        f"{persona.system_prompt}\n\n"
        f"Mục tiêu của bạn: {scenario_goal}\n\n"
        f"Câu trả lời vừa rồi của trợ lý:\n{last_agent_reply}\n\n"
        f"Trả lời NGẮN (1-3 câu) bằng {persona.language}. Nếu trợ lý đã trả lời "
        f"đủ cho mục tiêu, chỉ nói: 'Cảm ơn, tôi đã hiểu rõ.'"
    )
    msgs = [
        {"role": "system", "content": persona.system_prompt},
        {"role": "user", "content": prompt},
    ]
    try:
        return llm_fn(msgs).strip()
    except Exception as exc:
        logger.warning("persona LLM failed: %s", exc)
        return "Cảm ơn, tôi đã hiểu rõ."


_DONE_PHRASES = ("tôi đã hiểu rõ", "đã hiểu rõ", "cảm ơn, tôi đã hiểu")


def _is_done(msg: str) -> bool:
    low = (msg or "").lower()
    return any(p in low for p in _DONE_PHRASES)


def simulate_conversation(
    scenario,
    agent_runner: Callable,
    persona: UserPersona,
    *,
    max_turns: int = 8,
    llm_fn: Optional[Callable] = None,
) -> ConversationLog:
    """Run a multi-turn conversation between persona and agent.

    ``agent_runner(history, user_msg) -> dict`` must return at least
    ``{"response": str}``; optionally ``route``, ``tool_calls``, ``latency_ms``.
    ``llm_fn`` defaults to ``brain.groq_chat_complete``.
    """
    if llm_fn is None:
        from brain import groq_chat_complete as llm_fn  # lazy

    history: List[dict] = []
    turns: List[Turn] = []
    first_msg = scenario.initial_state or scenario.goal
    user_msg = first_msg
    reached = False

    for idx in range(max_turns):
        history.append({"role": "user", "content": user_msg})
        try:
            result = agent_runner(history[:-1], user_msg)
        except Exception as exc:
            logger.warning("agent_runner failed at turn %d: %s", idx, exc)
            turns.append(Turn(turn_idx=idx, user_msg=user_msg,
                              agent_reply=f"[agent_error: {exc}]", route=None))
            break
        reply = (result or {}).get("response", "")
        turns.append(Turn(
            turn_idx=idx, user_msg=user_msg, agent_reply=reply,
            route=(result or {}).get("route"),
            tool_calls=tuple((result or {}).get("tool_calls") or ()),
            latency_ms=float((result or {}).get("latency_ms", 0.0)),
        ))
        history.append({"role": "assistant", "content": reply})
        if _is_done(reply) or _is_done(user_msg):
            reached = True
            break
        user_msg = persona_message(persona, scenario.goal, reply, llm_fn)
        if _is_done(user_msg):
            reached = True
            turns.append(Turn(turn_idx=idx + 1, user_msg=user_msg,
                              agent_reply="", route=None))
            break
    else:
        pass

    truncated = not reached and len(turns) >= max_turns
    return ConversationLog(
        scenario_id=getattr(scenario, "scenario_id", "scenario"),
        persona=persona, turns=tuple(turns),
        reached_goal=reached, truncated=truncated,
    )


__all__ = [
    "UserPersona",
    "Turn",
    "ConversationLog",
    "build_persona",
    "persona_message",
    "simulate_conversation",
]