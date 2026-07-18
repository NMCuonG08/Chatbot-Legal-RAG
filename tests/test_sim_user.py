"""Unit tests for sim_user: persona prompt, terminate on goal, max_turns."""
from evaluation.sim_user import (
    ConversationLog,
    Turn,
    UserPersona,
    build_persona,
    persona_message,
    simulate_conversation,
)


class _Scenario:
    def __init__(self, goal, initial_state=""):
        self.scenario_id = "scn-1"
        self.goal = goal
        self.initial_state = initial_state


def test_build_persona_known_role():
    p = build_persona("junior_lawyer", name="ln")
    assert p.role == "junior_lawyer"
    assert p.name == "ln"
    assert "luật sư" in p.system_prompt


def test_build_persona_unknown_role_falls_back():
    p = build_persona("astronaut")
    assert p.role == "layperson"


def test_persona_message_uses_llm_fn():
    p = build_persona("layperson")
    captured = {}

    def llm_fn(messages):
        captured["messages"] = messages
        return "  còn giấy tờ gì nữa?  "

    out = persona_message(p, "mục tiêu X", "agent reply", llm_fn)
    assert out == "còn giấy tờ gì nữa?"
    assert any("mục tiêu" in m["content"] for m in captured["messages"])


def test_persona_message_llm_failure_returns_done():
    p = build_persona("layperson")

    def llm_fn(messages):
        raise RuntimeError("boom")

    out = persona_message(p, "goal", "reply", llm_fn)
    assert "đã hiểu rõ" in out


def test_simulate_terminates_on_agent_done():
    sc = _Scenario("đăng ký doanh nghiệp", initial_state="tôi muốn mở công ty")
    p = build_persona("layperson")
    calls = {"n": 0}

    def agent_runner(history, user_msg):
        calls["n"] += 1
        return {"response": "Bạn cần nộp hồ sơ. Cảm ơn, tôi đã hiểu rõ.",
                "route": "legal_rag", "tool_calls": [], "latency_ms": 12.0}

    log = simulate_conversation(sc, agent_runner, p, max_turns=8,
                                llm_fn=lambda m: "tiếp đi")
    assert log.reached_goal is True
    assert log.truncated is False
    assert len(log.turns) >= 1
    assert calls["n"] == 1


def test_simulate_terminates_on_persona_done():
    sc = _Scenario("giấy phép", initial_state="cần giấy phép gì?")
    p = build_persona("business")

    def agent_runner(history, user_msg):
        return {"response": "Bạn cần giấy phép A.", "route": "legal_rag"}

    def llm_fn(messages):
        return "Cảm ơn, tôi đã hiểu rõ."

    log = simulate_conversation(sc, agent_runner, p, max_turns=8, llm_fn=llm_fn)
    assert log.reached_goal is True
    assert any("đã hiểu rõ" in t.user_msg for t in log.turns)


def test_simulate_truncates_at_max_turns():
    sc = _Scenario("meta", initial_state="hi")
    p = build_persona("layperson")

    def agent_runner(history, user_msg):
        return {"response": "câu trả lời chung chung không kết thúc"}

    log = simulate_conversation(sc, agent_runner, p, max_turns=3,
                                llm_fn=lambda m: "tiếp hỏi")
    assert log.reached_goal is False
    assert log.truncated is True
    assert len(log.turns) == 3


def test_simulate_agent_error_recorded():
    sc = _Scenario("meta", initial_state="hi")
    p = build_persona("layperson")

    def agent_runner(history, user_msg):
        raise RuntimeError("agent boom")

    log = simulate_conversation(sc, agent_runner, p, max_turns=5,
                                llm_fn=lambda m: "x")
    assert any("agent_error" in t.agent_reply for t in log.turns)
    assert log.reached_goal is False