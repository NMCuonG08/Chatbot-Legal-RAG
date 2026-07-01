"""Phase E tests: ReAct tool_calls surfacing via the agent_tool_calls contextvar.

Verifies that @track_tool_call records into the contextvar accumulator only when
it is set, and that ai_agent_handle returns a (text, tool_calls) tuple carrying
whatever tools the agent invoked during the turn.
"""
import agent


def test_track_tool_call_noop_when_contextvar_unset():
    """When no accumulator is set (default None), the tool runs but records nothing."""
    @agent.track_tool_call
    def _stub(x):
        return f"out:{x}"
    # Default contextvar is None -> wrapper returns func result, no recording.
    assert agent.agent_tool_calls.get() is None
    assert _stub("a") == "out:a"


def test_track_tool_call_records_when_contextvar_set():
    @agent.track_tool_call
    def _stub(x):
        return f"out:{x}"
    token = agent.agent_tool_calls.set([])
    try:
        result = _stub("hi")
        assert result == "out:hi"
        acc = agent.agent_tool_calls.get()
        assert len(acc) == 1
        assert acc[0]["tool_name"] == "_stub"
        assert acc[0]["status"] == "success"
        assert acc[0]["args"] == {"x": "hi"}
    finally:
        agent.agent_tool_calls.reset(token)


def test_track_tool_call_records_error_status():
    @agent.track_tool_call
    def _boom():
        raise ValueError("nope")
    token = agent.agent_tool_calls.set([])
    try:
        try:
            _boom()
        except ValueError:
            pass
        acc = agent.agent_tool_calls.get()
        assert acc[0]["status"] == "error"
        assert "ValueError" in acc[0]["error"]
    finally:
        agent.agent_tool_calls.reset(token)


def test_ai_agent_handle_returns_tuple_and_surfaces_tool_calls(monkeypatch):
    """ai_agent_handle returns (text, tool_calls) with calls the agent made."""
    @agent.track_tool_call
    def _stub_tool(x):
        return f"out:{x}"

    class FakeAgent:
        async def arun(self, input=None, *a, **k):
            # Simulate the agent invoking a tracked tool during its turn.
            return _stub_tool("hi")

    captured = {"args": None}

    def fake_get_agent(user_id=None, conversation_id=None):
        captured["args"] = (user_id, conversation_id)
        return FakeAgent()

    monkeypatch.setattr(agent, "_get_ai_agent", fake_get_agent)

    res_text, tool_calls = agent.ai_agent_handle("question", "u1", "c1")

    assert res_text == "out:hi"
    assert captured["args"] == ("u1", "c1")  # per-conversation identity propagated
    assert len(tool_calls) == 1
    assert tool_calls[0]["tool_name"] == "_stub_tool"


def test_ai_agent_handle_no_agent_returns_tuple_with_empty_calls(monkeypatch):
    monkeypatch.setattr(agent, "_get_ai_agent", lambda *a, **k: None)
    res_text, tool_calls = agent.ai_agent_handle("q")
    assert isinstance(res_text, str)
    assert tool_calls == []


def test_ai_agent_handle_resets_contextvar_after_call(monkeypatch):
    """The contextvar must be restored to its prior value after the call."""
    @agent.track_tool_call
    def _stub_tool(x):
        return f"out:{x}"

    class FakeAgent:
        async def arun(self, input=None, *a, **k):
            return _stub_tool("hi")

    monkeypatch.setattr(agent, "_get_ai_agent", lambda *a, **k: FakeAgent())
    assert agent.agent_tool_calls.get() is None  # before
    agent.ai_agent_handle("q")
    assert agent.agent_tool_calls.get() is None  # after (reset)