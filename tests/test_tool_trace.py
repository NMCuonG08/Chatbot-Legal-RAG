"""Per-tool-call trace + latency tests (criterion #7: observability through loop).

Marker: unit. Stubs ``trace.emit_step`` (lazy import inside ``_emit_tool_trace``)
so no DB/Redis needed. Validates that ``@track_tool_call`` emits a ``tool_call``
trace event with latency when ``agent_run_id`` is set, and stays silent outside
a graph run (run_id None).
"""
import json
import types

import pytest

pytestmark = pytest.mark.unit


def _install_trace_stub(monkeypatch, sink):
    """Make the lazy ``from trace import emit_step`` resolve to a stub."""
    import sys

    fake = types.ModuleType("trace")
    def _emit(run_id, thread_id, node, event_type, payload):
        sink.append({
            "run_id": run_id, "thread_id": thread_id, "node": node,
            "event_type": event_type, "payload": payload,
        })
    fake.emit_step = _emit
    monkeypatch.setitem(sys.modules, "trace", fake)


def test_track_tool_call_emits_trace_with_latency(monkeypatch):
    import agent_tool_tracking as att

    sink = []
    _install_trace_stub(monkeypatch, sink)

    @att.track_tool_call
    def add(a, b):
        return json.dumps({"sum": a + b})

    acc = []
    tok = att.agent_tool_calls.set(acc)
    rid = att.agent_run_id.set("run-abc")
    tid = att.agent_thread_id.set("thread-xyz")
    try:
        out = add(2, 3)
    finally:
        att.agent_tool_calls.reset(tok)
        att.agent_run_id.reset(rid)
        att.agent_thread_id.reset(tid)

    assert json.loads(out)["sum"] == 5
    assert len(acc) == 1
    assert acc[0]["tool_name"] == "add"
    assert acc[0]["status"] == "success"
    # trace emitted
    assert len(sink) == 1
    ev = sink[0]
    assert ev["run_id"] == "run-abc"
    assert ev["thread_id"] == "thread-xyz"
    assert ev["node"] == "agent_tools"
    assert ev["event_type"] == "tool_call"
    assert ev["payload"]["tool_name"] == "add"
    assert ev["payload"]["status"] == "success"
    assert isinstance(ev["payload"]["latency_ms"], float)
    assert ev["payload"]["latency_ms"] >= 0.0


def test_track_tool_call_silent_without_run_id(monkeypatch):
    import agent_tool_tracking as att

    sink = []
    _install_trace_stub(monkeypatch, sink)

    @att.track_tool_call
    def ok():
        return json.dumps({"ok": 1})

    acc = []
    tok = att.agent_tool_calls.set(acc)
    # run_id left at default None => no trace emit
    try:
        ok()
    finally:
        att.agent_tool_calls.reset(tok)

    assert len(acc) == 1  # accumulator still records
    assert sink == []     # but no trace event


def test_track_tool_call_emits_on_error_status(monkeypatch):
    import agent_tool_tracking as att

    sink = []
    _install_trace_stub(monkeypatch, sink)

    @att.track_tool_call
    def boom():
        raise ValueError("bad arg")

    acc = []
    tok = att.agent_tool_calls.set(acc)
    rid = att.agent_run_id.set("run-err")
    try:
        with pytest.raises(ValueError):
            boom()
    finally:
        att.agent_tool_calls.reset(tok)
        att.agent_run_id.reset(rid)

    assert acc[0]["status"] == "error"
    assert "ValueError" in acc[0]["error"]
    assert sink[0]["payload"]["status"] == "error"
    assert "latency_ms" in sink[0]["payload"]


def test_track_tool_call_no_acc_silent(monkeypatch):
    """When agent_tool_calls is None (casual call, no run), record+trace skip."""
    import agent_tool_tracking as att

    sink = []
    _install_trace_stub(monkeypatch, sink)

    @att.track_tool_call
    def casual():
        return json.dumps({"x": 1})

    # default contextvar None => skip recording + trace
    assert casual() == json.dumps({"x": 1})
    assert sink == []