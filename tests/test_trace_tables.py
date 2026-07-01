"""Phase C unit tests for trace persistence (MySQL + Redis pub/sub).

Unit-level: monkeypatch the DB helpers and Redis publish so no live MySQL/Redis
is required. Verifies the emit_* API records the right events, the per-run step
counter is monotonic, and failures are swallowed (never raise into a graph node).
"""
import tasks
import trace


def test_emit_step_persists_and_publishes(monkeypatch):
    captured_steps = []
    published = []

    monkeypatch.setattr(trace, "save_agent_step",
                        lambda run_id, node, idx, et, payload: captured_steps.append((run_id, node, idx, et, payload)))
    monkeypatch.setattr(trace, "_publish",
                        lambda run_id, thread_id, node, idx, et, payload: published.append((node, et, idx)))

    trace.reset_step_index("run-1")
    trace.emit_step("run-1", "thread-1", "retrieve", "node_end", {"doc_count": 3})
    trace.emit_step("run-1", "thread-1", "generate", "node_end", {"answer_len": 10})

    assert [s[2] for s in captured_steps] == [1, 2]  # monotonic step_index
    assert [p[0] for p in published] == ["retrieve", "generate"]
    assert captured_steps[0][3] == "node_end"


def test_emit_run_start_resets_counter_and_persists_run(monkeypatch):
    runs = []
    monkeypatch.setattr(trace, "save_graph_run",
                        lambda run_id, thread_id, user_id, question, status="running": runs.append((run_id, status, question)))
    monkeypatch.setattr(trace, "save_agent_step", lambda *a, **k: None)
    monkeypatch.setattr(trace, "_publish", lambda *a, **k: None)

    # Pre-seed counter to prove reset.
    trace._step_counter["run-2"] = 99
    trace.emit_run_start("run-2", "thread-2", "u1", "hello?")
    assert trace._step_counter["run-2"] == 1  # run_start emitted one step after reset
    assert runs[0][0] == "run-2"
    assert runs[0][1] == "running"


def test_emit_run_end_updates_graph_run(monkeypatch):
    updates = []
    monkeypatch.setattr(trace, "update_graph_run",
                        lambda run_id, **kw: updates.append((run_id, kw)))
    monkeypatch.setattr(trace, "save_agent_step", lambda *a, **k: None)
    monkeypatch.setattr(trace, "_publish", lambda *a, **k: None)

    trace.emit_run_end("run-3", "thread-3", status="completed",
                       final_response="ans", route="legal_rag", reflection_count=1)

    assert updates[0][0] == "run-3"
    assert updates[0][1]["status"] == "completed"
    assert updates[0][1]["final_response"] == "ans"
    assert updates[0][1]["route"] == "legal_rag"
    assert updates[0][1]["reflection_count"] == 1


def test_emit_step_swallows_db_failure(monkeypatch):
    """A trace failure must never escape into a graph node."""
    monkeypatch.setattr(trace, "save_agent_step", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("db down")))
    monkeypatch.setattr(trace, "_publish", lambda *a, **k: None)
    trace.reset_step_index("run-4")
    # Should not raise.
    trace.emit_step("run-4", "thread-4", "retrieve", "node_end", {})


def test_emit_run_start_swallows_failure(monkeypatch):
    monkeypatch.setattr(trace, "save_graph_run", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("db down")))
    monkeypatch.setattr(trace, "save_agent_step", lambda *a, **k: None)
    monkeypatch.setattr(trace, "_publish", lambda *a, **k: None)
    trace.emit_run_start("run-5", "thread-5", None, "q")  # no raise


def test_trace_node_end_noop_without_run_id():
    """Legacy callers (run_id absent) must not emit anything."""
    called = {"n": 0}
    original = tasks.emit_step

    def boom(*a, **k):
        called["n"] += 1

    tasks.emit_step = boom
    try:
        state = {"thread_id": "t"}
        # Should return early because run_id missing.
        tasks._trace_node_end(state, "route", {"route": "legal_rag"})
        assert called["n"] == 0
    finally:
        tasks.emit_step = original


def test_ensure_database_schema_creates_trace_tables(monkeypatch):
    """ensure_database_schema must invoke create_all, which covers GraphRun/AgentStep."""
    called = {"n": 0}

    def fake_create_all(bind=None):
        called["n"] += 1

    import models
    monkeypatch.setattr(models.Base.metadata, "create_all", fake_create_all)
    models.ensure_database_schema()
    assert called["n"] == 1