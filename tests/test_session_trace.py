"""Session-trace replay tests (criterion #7: trace_id spans whole session).

Marker: unit. Uses sqlite in-memory so no MariaDB needed. Validates
``load_session_trace(thread_id)`` joins ``AgentStep`` to ``GraphRun`` by thread_id
and returns steps across all runs of a conversation in chronological order —
so a multi-turn conversation replays as one trace (run_id is per-message, but
thread_id = conversation_id is the session key).
"""
import datetime

import pytest

pytestmark = pytest.mark.unit


@pytest.fixture()
def db():
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from models import AgentStep, Base, GraphRun

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine, tables=[GraphRun.__table__, AgentStep.__table__])
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session, GraphRun, AgentStep
    session.close()


def test_load_session_trace_orders_steps_across_runs(db):
    session, GraphRun, AgentStep = db

    # Two runs in the SAME conversation (thread_id), one run in a DIFFERENT thread.
    r1 = GraphRun(id="r1", thread_id="conv-A", question="q1", user_id="u")
    r2 = GraphRun(id="r2", thread_id="conv-A", question="q2", user_id="u")
    r3 = GraphRun(id="r3", thread_id="conv-B", question="q3", user_id="u")
    r1.started_at = datetime.datetime(2026, 1, 1, 9, 0, 0)
    r2.started_at = datetime.datetime(2026, 1, 1, 9, 5, 0)
    r3.started_at = datetime.datetime(2026, 1, 1, 9, 2, 0)
    session.add_all([r1, r2, r3])

    # r1: two steps; r2: one step; r3 (other thread): one step that must NOT appear.
    session.add(AgentStep(run_id="r1", node="route", step_index=1, event_type="node_start", payload={}))
    session.add(AgentStep(run_id="r1", node="generate", step_index=2, event_type="node_end", payload={}))
    session.add(AgentStep(run_id="r2", node="route", step_index=1, event_type="run_start", payload={}))
    session.add(AgentStep(run_id="r3", node="route", step_index=1, event_type="node_start", payload={}))
    session.commit()

    from models import load_session_trace

    steps = load_session_trace("conv-A", db=session)
    # Ordered by run started_at then step_index: r1 step1, r1 step2, r2 step1.
    assert [s.run_id for s in steps] == ["r1", "r1", "r2"]
    assert all(s.run_id in {"r1", "r2"} for s in steps)


def test_load_session_trace_empty_for_unknown_thread(db):
    session, _GraphRun, _AgentStep = db
    from models import load_session_trace
    assert load_session_trace("no-such-thread", db=session) == []


def test_load_session_trace_isolates_threads(db):
    session, GraphRun, AgentStep = db
    r = GraphRun(id="rx", thread_id="conv-X", question="q", user_id="u")
    r.started_at = datetime.datetime(2026, 1, 1, 9, 0, 0)
    session.add(r)
    session.add(AgentStep(run_id="rx", node="n", step_index=1, event_type="node_start", payload={}))
    session.commit()

    from models import load_session_trace
    steps_x = load_session_trace("conv-X", db=session)
    steps_other = load_session_trace("conv-A", db=session)
    assert len(steps_x) == 1
    assert steps_other == []