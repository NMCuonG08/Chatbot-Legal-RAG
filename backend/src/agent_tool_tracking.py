"""Per-run tool-call tracking for the ReAct agent.

``agent_tool_calls`` is a contextvar accumulating the list of tool-call records
for the current run. ``track_tool_call`` is a decorator that wraps a tool fn so
every call appends a record (name, args, status, error, truncated result) AND
emits a per-tool-call trace event with latency (bridge loop -> harness: clean
per-step data for eval slicing).

Extracted from ``agent.py`` so the tool-wrapper module can import the decorator
without a circular import (agent.py imports the wrappers which import tracking).
"""

import contextvars
import functools
import inspect
import json
import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

# Each run resets this to ``[]`` via ``agent_tool_calls.set([])`` in
# ``ai_agent_handle``. ``@track_tool_call`` only records when a list is present
# (default ``None`` => no recording, so casual imports/calls stay silent).
agent_tool_calls: "contextvars.ContextVar[Optional[list]]" = contextvars.ContextVar(
    "agent_tool_calls", default=None
)

# Per-run user_id so tools (e.g. recall_user_memory) can scope retrieval to the
# current user without the agent having to pass it as an explicit argument.
# Set in ai_agent_handle alongside agent_tool_calls.
agent_user_id: "contextvars.ContextVar[Optional[str]]" = contextvars.ContextVar(
    "agent_user_id", default=None
)

# Per-run trace identity so @track_tool_call can emit a per-tool-call trace
# event (event_type="tool_call") with latency. Set in ai_agent_handle from the
# graph state (run_id, thread_id). Default None => no trace emit (silent for
# casual calls outside a graph run).
agent_run_id: "contextvars.ContextVar[Optional[str]]" = contextvars.ContextVar(
    "agent_run_id", default=None
)
agent_thread_id: "contextvars.ContextVar[Optional[str]]" = contextvars.ContextVar(
    "agent_thread_id", default=None
)


def _emit_tool_trace(call_record: dict, latency_ms: float) -> None:
    """Best-effort per-tool-call trace event. Never raises (trace failure must
    not break the chat flow). Only emits when run_id is set (inside a graph run).
    """
    run_id = agent_run_id.get()
    if run_id is None:
        return
    try:
        from trace import emit_step
        emit_step(
            run_id,
            agent_thread_id.get() or "",
            "agent_tools",
            "tool_call",
            {
                "tool_name": call_record["tool_name"],
                "status": call_record["status"],
                "error": call_record["error"],
                "latency_ms": round(latency_ms, 2),
            },
        )
    except Exception as exc:  # pragma: no cover - trace is best-effort
        logger.debug(f"tool-call trace emit skipped: {exc}")


def track_tool_call(func):
    """Decorator: record each call of a tool fn into ``agent_tool_calls`` and
    emit a per-tool-call trace event with latency.

    On exception the record is marked ``status="error"`` and the exception is
    re-raised. On a JSON result containing an ``error`` key the record is marked
    ``status="failed"`` (but the value is still returned to the caller).
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        acc = agent_tool_calls.get()
        if acc is None:
            return func(*args, **kwargs)

        sig = inspect.signature(func)
        bound = sig.bind_partial(*args, **kwargs)
        bound.apply_defaults()

        call_record = {
            "tool_name": func.__name__,
            "args": dict(bound.arguments),
            "status": "success",
            "error": None,
            "result": None,
        }
        acc.append(call_record)

        t0 = time.perf_counter()
        try:
            res = func(*args, **kwargs)
            call_record["result"] = str(res)[:1000]
            if isinstance(res, str):
                try:
                    parsed = json.loads(res)
                    if "error" in parsed:
                        call_record["status"] = "failed"
                        call_record["error"] = parsed["error"]
                except Exception:
                    pass
            return res
        except Exception as exc:
            call_record["status"] = "error"
            call_record["error"] = f"{type(exc).__name__}: {exc}"
            raise exc
        finally:
            _emit_tool_trace(call_record, (time.perf_counter() - t0) * 1000.0)

    return wrapper