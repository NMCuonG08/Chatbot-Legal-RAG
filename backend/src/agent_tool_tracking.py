"""Per-run tool-call tracking for the ReAct agent.

``agent_tool_calls`` is a contextvar accumulating the list of tool-call records
for the current run. ``track_tool_call`` is a decorator that wraps a tool fn so
every call appends a record (name, args, status, error, truncated result).

Extracted from ``agent.py`` so the tool-wrapper module can import the decorator
without a circular import (agent.py imports the wrappers which import tracking).
"""

import contextvars
import functools
import inspect
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Each run resets this to ``[]`` via ``agent_tool_calls.set([])`` in
# ``ai_agent_handle``. ``@track_tool_call`` only records when a list is present
# (default ``None`` => no recording, so casual imports/calls stay silent).
agent_tool_calls: "contextvars.ContextVar[Optional[list]]" = contextvars.ContextVar(
    "agent_tool_calls", default=None
)


def track_tool_call(func):
    """Decorator: record each call of a tool fn into ``agent_tool_calls``.

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

    return wrapper