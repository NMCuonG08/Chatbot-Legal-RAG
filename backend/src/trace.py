"""Trace persistence for LangGraph runs (Phase C).

Best-effort: every emit_* call swallows exceptions so a trace failure never
breaks the chat flow. Persists AgentStep rows to MySQL and publishes the same
event to a Redis pub/sub channel for live SSE streaming (Phase F).
"""
import json
import logging
import threading

import redis

from database import settings
from models import save_agent_step, save_graph_run, update_graph_run

logger = logging.getLogger(__name__)

_pubsub_client = None
_pubsub_lock = threading.Lock()
_step_counter: dict[str, int] = {}
_step_lock = threading.Lock()


def get_redis_pubsub_client():
    """Singleton Redis client for trace pub/sub (decode_responses for text frames)."""
    global _pubsub_client
    with _pubsub_lock:
        if _pubsub_client is None:
            _pubsub_client = redis.from_url(settings.trace_redis_url, decode_responses=True)
        return _pubsub_client


def _next_step_index(run_id: str) -> int:
    with _step_lock:
        n = _step_counter.get(run_id, 0) + 1
        _step_counter[run_id] = n
        return n


def reset_step_index(run_id: str) -> None:
    with _step_lock:
        _step_counter.pop(run_id, None)


def _publish(run_id: str, thread_id: str, node: str, step_index: int,
             event_type: str, payload) -> None:
    try:
        msg = json.dumps(
            {
                "run_id": run_id,
                "thread_id": thread_id,
                "node": node,
                "step_index": step_index,
                "event_type": event_type,
                "payload": payload,
            },
            ensure_ascii=False,
            default=str,
        )
        get_redis_pubsub_client().publish(settings.trace_redis_channel, msg)
    except Exception as e:
        logger.warning(f"trace publish failed: {e}")


def _bridge(fn_name: str, **kwargs) -> None:
    """Mirror a trace event to the OTel bridge if enabled. Never raises."""
    try:
        from config import OTEL_BRIDGE_ENABLED
        if not OTEL_BRIDGE_ENABLED:
            return
        from evaluation import otel_bridge
        getattr(otel_bridge, fn_name)(**kwargs)
    except Exception as e:
        logger.debug(f"otel bridge {fn_name} skipped: {e}")


def emit_step(run_id: str, thread_id: str, node: str, event_type: str, payload) -> None:
    """Record one trace event: persist to agent_steps + publish to Redis channel."""
    try:
        idx = _next_step_index(run_id)
        save_agent_step(run_id, node, idx, event_type, payload)
        _publish(run_id, thread_id, node, idx, event_type, payload)
    except Exception as e:
        logger.warning(f"trace emit_step failed ({node}/{event_type}): {e}")
    _bridge("bridge_emit_step", run_id=run_id, thread_id=thread_id,
            node=node, event_type=event_type, payload=payload)


def emit_run_start(run_id: str, thread_id: str, user_id: str | None, question: str) -> None:
    try:
        reset_step_index(run_id)
        save_graph_run(run_id, thread_id, user_id, question, status="running")
        emit_step(run_id, thread_id, "__root__", "run_start", {"question": question})
    except Exception as e:
        logger.warning(f"trace emit_run_start failed: {e}")
    _bridge("bridge_emit_run_start", run_id=run_id, thread_id=thread_id,
            user_id=user_id, question=question)


def emit_run_end(run_id: str, thread_id: str, *, status: str = "completed",
                 final_response: str | None = None, route: str | None = None,
                 reflection_count: int | None = None, tool_calls_json=None) -> None:
    try:
        update_graph_run(
            run_id,
            status=status,
            final_response=final_response,
            route=route,
            reflection_count=reflection_count,
            tool_calls_json=tool_calls_json,
        )
        emit_step(run_id, thread_id, "__root__", "run_end", {"status": status})
    except Exception as e:
        logger.warning(f"trace emit_run_end failed: {e}")
    _bridge("bridge_emit_run_end", run_id=run_id, status=status, route=route)