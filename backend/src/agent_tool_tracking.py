"""Per-run tool-call tracking for the ReAct agent.

``agent_tool_calls`` is a contextvar accumulating the list of tool-call records
for the current run. ``track_tool_call`` is a decorator that wraps a tool fn so
every call appends a record (name, args, status, error, truncated result) AND
emits a per-tool-call trace event with latency (bridge loop -> harness: clean
per-step data for eval slicing).

It ALSO enforces the empty-result no-retry guard: a ReAct agent that calls a
search tool, gets an empty result, then calls the SAME tool with the SAME args
again gets ZERO new information — but burns another LLM round trip + reasoning
tokens, compounding to 60-90s latency. The decorator blocks the repeat call
with a sentinel that forces the LLM to reformulate (broaden / drop a clause /
synonym) or stop and answer "không tìm thấy". After ``AGENT_MAX_EMPTY_STREAK``
consecutive empties, the guard hard-stops further calls. This is the single
biggest fix for the 75s lũy kế.

Extracted from ``agent.py`` so the tool-wrapper module can import the decorator
without a circular import (agent.py imports the wrappers which import tracking).
"""

import contextvars
import functools
import hashlib
import inspect
import json
import logging
import time
from typing import Any, Optional

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

# Per-run accumulator for SOURCE chunks surfaced by retrieval tools during the
# current agent run. Populated by ``record_agent_source`` inside the retrieval
# tool wrappers (article_lookup / cross_reference / verify_citation /
# precedent_lookup / recall_legal_graph). ``generate_agent_answer`` resets this
# to ``[]`` at run start and returns the accumulated sources so the graph can
# hand them to ``verify_answer_node`` — closing the gap where the agent route
# used to carry ``sources=[]`` and skip citation groundedness entirely.
# Default ``None`` => no recording (casual calls outside a graph run stay silent).
agent_sources: "contextvars.ContextVar[Optional[list]]" = contextvars.ContextVar(
    "agent_sources", default=None
)

# ---- Empty-result no-retry guard (senior: kill the 75s lũy kế) ----
# Consecutive-empty counter for the current run. Incremented by
# ``mark_tool_empty`` when a tool returns an empty result; reset to 0 on a
# non-empty result. When it reaches ``config.AGENT_MAX_EMPTY_STREAK`` the
# decorator blocks further tool calls with a stop sentinel.
agent_empty_streak: "contextvars.ContextVar[int]" = contextvars.ContextVar(
    "agent_empty_streak", default=0
)
# Per-tool last-call signature: ``{tool_name: {"args_hash": str, "empty": bool}}``.
# Lets the decorator detect an identical-args repeat of an empty call and block
# it (zero information gain -> zero point retrying). Reset to ``{}`` at run start.
agent_prev_tool_args: "contextvars.ContextVar[Optional[dict]]" = contextvars.ContextVar(
    "agent_prev_tool_args", default=None
)

# Sentinel returned to the LLM when a repeat empty call is blocked. Tells the
# agent explicitly to reformulate OR stop — not a silent empty (which the LLM
# would just retry again, perpetuating the lũy kế).
EMPTY_BLOCK_SENTINEL = (
    "[EMPTY_REPEAT_BLOCKED] Tool vừa trả kết quả rỗng với tham số y hệt lần trước — "
    "gọi lại không mang thông tin mới. BẮT BUỘC: đổi query (mở rộng / bỏ điều kiện "
    "khoản-điểm / dùng từ đồng nghĩa) HOẶC dừng và trả lời user là \"không tìm thấy "
    "tài liệu phù hợp\". Không gọi lại tool với cùng tham số."
)
EMPTY_STREAK_SENTINEL = (
    "[EMPTY_STREAK_STOP] Đã gọi tool trả rỗng nhiều lần liên tiếp — tiếp tục retry "
    "vô ích. DỪNG: tổng hợp câu trả lời \"không tìm thấy tài liệu pháp luật phù hợp "
    "trong kho dữ liệu, gợi ý đặt câu hỏi cụ thể hơn (ghi rõ tên luật + số điều)\"."
)


def record_agent_source(source: dict) -> None:
    """Append a source chunk to the per-run ``agent_sources`` accumulator.

    No-op when no run is active (contextvar is ``None``). De-duplicates by
    ``doc_id``/``chunk_id``/``id`` when present so a re-called tool does not
    double-count the same chunk. Best-effort: never raises.
    """
    acc = agent_sources.get()
    if acc is None or not isinstance(source, dict):
        return
    key = source.get("doc_id") or source.get("chunk_id") or source.get("id")
    if key:
        for existing in acc:
            if existing.get("doc_id") == key or existing.get("chunk_id") == key or existing.get("id") == key:
                return
    acc.append(source)


# ---- Empty-result no-retry guard helpers ----

# Result keys whose list value being empty means the tool found nothing.
# Covers every retrieval tool's JSON shape (article_lookup/precedent/cross_ref/
# graph/verify_citation). Conservative: only treat as empty when a known
# result-list key exists AND is empty/None — unknown shapes fall through to
# "non-empty" so we never wrongly suppress a real result.
_EMPTY_RESULT_KEYS = (
    "matches", "precedents", "referencing_texts", "relations",
    "articles", "results", "sources", "facts",
)


def _args_hash(args_dict: dict) -> str:
    """Stable hash of a tool's bound args so identical repeat calls are
    detectable. JSON-sort + sha1 (handles un-orderable types via default=str)."""
    try:
        raw = json.dumps(args_dict, sort_keys=True, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        raw = repr(sorted(args_dict.items()))
    return hashlib.sha1(raw.encode("utf-8", "replace")).hexdigest()


def _is_empty_result(res: Any) -> bool:
    """True if a tool result carries no retrievable info.

    Detects: JSON with an ``error`` key (failed lookup), or a known result-list
    key present but empty/None. Non-JSON strings (e.g. free-text web search
    fallbacks) are treated as non-empty — only structured-empty counts, so we
    never block a tool that legitimately returned a text answer.
    """
    if res is None:
        return True
    if not isinstance(res, str):
        return False
    try:
        parsed = json.loads(res)
    except Exception:
        return False  # free-text result -> not empty
    if not isinstance(parsed, dict):
        return False
    if "error" in parsed and parsed["error"]:
        return True
    for key in _EMPTY_RESULT_KEYS:
        if key in parsed:
            val = parsed[key]
            if val is None or (isinstance(val, (list, dict, str)) and len(val) == 0):
                return True
    return False


def mark_tool_empty(tool_name: str, args_hash: str, is_empty: bool) -> None:
    """Record this tool call's empty/non-empty outcome + advance/reset streak.

    Called by ``track_tool_call`` AFTER the tool runs. On non-empty the streak
    resets to 0 (a success breaks the empty run). On empty the streak advances
    and the per-tool last-args signature is stored so the NEXT call can be
    compared for an identical-args repeat. No-op outside a run (contextvar None).
    """
    prev = agent_prev_tool_args.get()
    if prev is None:
        return
    prev[tool_name] = {"args_hash": args_hash, "empty": is_empty}
    streak = agent_empty_streak.get()
    streak = (streak + 1) if is_empty else 0
    agent_empty_streak.set(streak)


def should_block_repeat(tool_name: str, args_hash: str) -> Optional[str]:
    """Return a sentinel string to short-circuit the call, or None to proceed.

    Blocks when EITHER:
      - the same tool was just called with the same args AND returned empty
        (identical retry = zero info gain), returning EMPTY_BLOCK_SENTINEL; OR
      - the run-wide empty streak has reached AGENT_MAX_EMPTY_STREAK
        (hard stop), returning EMPTY_STREAK_SENTINEL.

    The sentinel is returned to the LLM as the tool result, steering it to
    reformulate or stop instead of burning another round trip. No-op outside a
    run (contextvar None -> None, call proceeds).
    """
    prev = agent_prev_tool_args.get()
    if prev is None:
        return None
    streak = agent_empty_streak.get()
    try:
        import config as _cfg
        max_streak = getattr(_cfg, "AGENT_MAX_EMPTY_STREAK", 2)
    except Exception:
        max_streak = 2
    if streak >= max_streak:
        return EMPTY_STREAK_SENTINEL
    last = prev.get(tool_name)
    if last and last.get("empty") and last.get("args_hash") == args_hash:
        return EMPTY_BLOCK_SENTINEL
    return None


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
        args_dict = dict(bound.arguments)
        a_hash = _args_hash(args_dict)

        # Empty-result no-retry guard (BEFORE the call): block an identical-args
        # repeat of an empty result, or a call after the empty streak hit the cap.
        # The sentinel is recorded as a tool-call result so the trace + agent_tool_calls
        # log shows WHY the call was short-circuited (auditable), then returned to
        # the LLM to steer it toward reformulation/stop.
        block = should_block_repeat(func.__name__, a_hash)
        if block:
            call_record = {
                "tool_name": func.__name__,
                "args": args_dict,
                "status": "blocked_empty_retry",
                "error": None,
                "result": block[:1000],
            }
            acc.append(call_record)
            _emit_tool_trace(call_record, 0.0)
            logger.info(
                "[TOOL-GUARD] blocked %s (args-hash=%s streak=%s) — sentinel returned",
                func.__name__, a_hash[:8], agent_empty_streak.get(),
            )
            return block

        call_record = {
            "tool_name": func.__name__,
            "args": args_dict,
            "status": "success",
            "error": None,
            "result": None,
        }
        acc.append(call_record)

        t0 = time.perf_counter()
        try:
            res = func(*args, **kwargs)
            call_record["result"] = str(res)[:1000]
            is_failed = False
            if isinstance(res, str):
                try:
                    parsed = json.loads(res)
                    if "error" in parsed:
                        call_record["status"] = "failed"
                        call_record["error"] = parsed["error"]
                        is_failed = True
                except Exception:
                    pass
            # Empty-result tracking (AFTER the call): advance/reset streak +
            # store this call's signature so the next call can be diffed.
            mark_tool_empty(func.__name__, a_hash, _is_empty_result(res) or is_failed)
            return res
        except Exception as exc:
            call_record["status"] = "error"
            call_record["error"] = f"{type(exc).__name__}: {exc}"
            # An exception is not an "empty result" — don't advance the empty
            # streak (the error path is separate from the no-info-found path).
            raise exc
        finally:
            _emit_tool_trace(call_record, (time.perf_counter() - t0) * 1000.0)

    return wrapper