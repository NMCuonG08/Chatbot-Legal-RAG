"""Graph loop-control helpers (senior: bound the loop, never hang).

Kept dependency-free so it is unit-testable without importing ``tasks`` (which
pulls the full agent/guardrail/langchain_groq chain). ``tasks.run_chat_graph``
uses ``_invoke_with_deadline`` + ``GraphRunTimeout`` from here.
"""
from __future__ import annotations

import concurrent.futures


class GraphRunTimeout(Exception):
    """Wall-clock deadline for one ``graph.invoke`` exceeded.

    Deliberately NOT a transient exception (not in
    ``retry_utils.is_retryable_exception``) so ``with_retry`` does not retry a
    hung graph — it propagates to the graceful-degrade path instead.
    """


def _invoke_with_deadline(graph, state, config, timeout_s: float):
    """Run ``graph.invoke`` with a wall-clock deadline.

    Python cannot forcibly kill a blocked thread, so on timeout we abandon the
    worker thread (it fails on its own per-LLM ``request_timeout``) and raise
    ``GraphRunTimeout`` so the caller unblocks and degrades gracefully.
    ``shutdown(wait=False, cancel_futures=True)`` avoids joining the hung thread.
    """
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    fut = executor.submit(graph.invoke, state, config)
    try:
        return fut.result(timeout=timeout_s)
    except concurrent.futures.TimeoutError as exc:
        raise GraphRunTimeout(
            f"graph.invoke exceeded wall-clock deadline {timeout_s}s"
        ) from exc
    finally:
        executor.shutdown(wait=False, cancel_futures=True)