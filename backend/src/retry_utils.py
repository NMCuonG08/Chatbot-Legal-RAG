"""Retry/backoff helpers for transient network + HTTP errors.

Kept deliberately tiny: ``with_retry`` (sync) + ``awith_retry`` (async)
decorators that retry only on errors deemed transient by
``is_retryable_exception``. Non-retryable exceptions propagate immediately so
callers keep their existing fallback behavior.

Used to harden the hot LLM/embedding call paths against Groq 429 rate limits
and transient 5xx/timeout responses without changing the failure contract of
the wrapped functions (they still surface the same exception after exhausting
retries, so outer ``try/except`` fallbacks remain in control).
"""

import asyncio
import logging
import time
from functools import wraps
from typing import Callable

logger = logging.getLogger(__name__)

# Retryable HTTP status codes: rate limit + transient server errors.
RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


def build_transient_exceptions() -> tuple:
    """Return a tuple of exception classes worth a task-level retry.

    Covers HTTP (status_code attr), DB (SQLAlchemy interface/operational), cache
    (Redis connection/timeout), and generic OS network errors. Non-retryable
    logic errors (ValueError, KeyError, AttributeError, ...) are excluded so a
    bug never gets retried. Optional deps are imported lazily so this module
    never hard-fails when a client lib is absent.
    """
    excs: list = [ConnectionError, TimeoutError, OSError]
    try:
        import requests
        excs += [requests.ConnectionError, requests.Timeout]
    except ImportError:  # pragma: no cover
        pass
    try:
        import redis.exceptions
        excs += [redis.exceptions.ConnectionError, redis.exceptions.TimeoutError]
    except ImportError:  # pragma: no cover
        pass
    try:
        import sqlalchemy.exc
        excs += [sqlalchemy.exc.OperationalError, sqlalchemy.exc.InterfaceError]
    except ImportError:  # pragma: no cover
        pass
    seen, out = set(), []
    for e in excs:
        if e not in seen:
            seen.add(e)
            out.append(e)
    return tuple(out)


def is_retryable_exception(exc: BaseException) -> bool:
    """Heuristic: True for transient network/HTTP errors worth retrying.

    Checks for ``requests`` connection/timeout exceptions and for SDK-specific
    rate-limit/server exceptions that expose a ``status_code`` attribute (Groq,
    OpenAI). Non-retryable errors (4xx auth/validation, value errors, etc.)
    return False so they propagate immediately.
    """
    try:
        import requests  # local import keeps the util dependency-optional
        if isinstance(exc, (requests.ConnectionError, requests.Timeout)):
            return True
    except ImportError:  # pragma: no cover - requests is a hard dep in practice
        pass

    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and status in RETRYABLE_STATUS_CODES:
        return True

    # Some SDKs nest the status on a ``response`` object.
    resp = getattr(exc, "response", None)
    if resp is not None:
        status = getattr(resp, "status_code", None)
        if isinstance(status, int) and status in RETRYABLE_STATUS_CODES:
            return True

    return False


def with_retry(max_attempts: int = 3, base_delay: float = 1.0, max_delay: float = 8.0) -> Callable:
    """Retry a sync callable on transient errors with exponential backoff.

    ``max_attempts`` total tries (1 + retries). Sleeps ``base_delay * 2^(n-1)``
    capped at ``max_delay`` between attempts. The final exception propagates so
    the caller's existing ``try/except`` fallback path stays in charge of the
    user-facing result.
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    if not is_retryable_exception(exc):
                        raise
                    attempt += 1
                    if attempt >= max_attempts:
                        logger.warning(
                            f"with_retry: {func.__name__} exhausted {max_attempts} attempts; "
                            f"giving up ({exc!r})"
                        )
                        raise
                    delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
                    logger.warning(
                        f"with_retry: {func.__name__} attempt {attempt + 1}/{max_attempts} "
                        f"failed transiently ({exc!r}); retrying in {delay:.1f}s"
                    )
                    time.sleep(delay)

        return wrapper

    return decorator


def awith_retry(max_attempts: int = 3, base_delay: float = 1.0, max_delay: float = 8.0) -> Callable:
    """Async twin of ``with_retry``. Retries an awaitable on transient errors.

    Same backoff policy (``base_delay * 2^(n-1)`` capped at ``max_delay``) but
    sleeps via ``asyncio.sleep`` so it never blocks the event loop. Final
    exception propagates so the caller's fallback path stays in control.

    NOTE: only applies to coroutine functions. Generator-based stream methods
    yield lazily, so the first ``return func(...)`` returns a generator object
    without executing the body — a decorator cannot observe mid-stream errors.
    Streams must stay on their own try/except fallback path.
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            attempt = 0
            while True:
                try:
                    return await func(*args, **kwargs)
                except Exception as exc:
                    if not is_retryable_exception(exc):
                        raise
                    attempt += 1
                    if attempt >= max_attempts:
                        logger.warning(
                            f"awith_retry: {func.__name__} exhausted {max_attempts} attempts; "
                            f"giving up ({exc!r})"
                        )
                        raise
                    delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
                    logger.warning(
                        f"awith_retry: {func.__name__} attempt {attempt + 1}/{max_attempts} "
                        f"failed transiently ({exc!r}); retrying in {delay:.1f}s"
                    )
                    await asyncio.sleep(delay)

        return wrapper

    return decorator