"""OpenTelemetry bridge for trace.py.

Adds an OTel span stream alongside the existing MySQL+Redis trace. The bridge is
a subscriber, NOT a replacement: ``trace.py`` still owns persistence. When
``OTEL_BRIDGE_ENABLED`` is false (default) or the OTLP endpoint is unset, every
call is a no-op. Every bridge call is fire-and-forget (try/except) so an OTel
failure never breaks the trace path.

Public surface:
- ``setup_otel``, ``shutdown_otel``, ``span``, ``bridge_emit_step``,
  ``bridge_emit_run_start``, ``bridge_emit_run_end``.
"""
from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

logger = logging.getLogger(__name__)

_ENABLED = (
    os.environ.get("OTEL_BRIDGE_ENABLED", "false").lower() == "true"
)
_OTLP_ENDPOINT = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT") or None
_SERVICE_NAME = os.environ.get("OTEL_SERVICE_NAME", "legal-chatbot")

_tracer = None
_provider = None
_setup_done = False


def _ensure_setup() -> Optional[Any]:
    """Lazily build the tracer once. Returns None if disabled/unavailable."""
    global _tracer, _provider, _setup_done
    if not _ENABLED or not _OTLP_ENDPOINT:
        return None
    if _setup_done:
        return _tracer
    _setup_done = True
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )

        resource = Resource.create({"service.name": _SERVICE_NAME})
        provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter(endpoint=_OTLP_ENDPOINT)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        _provider = provider
        _tracer = trace.get_tracer(__name__)
        logger.info("OTel bridge enabled -> %s", _OTLP_ENDPOINT)
        return _tracer
    except Exception as exc:
        logger.warning("OTel setup failed (bridge disabled): %s", exc)
        _tracer = None
        return None


def setup_otel(service_name: str = _SERVICE_NAME,
               otlp_endpoint: Optional[str] = None) -> bool:
    """Explicit setup (idempotent). Returns True if the tracer is live."""
    global _SERVICE_NAME, _OTLP_ENDPOINT, _ENABLED, _setup_done
    if otlp_endpoint:
        _OTLP_ENDPOINT = otlp_endpoint
        _ENABLED = True
    if service_name:
        _SERVICE_NAME = service_name
    _setup_done = False  # force re-init with new config
    return _ensure_setup() is not None


def shutdown_otel() -> None:
    """Gracefully shutdown the provider if one was built."""
    global _provider, _tracer, _setup_done
    if _provider is not None:
        try:
            _provider.shutdown()
        except Exception as exc:
            logger.warning("OTel shutdown failed: %s", exc)
    _provider = None
    _tracer = None
    _setup_done = False


@contextmanager
def span(name: str, attributes: Optional[Dict[str, Any]] = None) -> Iterator[Any]:
    """Open a span if the tracer is live, else yield None. Never raises."""
    tracer = _ensure_setup()
    if tracer is None:
        yield None
        return
    try:
        with tracer.start_as_current_span(name) as sp:
            if attributes:
                for k, v in attributes.items():
                    try:
                        sp.set_attribute(k, v)
                    except Exception:
                        pass
            yield sp
    except Exception as exc:
        logger.debug("OTel span '%s' skipped: %s", name, exc)
        yield None


def bridge_emit_step(run_id: str, thread_id: str, node: str,
                     event_type: str, payload: Any) -> None:
    """Emit a span mirroring ``trace.emit_step``. Fire-and-forget."""
    tracer = _ensure_setup()
    if tracer is None:
        return
    try:
        with tracer.start_as_current_span(f"step.{node}.{event_type}") as sp:
            sp.set_attribute("run_id", run_id)
            sp.set_attribute("thread_id", thread_id)
            sp.set_attribute("node", node)
            sp.set_attribute("event_type", event_type)
    except Exception as exc:
        logger.debug("bridge_emit_step failed: %s", exc)


def bridge_emit_run_start(run_id: str, thread_id: str,
                          user_id: Optional[str], question: str) -> None:
    tracer = _ensure_setup()
    if tracer is None:
        return
    try:
        with tracer.start_as_current_span("run.start") as sp:
            sp.set_attribute("run_id", run_id)
            sp.set_attribute("thread_id", thread_id)
            if user_id:
                sp.set_attribute("user_id", user_id)
    except Exception as exc:
        logger.debug("bridge_emit_run_start failed: %s", exc)


def bridge_emit_run_end(run_id: str, status: str,
                        route: Optional[str] = None) -> None:
    tracer = _ensure_setup()
    if tracer is None:
        return
    try:
        with tracer.start_as_current_span("run.end") as sp:
            sp.set_attribute("run_id", run_id)
            sp.set_attribute("status", status)
            if route:
                sp.set_attribute("route", route)
    except Exception as exc:
        logger.debug("bridge_emit_run_end failed: %s", exc)


__all__ = [
    "setup_otel",
    "shutdown_otel",
    "span",
    "bridge_emit_step",
    "bridge_emit_run_start",
    "bridge_emit_run_end",
]