"""Unit tests for otel_bridge: disabled no-op, idempotent setup, failure isolation."""
from evaluation import otel_bridge


def test_disabled_by_default_is_noop(monkeypatch):
    monkeypatch.setattr(otel_bridge, "_ENABLED", False)
    otel_bridge.bridge_emit_step("r1", "t1", "node", "step", {"x": 1})
    otel_bridge.bridge_emit_run_start("r1", "t1", "u1", "q")
    otel_bridge.bridge_emit_run_end("r1", "completed", route="legal_rag")


def test_span_yields_none_when_disabled(monkeypatch):
    monkeypatch.setattr(otel_bridge, "_ENABLED", False)
    with otel_bridge.span("test") as sp:
        assert sp is None


def test_setup_noop_without_endpoint(monkeypatch):
    monkeypatch.setattr(otel_bridge, "_ENABLED", True)
    monkeypatch.setattr(otel_bridge, "_OTLP_ENDPOINT", None)
    monkeypatch.setattr(otel_bridge, "_setup_done", False)
    assert otel_bridge._ensure_setup() is None
    with otel_bridge.span("x") as sp:
        assert sp is None


def test_setup_failure_does_not_raise(monkeypatch):
    """If opentelemetry import fails, bridge degrades to no-op silently."""
    monkeypatch.setattr(otel_bridge, "_ENABLED", True)
    monkeypatch.setattr(otel_bridge, "_OTLP_ENDPOINT", "http://localhost:4318")
    monkeypatch.setattr(otel_bridge, "_setup_done", False)
    import builtins
    real_import = builtins.__import__

    def _fail(name, *a, **k):
        if name.startswith("opentelemetry"):
            raise ImportError("simulated absent otel")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _fail)
    assert otel_bridge._ensure_setup() is None
    otel_bridge.bridge_emit_step("r1", "t1", "n", "e", {})


def test_setup_otel_returns_false_when_disabled(monkeypatch):
    monkeypatch.setattr(otel_bridge, "_ENABLED", False)
    monkeypatch.setattr(otel_bridge, "_OTLP_ENDPOINT", None)
    monkeypatch.setattr(otel_bridge, "_setup_done", False)
    assert otel_bridge.setup_otel() is False


def test_shutdown_is_safe_when_no_provider(monkeypatch):
    monkeypatch.setattr(otel_bridge, "_provider", None)
    otel_bridge.shutdown_otel()