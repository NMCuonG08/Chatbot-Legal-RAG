"""Phase F tests: SSE trace stream at GET /chat/stream/{task_id}.

Fakes redis.asyncio pub/sub so no live Redis is required. Verifies the endpoint
resolves run_id from the task_id, filters pub/sub frames by run_id, and closes
after the run_end event.
"""
import json

import backend.src.app as app_module
from fastapi.testclient import TestClient


class _FakePubSub:
    def __init__(self, messages):
        self._messages = messages
        self._i = 0

    async def subscribe(self, *channels):
        return None

    async def get_message(self, ignore_subscribe_messages=True, timeout=None):
        if self._i < len(self._messages):
            m = self._messages[self._i]
            self._i += 1
            return m
        return None  # no more frames

    async def unsubscribe(self, *channels):
        return None

    async def aclose(self):
        return None


class _FakeClient:
    def __init__(self, messages):
        self._messages = messages

    def pubsub(self):
        return _FakePubSub(self._messages)

    async def aclose(self):
        return None


def _frame(run_id, node, event_type, payload):
    return {
        "type": "message",
        "data": json.dumps({
            "run_id": run_id,
            "node": node,
            "event_type": event_type,
            "payload": payload,
        }, ensure_ascii=False),
    }


def test_sse_stream_filters_by_run_id_and_closes_on_run_end(monkeypatch):
    messages = [
        _frame("other-run", "route", "node_end", {"route": "legal_rag"}),     # filtered out
        _frame("run-xyz", "route", "node_end", {"route": "legal_rag"}),        # included
        _frame("run-xyz", "retrieve", "node_end", {"doc_count": 3}),           # included
        _frame("run-xyz", "__root__", "run_end", {"status": "completed"}),     # included + close
    ]

    async def fake_resolve(task_id, timeout=15.0):
        return "run-xyz"

    monkeypatch.setattr(app_module, "_resolve_run_id", fake_resolve)
    monkeypatch.setattr(app_module.aioredis, "from_url",
                        lambda *a, **k: _FakeClient(messages))

    client = TestClient(app_module.app)
    with client.stream("GET", "/chat/stream/task-1") as resp:
        assert resp.status_code == 200
        lines = list(resp.iter_lines())

    # sse-starlette emits "event: <type>" and "data: <json>" lines.
    data_blobs = [ln[len("data:"):].strip() for ln in lines if ln.startswith("data:")]
    parsed = [json.loads(b) for b in data_blobs if b]

    # Non-matching run_id must NOT appear.
    assert all(p.get("run_id") != "other-run" for p in parsed)
    # All streamed step frames belong to run-xyz.
    step_frames = [p for p in parsed if p.get("event_type") in ("node_end", "run_end")]
    assert all(p.get("run_id") == "run-xyz" for p in step_frames)
    assert any(p.get("event_type") == "run_end" for p in parsed)


def test_sse_stream_404_when_run_not_resolved(monkeypatch):
    async def fake_resolve(task_id, timeout=15.0):
        return None

    monkeypatch.setattr(app_module, "_resolve_run_id", fake_resolve)

    client = TestClient(app_module.app)
    resp = client.get("/chat/stream/task-missing")
    assert resp.status_code == 404