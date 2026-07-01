"""Phase B tests: LangGraph checkpoint saver wiring."""
import tasks
from langgraph.checkpoint.memory import MemorySaver


def test_checkpointer_memory_saver_when_flag_set():
    original = tasks.settings.use_memory_saver
    try:
        tasks.settings.use_memory_saver = True
        tasks._get_checkpointer.cache_clear()
        saver = tasks._get_checkpointer()
        assert isinstance(saver, MemorySaver)
    finally:
        tasks.settings.use_memory_saver = original
        tasks._get_checkpointer.cache_clear()


def test_checkpointer_memory_saver_when_redis_unreachable(monkeypatch):
    class FakeFromUrl:
        def __init__(self, *a, **k):
            pass

        def ping(self):
            raise RuntimeError("no redis")

        def close(self):
            pass

    monkeypatch.setattr(tasks.settings, "use_memory_saver", False)
    monkeypatch.setattr(tasks.redis, "from_url", lambda *a, **k: FakeFromUrl())
    tasks._get_checkpointer.cache_clear()
    try:
        saver = tasks._get_checkpointer()
        assert isinstance(saver, MemorySaver)
    finally:
        tasks._get_checkpointer.cache_clear()


def test_graph_compiled_with_checkpointer():
    tasks._get_checkpointer.cache_clear()
    graph = tasks.get_chat_graph()
    assert graph.checkpointer is not None