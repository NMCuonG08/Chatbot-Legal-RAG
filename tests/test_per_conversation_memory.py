"""Phase E tests: per-conversation agent memory isolation (cross-user leak fix).

Previously a single global ChatMemoryBuffer was shared across all users. Now
memory (and the agent cache) is keyed by (user_id, conversation_id). These tests
confirm distinct keys yield distinct objects and the same key is stable.
"""
import agent


def test_get_agent_memory_distinct_per_user():
    m1 = agent.get_agent_memory("u1", "c1")
    m2 = agent.get_agent_memory("u2", "c1")
    assert m1 is not m2


def test_get_agent_memory_distinct_per_conversation():
    m1 = agent.get_agent_memory("u1", "c1")
    m2 = agent.get_agent_memory("u1", "c2")
    assert m1 is not m2


def test_get_agent_memory_same_key_stable():
    m1 = agent.get_agent_memory("u1", "c1")
    m2 = agent.get_agent_memory("u1", "c1")
    assert m1 is m2


def test_get_agent_memory_lru_evicts_oldest():
    # Fill beyond cap to force eviction of the oldest key.
    agent._agent_memories.clear()
    for i in range(agent._AGENT_MEMORY_CAP + 2):
        agent.get_agent_memory(f"u{i}", f"c{i}")
    # Oldest keys evicted; the cache must not exceed the cap.
    assert len(agent._agent_memories) <= agent._AGENT_MEMORY_CAP
    # The newest key is still present.
    newest = (f"u{agent._AGENT_MEMORY_CAP + 1}", f"c{agent._AGENT_MEMORY_CAP + 1}")
    assert newest in agent._agent_memories
    # The oldest key was evicted.
    assert ("u0", "c0") not in agent._agent_memories


def test_get_ai_agent_distinct_per_key(monkeypatch):
    """_get_ai_agent returns a distinct agent per (user, conversation)."""
    agent._ai_agent_cache.clear()
    # Stub the LLM + builder so no real network/credentials are needed.
    monkeypatch.setattr(agent, "_build_llm", lambda: object())  # truthy non-None LLM

    class FakeAgent:
        def __init__(self, tag):
            self.tag = tag

    # Tag the agent with the memory object identity to prove isolation.
    monkeypatch.setattr(agent, "_build_react_agent", lambda llm, memory: FakeAgent(id(memory)))

    a1 = agent._get_ai_agent("u1", "c1")
    a2 = agent._get_ai_agent("u2", "c1")
    assert a1 is not a2
    assert a1.tag != a2.tag  # built from distinct memory objects
    # Same key returns the cached agent.
    assert agent._get_ai_agent("u1", "c1") is a1


def test_get_ai_agent_legacy_default_args(monkeypatch):
    """Callers using _get_ai_agent() with no args still work (shared default)."""
    agent._ai_agent_cache.clear()
    monkeypatch.setattr(agent, "_build_llm", lambda: object())
    monkeypatch.setattr(agent, "_build_react_agent", lambda llm, memory: object())
    a = agent._get_ai_agent()
    assert a is not None
    assert agent._get_ai_agent() is a  # cached under (None, None)