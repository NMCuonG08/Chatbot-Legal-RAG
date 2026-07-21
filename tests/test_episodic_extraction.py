"""P2 — Event-driven episodic extraction tests (FLAW 2 fix).

Covers:
- extracts ONLY user personal facts (delta-only, O(1)/turn)
- NEVER ingests bot legal text ("Điều 100…", "Luật…") — pollution regression
- delta_message path does NOT load full history (no DB read)
- dedup (0.88) still skips near-identical stored facts
- strip_legal_citations removes citations before extraction

Heavy import chain (langchain_groq via summarizer) is stubbed the same way
``test_canary_shadow`` does, so ``tasks`` imports without a live Groq install.
"""
from __future__ import annotations

import json
import sys
import types

import pytest


@pytest.fixture
def tasks_module(monkeypatch):
    """Import tasks with heavy deps stubbed. Yields the tasks module."""
    for name in ("agent", "verify_answer", "metacognitive", "rlhf_store",
                 "summarizer", "guardrails_manager"):
        mod = types.ModuleType(name)
        if name == "agent":
            mod.ai_agent_handle = lambda *a, **k: None
            mod.clear_user_runtime_caches = lambda *a, **k: None
        if name == "verify_answer":
            mod.judge_answer = lambda *a, **k: None
        if name == "metacognitive":
            mod.build_escalation = lambda *a, **k: None
            mod.ESCALATION_PREFIX = "[ESCALATION]"
        if name == "rlhf_store":
            mod.find_similar_good = lambda *a, **k: None
        if name == "summarizer":
            mod.summarize_text = lambda *a, **k: "summary"
        if name == "guardrails_manager":
            mod.LegalGuardrailsManager = type("LegalGuardrailsManager", (), {"__init__": lambda self, *a, **k: None})
        monkeypatch.setitem(sys.modules, name, mod)
    import tasks  # noqa: WPS433
    return tasks


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def test_strip_legal_citations_removes_dieu_and_luat(tasks_module):
    strip = tasks_module.strip_legal_citations
    text = "Tôi tranh chấp đất ở Hà Nội. Điều 100 Bộ luật dân sự quy định thừa kế. [Tài liệu 2]"
    cleaned = strip(text)
    assert "Điều 100" not in cleaned
    assert "[Tài liệu 2]" not in cleaned
    assert "tranh chấp đất" in cleaned
    assert "Hà Nội" in cleaned


def test_strip_legal_citations_idempotent_and_safe_on_plain(tasks_module):
    strip = tasks_module.strip_legal_citations
    assert strip("") == ""
    plain = "Tôi tên A, sinh 1990, ở Đà Nẵng"
    assert strip(plain) == plain
    assert strip(strip(plain)) == plain


def test_parse_episodic_json_none_and_valid(tasks_module):
    parse = tasks_module._parse_episodic_json
    assert parse("NONE") is None
    assert parse("") is None
    assert parse("not json") is None
    payload = '{"facts": ["a"], "structured": {"name": "A"}}'
    parsed = parse(payload)
    assert parsed["facts"] == ["a"]
    parsed2 = parse(f'Here is result:\n{payload}\nDone.')
    assert parsed2["structured"]["name"] == "A"


# ---------------------------------------------------------------------------
# save_episodic_memory_task — delta path, no full-history load
# ---------------------------------------------------------------------------

@pytest.fixture
def stub_deps(tasks_module, monkeypatch):
    """Stub LLM + embedding + vector + DB so no live services are touched."""
    tasks = tasks_module
    calls = {"get_conv": 0, "save_episode": [], "add_vector": [], "dedup": []}

    monkeypatch.setattr(tasks, "openai_chat_complete", lambda msgs: json.dumps({
        "facts": ["user ten A, sinh 1990, o Ha Noi"],
        "structured": {"name": "A", "birth_year": 1990, "location": "Ha Noi",
                       "case_type": None, "case_summary": None, "gender": None},
    }, ensure_ascii=False))
    monkeypatch.setattr(tasks, "get_embedding", lambda text: [0.1, 0.2, 0.3])
    monkeypatch.setattr(tasks, "search_vector", lambda **kw: (calls["dedup"].append(kw) or None))
    import models
    import vectorize
    monkeypatch.setattr(models, "get_conversation_messages",
                        lambda conv_id: (calls.__setitem__("get_conv", calls["get_conv"] + 1), [])[1])
    monkeypatch.setattr(models, "save_user_episode",
                        lambda uid, txt: calls["save_episode"].append((uid, txt)))
    monkeypatch.setattr(vectorize, "add_vector", lambda **kw: calls["add_vector"].append(kw))
    return calls


def test_delta_path_does_not_load_history(stub_deps, monkeypatch, tasks_module):
    """delta_message present -> get_conversation_messages NEVER called (O(1))."""
    import models
    def _boom(conv_id):
        raise AssertionError("delta path must not load history")
    monkeypatch.setattr(models, "get_conversation_messages", _boom)

    result = tasks_module.save_episodic_memory_task(
        "u1", "c1", delta_message={"role": "user", "content": "Tôi tên A, sinh 1990"}
    )
    assert result == "success"
    assert stub_deps["get_conv"] == 0
    saved = stub_deps["save_episode"][0][1]
    assert "sinh 1990" in saved


def test_does_not_ingest_bot_legal_text(stub_deps, monkeypatch, tasks_module):
    """Regression: bot law text in delta must NOT reach the extraction prompt."""
    tasks = tasks_module
    bot_law = "Theo Điều 100 Bộ luật dân sự, thừa kế theo pháp luật được quy định..."
    captured = {}
    def fake_llm(msgs):
        captured["prompt"] = msgs[1]["content"]
        return json.dumps({"facts": ["user o Ha Noi"], "structured": {}}, ensure_ascii=False)
    monkeypatch.setattr(tasks, "openai_chat_complete", fake_llm)

    tasks.save_episodic_memory_task(
        "u1", "c1", delta_message={"role": "user", "content": bot_law + " Tôi ở Hà Nội."}
    )
    prompt = captured["prompt"]
    # "Bộ luật dân sự" is bot text, NOT in the prompt template (template only
    # cites "Điều 100"/"Luật Đất đai" as examples). Its absence proves citation
    # stripping removed the bot law text before extraction.
    assert "Bộ luật dân sự" not in prompt
    assert "Hà Nội" in prompt


def test_returns_skipped_none_when_no_fact(stub_deps, monkeypatch, tasks_module):
    """Generic legal question with no personal fact -> NONE -> skip."""
    monkeypatch.setattr(tasks_module, "openai_chat_complete", lambda msgs: "NONE")
    result = tasks_module.save_episodic_memory_task(
        "u1", "c1", delta_message={"role": "user", "content": "Thủ tục ly hôn thế nào?"}
    )
    assert result == "skipped_none"
    assert stub_deps["save_episode"] == []
    assert stub_deps["add_vector"] == []


def test_dedup_skips_near_identical_fact(stub_deps, monkeypatch, tasks_module):
    """Existing similar fact in Qdrant -> skip save (dedup unchanged)."""
    monkeypatch.setattr(tasks_module, "search_vector", lambda **kw: [{"id": "x"}])  # hit
    result = tasks_module.save_episodic_memory_task(
        "u1", "c1", delta_message={"role": "user", "content": "Tôi tên A, sinh 1990"}
    )
    assert result == "skipped_duplicate"
    assert stub_deps["save_episode"] == []
    assert stub_deps["add_vector"] == []


def test_legacy_path_ingests_only_last_user_message(stub_deps, monkeypatch, tasks_module):
    """delta_message=None + EPISODIC_DELTA_ENABLED -> LAST user turn only.

    Bot assistant content must NEVER be ingested.
    """
    import models
    history = [
        {"role": "user", "content": "Tôi tên A, sinh 1990"},
        {"role": "assistant", "content": "Theo Điều 100 Bộ luật dân sự, thừa kế..."},
        {"role": "user", "content": "Tôi ở Đà Nẵng"},  # <- LAST user turn
    ]
    monkeypatch.setattr(models, "get_conversation_messages", lambda conv_id: history)
    captured = {}
    def fake_llm(msgs):
        captured["prompt"] = msgs[1]["content"]
        return json.dumps({"facts": ["user o Da Nang"], "structured": {}}, ensure_ascii=False)
    monkeypatch.setattr(tasks_module, "openai_chat_complete", fake_llm)

    result = tasks_module.save_episodic_memory_task("u1", "c1", delta_message=None)
    assert result == "success"
    assert "Đà Nẵng" in captured["prompt"]
    # Bot's "Bộ luật dân sự" text stripped; earlier user "sinh 1990" not ingested.
    assert "Bộ luật dân sự" not in captured["prompt"]
    assert "sinh 1990" not in captured["prompt"]