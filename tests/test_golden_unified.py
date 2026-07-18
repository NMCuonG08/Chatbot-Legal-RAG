"""Unit tests for golden_unified: dedup, 3-source merge, roundtrip, to_eval_sample."""
import pytest

from evaluation import golden_unified as gu
from evaluation.golden_unified import GoldenItem


def _norm(t):
    return (t or "").strip().lower()


@pytest.fixture
def stub_sources(monkeypatch):
    """Stub the 3 source loaders with deterministic synthetic items."""
    a = GoldenItem(sample_id="a1", question="Điều 10 nói gì?",
                   expected_route="legal_rag", expected_tool="search_law",
                   source="run_question_test")
    b = GoldenItem(sample_id="b1", question="Điều 10 nói gì?",  # dup of a
                   expected_route="legal_rag", source="eval_router")
    c = GoldenItem(sample_id="c1", question="Hello world",
                   expected_route="general_chat", source="eval_prompts")
    monkeypatch.setattr(gu, "_from_run_question_test", lambda: [a])
    monkeypatch.setattr(gu, "_from_eval_router", lambda: [b])
    monkeypatch.setattr(gu, "_from_eval_prompts", lambda: [c])
    return [a, b, c]


def test_unify_dedup_run_question_test_wins(stub_sources):
    items = gu.unify_golden()
    assert len(items) == 2  # a/b deduped
    a_item = next(i for i in items if i.sample_id == "a1")
    assert a_item.expected_tool == "search_law"  # run_question_test wins
    assert all(i.sample_id != "b1" for i in items)


def test_unify_three_sources_covered(stub_sources):
    items = gu.unify_golden()
    sources = {i.source for i in items}
    assert "run_question_test" in sources
    assert "eval_prompts" in sources


def test_write_load_roundtrip(stub_sources, tmp_path):
    path = tmp_path / "golden.jsonl"
    gu.write_unified_dataset(path)
    loaded = gu.load_unified_dataset(path)
    assert len(loaded) == 2
    ids = {i.sample_id for i in loaded}
    assert ids == {"a1", "c1"}
    a = next(i for i in loaded if i.sample_id == "a1")
    assert a.expected_route == "legal_rag"
    assert a.expected_tool == "search_law"


def test_load_missing_returns_empty(tmp_path):
    assert gu.load_unified_dataset(tmp_path / "nope.jsonl") == []


def test_to_eval_sample():
    item = GoldenItem(sample_id="x", question="q?", expected_route="legal_rag",
                      expected_answer="ans", expected_tool="search_law",
                      expected_block=True)
    es = gu.to_eval_sample(item)
    assert es.sample_id == "x"
    assert es.gold_context == "ans"
    assert es.expected_route == "legal_rag"
    assert es.expected_tool == "search_law"
    assert es.expected_block is True


def test_golden_item_frozen():
    item = GoldenItem(sample_id="x", question="q")
    with pytest.raises(Exception):
        item.sample_id = "y"  # frozen dataclass