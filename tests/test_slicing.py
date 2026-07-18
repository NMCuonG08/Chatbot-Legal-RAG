"""Unit tests for slicing: intent, difficulty, language, oos, apply, summarize."""
from evaluation.dataset import EvalSample
from evaluation.slicing import (
    SliceSpec,
    apply_slices,
    detect_language,
    slice_by_difficulty,
    slice_by_intent,
    slice_by_language,
    slice_by_oos,
    summarize_by_slice,
)


def _s(sid, q, route=None):
    return EvalSample(sample_id=sid, question=q, gold_context="ctx",
                      expected_route=route)


def test_slice_by_intent():
    samples = [_s("a", "q1", "legal_rag"), _s("b", "q2", "agent_tools"),
               _s("c", "q3", "legal_rag"), _s("d", "q4", None)]
    out = slice_by_intent(samples)
    assert len(out["legal_rag"]) == 2
    assert len(out["agent_tools"]) == 1
    assert len(out["unknown"]) == 1


def test_slice_by_oos():
    samples = [_s("a", "q1", "legal_rag"), _s("b", "q2", None)]
    out = slice_by_oos(samples)
    assert len(out["oos"]) == 1
    assert len(out["in_scope"]) == 1


def test_detect_language_vietnamese():
    assert detect_language("Điều 10 Bộ luật Dân sự") == "vi"


def test_detect_language_english():
    assert detect_language("What is the penalty for theft?") == "en"


def test_detect_language_empty():
    assert detect_language("") == "vi"


def test_slice_by_difficulty_easy():
    samples = [_s("a", "Xin chào")]
    out = slice_by_difficulty(samples)
    assert len(out["easy"]) == 1


def test_slice_by_difficulty_hard():
    q = ("Điều 418 Bộ luật Dân sự 2015 quy định mức phạt vi phạm hợp đồng tối đa "
         "bao nhiêu phần trăm trong giao dịch dân sự này khi một bên vi phạm nghĩa "
         "vụ hợp đồng đã thỏa thuận và gây thiệt hại lớn cho bên kia?")
    out = slice_by_difficulty([_s("a", q)])
    assert len(out["hard"]) == 1


def test_slice_by_language_groups():
    samples = [_s("a", "Điều 10"), _s("b", "Hello world")]
    out = slice_by_language(samples)
    assert len(out["vi"]) == 1
    assert len(out["en"]) == 1


def test_apply_slices_with_specs():
    samples = [_s("a", "Điều 10", "legal_rag"), _s("b", "Hello", "general_chat")]
    specs = [SliceSpec("rag", lambda s: s.expected_route == "legal_rag"),
             SliceSpec("short", lambda s: len(s.question) < 20)]
    out = apply_slices(samples, specs)
    assert [s.sample_id for s in out["rag"]] == ["a"]
    assert set(s.sample_id for s in out["short"]) == {"a", "b"}


def test_summarize_by_slice():
    slices = {"rag": [_s("a", "q"), _s("b", "q")], "oos": []}
    summary = summarize_by_slice(slices, lambda items: len(items) / 10)
    assert summary["rag"] == {"n": 2, "metric": 0.2}
    assert summary["oos"] == {"n": 0, "metric": 0.0}