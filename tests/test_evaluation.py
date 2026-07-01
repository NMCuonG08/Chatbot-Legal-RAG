"""Tests for pure evaluation functions (no external services needed).

Covers:
- metrics_retrieval: hit@k, recall@k, mrr, ndcg@k, aggregate
- failure_analysis: classify_sample_failure, summarize_failures
- dataset.gold_in_retrieved: including the length-ratio guard added in Plan 4
- metrics_generation: claim-decomposition faithfulness (server-side score)
"""
import json

import pytest

from evaluation.dataset import EvalSample, gold_in_retrieved
from evaluation.failure_analysis import (
    FailureCategory,
    classify_sample_failure,
    summarize_failures,
)
from evaluation.metrics_generation import evaluate_faithfulness
from evaluation.metrics_retrieval import (
    aggregate_retrieval_metrics,
    hit_at_k,
    mrr,
    ndcg_at_k,
    recall_at_k,
)
from evaluation.eval_e2e import _resolve_expected_route, get_expected_route
from evaluation.evaluate_model import RunConfig


# ===== metrics_retrieval =====


class TestRetrievalMetrics:
    def test_hit_at_k_inside(self):
        assert hit_at_k(3, 5) == 1.0

    def test_hit_at_k_outside(self):
        assert hit_at_k(6, 5) == 0.0

    def test_hit_at_k_not_found(self):
        assert hit_at_k(0, 5) == 0.0

    def test_recall_at_k_single_relevant_equals_hit(self):
        assert recall_at_k(2, 5) == hit_at_k(2, 5)

    def test_recall_at_k_zero_total_relevant(self):
        # max(total_relevant, 1) guards divide-by-zero.
        assert recall_at_k(2, 5, total_relevant=0) == hit_at_k(2, 5)

    def test_mrr_found(self):
        assert mrr(2) == 0.5

    def test_mrr_not_found(self):
        assert mrr(0) == 0.0

    def test_ndcg_at_k_rank1(self):
        # IDCG = 1/log2(2) = 1.0; rank 1 -> DCG = 1.0 -> nDCG = 1.0
        assert ndcg_at_k(1, 5) == pytest.approx(1.0 / math_log2(2))

    def test_ndcg_at_k_outside(self):
        assert ndcg_at_k(6, 5) == 0.0

    def test_aggregate_empty(self):
        assert aggregate_retrieval_metrics([]) == {"n_queries": 0}

    def test_aggregate_basic(self):
        out = aggregate_retrieval_metrics([1, 0, 3], ks=(1, 3))
        assert out["n_queries"] == 3
        assert out["hit@1"] == pytest.approx(1 / 3)
        assert out["hit@3"] == pytest.approx(2 / 3)
        assert "mrr" in out


def math_log2(x):
    import math
    return math.log2(x)


# ===== failure_analysis =====


class TestFailureClassification:
    def test_error_priority(self):
        assert classify_sample_failure(error="boom") == FailureCategory.ERROR

    def test_routing_fail(self):
        assert classify_sample_failure(
            actual_route="general_chat", expected_route="agent_tools"
        ) == FailureCategory.ROUTING_FAIL

    def test_routing_legal_routes_not_penalized(self):
        # legal_rag vs agent_tools both legal -> not a routing failure.
        assert classify_sample_failure(
            actual_route="agent_tools", expected_route="legal_rag"
        ) != FailureCategory.ROUTING_FAIL

    def test_retrieval_fail(self):
        assert classify_sample_failure(
            actual_route="legal_rag", expected_route="legal_rag", retrieval_hit=False
        ) == FailureCategory.RETRIEVER_FAIL

    def test_hallucination(self):
        assert classify_sample_failure(
            retrieval_hit=True, faithfulness=0.4, answer_relevance=0.9
        ) == FailureCategory.HALLUCINATION

    def test_irrelevance(self):
        assert classify_sample_failure(
            retrieval_hit=True, faithfulness=0.9, answer_relevance=0.4
        ) == FailureCategory.IRRELEVANCE

    def test_success(self):
        assert classify_sample_failure(
            retrieval_hit=True, faithfulness=0.9, answer_relevance=0.9
        ) == FailureCategory.SUCCESS

    def test_summarize_empty(self):
        assert summarize_failures([]) == {}

    def test_summarize_counts(self):
        cats = [FailureCategory.SUCCESS, FailureCategory.SUCCESS, FailureCategory.RETRIEVER_FAIL]
        out = summarize_failures(cats)
        assert out[FailureCategory.SUCCESS] == pytest.approx(2 / 3)
        assert out[FailureCategory.RETRIEVER_FAIL] == pytest.approx(1 / 3)


# ===== dataset.gold_in_retrieved =====


def _sample(gold):
    return EvalSample(sample_id="s1", question="q", gold_context=gold)


class TestGoldInRetrieved:
    def test_exact_match(self):
        assert gold_in_retrieved(_sample("abc"), ["abc"]) == 1

    def test_not_found(self):
        assert gold_in_retrieved(_sample("abc"), ["xyz", "def"]) == 0

    def test_comparable_substring_match(self):
        # gold is a substring of retrieved, comparable length -> hit.
        gold = "Điều 418 Bộ luật Dân sự 2015 quy định mức phạt vi phạm."
        retrieved = gold + " Đây là tài liệu liên quan."
        # len ratio high -> hit
        assert gold_in_retrieved(_sample(gold), [retrieved]) == 1

    def test_length_ratio_guard_rejects_unrelated_long_chunk(self):
        # gold short, retrieved very long and merely contains gold -> NOT a hit.
        gold = "phạt"
        retrieved = "Điều 418 quy định mức phạt..." + (" nội dung không liên quan. " * 50)
        assert gold_in_retrieved(_sample(gold), [retrieved]) == 0

    def test_empty_retrieved_skipped(self):
        assert gold_in_retrieved(_sample("abc"), ["", "abc"]) == 2

    def test_empty_gold_list_returns_zero(self):
        assert gold_in_retrieved(_sample("abc"), []) == 0


# ===== metrics_generation: claim-decomposition faithfulness =====


def _judge_factory(payload: dict):
    """Build a fake judge_fn returning the given JSON-serializable payload."""
    def _judge(messages):
        return json.dumps(payload, ensure_ascii=False)
    return _judge


class TestFaithfulnessClaimDecomp:
    def test_empty_answer(self):
        out = evaluate_faithfulness("q", "", ["ctx"], _judge_factory({}))
        assert out.score == 0.0
        assert "empty_answer" in out.rationale

    def test_no_context(self):
        out = evaluate_faithfulness("q", "a", [], _judge_factory({}))
        assert out.score == 0.0
        assert "no_context" in out.rationale

    def test_all_claims_supported(self):
        judge = _judge_factory({
            "claims": ["c1", "c2"],
            "supported": [True, True],
            "reason": "ok",
        })
        out = evaluate_faithfulness("q", "a", ["ctx"], judge)
        assert out.score == pytest.approx(1.0)
        assert "2/2" in out.rationale

    def test_half_claims_supported(self):
        judge = _judge_factory({
            "claims": ["c1", "c2"],
            "supported": [True, False],
            "reason": "partial",
        })
        out = evaluate_faithfulness("q", "a", ["ctx"], judge)
        assert out.score == pytest.approx(0.5)
        assert "1/2" in out.rationale

    def test_truthy_string_flags(self):
        # Judge returns "yes"/"no" strings; both must be interpreted.
        judge = _judge_factory({
            "claims": ["c1", "c2", "c3"],
            "supported": ["yes", "no", "1"],
            "reason": "mixed",
        })
        out = evaluate_faithfulness("q", "a", ["ctx"], judge)
        assert out.score == pytest.approx(2 / 3)

    def test_hallucinated_score_replaced_by_server_score(self):
        # Judge claims score=1.0 but only 1/3 supported -> server score wins.
        judge = _judge_factory({
            "claims": ["c1", "c2", "c3"],
            "supported": [True, False, False],
            "score": 1.0,
            "reason": "judge lies",
        })
        out = evaluate_faithfulness("q", "a", ["ctx"], judge)
        assert out.score == pytest.approx(1 / 3)
        assert "1/3" in out.rationale

    def test_missing_supported_falls_back_to_judge_score(self):
        # No supported array -> fall back to judge's aggregate score.
        judge = _judge_factory({"score": 0.7, "reason": "no claims"})
        out = evaluate_faithfulness("q", "a", ["ctx"], judge)
        assert out.score == pytest.approx(0.7)


# ===== eval_e2e: routing labels =====


class TestRoutingLabels:
    def test_heuristic_calc_keyword_routes_to_agent_tools(self):
        assert get_expected_route("tính tiền phạt vi phạm hợp đồng") == "agent_tools"

    def test_heuristic_default_routes_to_legal_rag(self):
        assert get_expected_route("quy định về tuổi kết hôn") == "legal_rag"

    def test_human_label_overrides_heuristic(self):
        # Question looks like a calc (heuristic -> agent_tools), but the human
        # gold label says legal_rag. The label must win.
        sample = EvalSample(
            sample_id="s1",
            question="tính tiền phạt",
            gold_context="ctx",
            expected_route="legal_rag",
        )
        assert _resolve_expected_route(sample) == "legal_rag"

    def test_no_label_falls_back_to_heuristic(self):
        sample = EvalSample(
            sample_id="s1",
            question="kiểm tra tuổi kết hôn",
            gold_context="ctx",
            expected_route=None,
        )
        assert _resolve_expected_route(sample) == "agent_tools"


# ===== evaluate_model: RunConfig parser =====


class TestRunConfigParse:
    def test_label_with_one_override(self):
        cfg = RunConfig.parse("llama-8b:LLM_MODEL=llama-3.1-8b-instant")
        assert cfg.label == "llama-8b"
        assert cfg.env_overrides == {"LLM_MODEL": "llama-3.1-8b-instant"}

    def test_label_with_multiple_overrides(self):
        cfg = RunConfig.parse("local:LLM_PROVIDER=ollama,LLM_MODEL=vietnamese-legal-llm")
        assert cfg.label == "local"
        assert cfg.env_overrides == {
            "LLM_PROVIDER": "ollama",
            "LLM_MODEL": "vietnamese-legal-llm",
        }

    def test_label_only_no_overrides(self):
        cfg = RunConfig.parse("baseline:")
        assert cfg.label == "baseline"
        assert cfg.env_overrides == {}

    def test_missing_colon_raises(self):
        with pytest.raises(Exception):
            RunConfig.parse("no-colon-here")

    def test_missing_equals_raises(self):
        with pytest.raises(Exception):
            RunConfig.parse("bad:LLM_MODEL-no-equals")