"""Unit tests for judge_panel: swap flip, CoT emit, panel aggregation, kappa."""
import re

from evaluation.judge_panel import (
    cohen_kappa,
    cot_score,
    multi_judge_panel,
    swap_augment_pairwise,
    JudgeVote,
)


# Parse the two labelled answer blocks from a pairwise prompt, in order.
_BLOCK_RE = re.compile(r"Câu trả lời (A|B):\s*(.*?)(?=Câu trả lời |\nTiêu chí|\Z)", re.DOTALL)


def _blocks(text):
    return [(label, txt.strip()) for label, txt in _BLOCK_RE.findall(text)]


def _content_judge(win_token):
    """Prefer the answer whose text contains ``win_token`` (order-invariant)."""
    def _fn(messages):
        blocks = _blocks(messages[-1]["content"])
        for label, txt in blocks:
            if win_token in txt:
                return f'{{"winner": "{label}", "reason": ""}}'
        return '{"winner": "tie", "reason": ""}'
    return _fn


def _position_judge(messages):
    """Always pick the FIRST-presented answer label (position bias)."""
    blocks = _blocks(messages[-1]["content"])
    if not blocks:
        return '{"winner": "tie", "reason": ""}'
    return f'{{"winner": "{blocks[0][0]}", "reason": ""}}'


# ----- swap_augment_pairwise -----

def test_swap_content_judge_agrees_a_wins():
    j = _content_judge("WIN")
    v = swap_augment_pairwise("q", "ansWIN", "ansB", j)
    assert v.score == 1.0
    assert v.aggregation == "swap"
    assert len(v.votes) == 2
    assert v.votes[0].score == 1.0 and v.votes[1].score == 1.0


def test_swap_content_judge_b_wins():
    j = _content_judge("WIN")
    v = swap_augment_pairwise("q", "ansA", "ansWIN", j)
    assert v.score == -1.0


def test_swap_position_bias_becomes_tie():
    v = swap_augment_pairwise("q", "ansA", "ansB", _position_judge)
    # run1 first label A, run2 first label B -> disagree -> tie
    assert v.score == 0.0
    assert v.votes[0].score == 1.0  # run1 -> A
    assert v.votes[1].score == -1.0  # run2 -> B


def test_swap_tie_consistent():
    j = lambda m: '{"winner": "tie", "reason": ""}'
    v = swap_augment_pairwise("q", "ansA", "ansB", j)
    assert v.score == 0.0
    assert v.votes[0].score == 0.0 and v.votes[1].score == 0.0


def test_swap_unparseable_defaults_tie():
    j = lambda m: "garbage no json"
    v = swap_augment_pairwise("q", "ansA", "ansB", j)
    assert v.score == 0.0


# ----- cot_score -----

def test_cot_score_emits_score():
    j = lambda m: '{"reasoning": "ok", "score": 0.8}'
    vote = cot_score("faithfulness", "q", "a", "ctx", j)
    assert isinstance(vote, JudgeVote)
    assert vote.score == 0.8
    assert vote.rationale == "ok"


def test_cot_score_clamps_and_defaults():
    j = lambda m: '{"reasoning": "", "score": 5}'
    vote = cot_score("faithfulness", "q", "a", "ctx", j)
    assert vote.score == 1.0  # clamped
    j2 = lambda m: '{"reasoning": ""}'
    vote2 = cot_score("faithfulness", "q", "a", "ctx", j2)
    assert vote2.score == 0.0  # missing -> default


# ----- multi_judge_panel -----

def _call_builder(vote_score=0.5):
    """Builder that actually invokes the judge (so failures surface)."""
    def _prompt_builder(judge, judge_id):
        raw = judge([{"role": "user", "content": "score this"}])
        return JudgeVote(judge_id=judge_id, score=vote_score, raw=raw)
    return _prompt_builder


def test_panel_mean_aggregation():
    judges = [lambda m: "x", lambda m: "y", lambda m: "z"]
    v = multi_judge_panel(judges, _call_builder(0.6), metric="faithfulness",
                          aggregation="mean")
    assert v.score == 0.6
    assert len(v.votes) == 3
    assert v.aggregation == "mean"


def test_panel_majority_robust_to_outlier():
    judges = [lambda m: "x"] * 5
    def builder(judge, judge_id):
        judge([{"role": "user", "content": "x"}])  # call to be realistic
        idx = int(judge_id.split("_")[1])
        return JudgeVote(judge_id=judge_id, score=0.9 if idx == 0 else 0.2)
    v = multi_judge_panel(judges, builder, aggregation="majority")
    # sorted [0.2,0.2,0.2,0.2,0.9] -> median index 2 -> 0.2
    assert v.score == 0.2


def test_panel_drops_failed_judges():
    def bad_judge(messages):
        raise RuntimeError("boom")
    judges = [bad_judge, lambda m: "ok"]
    v = multi_judge_panel(judges, _call_builder(0.5), aggregation="mean")
    assert len(v.votes) == 1
    assert v.score == 0.5


def test_panel_all_fail_returns_zero():
    def bad(messages):
        raise RuntimeError("boom")
    v = multi_judge_panel([bad, bad], _call_builder(0.5), aggregation="mean")
    assert v.score == 0.0
    assert v.votes == []


# ----- cohen_kappa -----

def test_kappa_perfect_agreement():
    k = cohen_kappa(["low", "mid", "high"], ["low", "mid", "high"])
    assert k == 1.0


def test_kappa_no_agreement():
    k = cohen_kappa(["low", "low", "high", "high"],
                    ["high", "high", "low", "low"])
    assert k < 0.0


def test_kappa_empty_or_mismatched_length():
    assert cohen_kappa([], []) == 0.0
    assert cohen_kappa(["low"], ["low", "mid"]) == 0.0


def test_kappa_constant_same_label():
    # both constant on "low" -> perfect by convention
    assert cohen_kappa(["low", "low"], ["low", "low"]) == 1.0