"""Unit tests for stats: bootstrap, McNemar, Wilcoxon, pass@k/pass^k, Holm."""
import math

import pytest

from evaluation.stats import (
    bootstrap_ci,
    effect_size_hedges,
    holm_bonferroni,
    paired_mcnemar,
    pass_at_k,
    pass_pow_k,
    wilcoxon,
)


def test_bootstrap_ci_covers_mean():
    scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    ci = bootstrap_ci(scores, n_boot=500, seed=1)
    assert ci.lower <= ci.estimate <= ci.upper
    assert ci.n == 10


def test_bootstrap_ci_empty():
    ci = bootstrap_ci([], n_boot=100, seed=1)
    assert ci.n == 0
    assert math.isnan(ci.estimate)


def test_mcnemar_exact_small_discordant():
    a = [True, True, True, False, False, False]
    b = [True, True, True, True, True, True]
    res = paired_mcnemar(a, b)
    assert res.method == "exact"
    assert res.discordant_b_only == 3
    assert res.discordant_a_only == 0
    assert res.p_value <= 0.25


def test_mcnemar_no_discordant_is_unit_p():
    res = paired_mcnemar([True, False], [True, False])
    assert res.p_value == 1.0
    assert res.discordant_b_only == 0


def test_mcnemar_chisq_large_discordant():
    a = [False] * 40 + [True] * 10
    b = [True] * 40 + [True] * 10
    res = paired_mcnemar(a, b)
    assert res.method == "chisq"
    assert res.p_value < 0.05


def test_mcnemar_length_mismatch_raises():
    with pytest.raises(ValueError):
        paired_mcnemar([True, False], [True])


def test_wilcoxon_identical_is_unit_p():
    res = wilcoxon([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    assert res.p_value == 1.0
    assert res.n_pairs == 3


def test_wilcoxon_shift_significant():
    a = [0.9] * 10
    b = [0.1] * 10
    res = wilcoxon(a, b)
    assert res.p_value < 0.05


def test_wilcoxon_length_mismatch_raises():
    with pytest.raises(ValueError):
        wilcoxon([1.0, 2.0], [1.0])


def test_pass_at_k_mean():
    assert pass_at_k([True, False, True]) == pytest.approx(2 / 3)


def test_pass_at_k_empty():
    assert pass_at_k([]) == 0.0


def test_pass_pow_k_all_succeed():
    assert pass_pow_k([True, True, True]) == 1.0


def test_pass_pow_k_any_fail():
    assert pass_pow_k([True, False, True]) == 0.0


def test_pass_pow_k_stricter_than_pass_at_k():
    runs = [True, True, False]
    assert pass_pow_k(runs) < pass_at_k(runs)


def test_effect_size_hedges_zero():
    assert effect_size_hedges([1.0, 1.0, 1.0], [1.0, 1.0, 1.0]) == 0.0


def test_effect_size_hedges_small_samples_nan():
    assert math.isnan(effect_size_hedges([1.0], [2.0]))


def test_holm_bonferroni_corrects():
    out = holm_bonferroni([0.04, 0.03, 0.20], alpha=0.05)
    assert len(out) == 3
    for c in out:
        assert c.adjusted_p >= c.original_p - 1e-12
        assert c.reject is False


def test_holm_bonferroni_empty():
    assert holm_bonferroni([]) == []


def test_holm_bonferroni_rejects_small_p():
    out = holm_bonferroni([0.001, 0.01, 0.04], alpha=0.05)
    assert out[0].reject is True