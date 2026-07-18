"""Statistical helpers for eval comparison + reliability.

Pure functions over numpy/scipy. Deterministic given a seed. Used by
``regression.diff_runs`` to decide whether a metric delta is signal or noise,
and by the harness to report ``pass@k`` / ``pass^k`` reliability.

Public surface:
- ``bootstrap_ci`` — percentile CI of a statistic over resampled scores.
- ``paired_mcnemar`` — paired binary test (exact for small discordant, chisq else).
- ``wilcoxon`` — paired signed-rank test for continuous scores.
- ``pass_at_k`` / ``pass_pow_k`` — any-k / all-k reliability.
- ``effect_size_hedges`` — standardized mean difference.
- ``holm_bonferroni`` — multiplicity correction over a list of p-values.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Sequence

import numpy as np
from scipy import stats as sp_stats


@dataclass(frozen=True)
class CIMean:
    """Bootstrap confidence interval for a scalar statistic."""
    estimate: float
    lower: float
    upper: float
    n: int


@dataclass(frozen=True)
class McNemarResult:
    """Paired binary test result."""
    statistic: float
    p_value: float
    discordant_b_only: int  # b succeeded where a failed
    discordant_a_only: int  # a succeeded where b failed
    method: str  # "exact" | "chisq"


@dataclass(frozen=True)
class WilcoxonResult:
    statistic: float
    p_value: float
    n_pairs: int


def bootstrap_ci(
    scores: Sequence[float],
    *,
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
    statistic: Callable[[np.ndarray], float] = np.mean,
) -> CIMean:
    """Percentile bootstrap CI of ``statistic`` over ``scores``."""
    arr = np.asarray(scores, dtype=float)
    n = int(arr.size)
    if n == 0:
        return CIMean(estimate=float("nan"), lower=float("nan"),
                      upper=float("nan"), n=0)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = rng.choice(arr, size=n, replace=True)
        boots[i] = float(statistic(sample))
    alpha = (1.0 - ci) / 2.0
    lower = float(np.quantile(boots, alpha))
    upper = float(np.quantile(boots, 1.0 - alpha))
    return CIMean(estimate=float(statistic(arr)), lower=lower, upper=upper, n=n)


def paired_mcnemar(successes_a: Sequence[bool], successes_b: Sequence[bool]) -> McNemarResult:
    """McNemar test on paired binary outcomes.

    ``successes_a`` / ``successes_b`` are aligned per-item success flags for
    systems A and B. Uses the exact binomial when discordants are small (<25),
    otherwise the continuity-corrected chi-square.
    """
    a = np.asarray(successes_a, dtype=bool)
    b = np.asarray(successes_b, dtype=bool)
    if a.size != b.size:
        raise ValueError("paired_mcnemar requires equal-length inputs")
    b_only = int(np.sum(~a & b))   # B succeeded, A failed
    a_only = int(np.sum(a & ~b))   # A succeeded, B failed
    discordant = b_only + a_only
    if discordant == 0:
        return McNemarResult(statistic=0.0, p_value=1.0,
                             discordant_b_only=b_only,
                             discordant_a_only=a_only, method="exact")
    if discordant < 25:
        # Exact two-sided binomial on b_only ~ Bin(discordant, 0.5).
        p = float(sp_stats.binomtest(b_only, discordant, 0.5,
                                     alternative="two-sided").pvalue)
        return McNemarResult(statistic=float(b_only), p_value=p,
                             discordant_b_only=b_only,
                             discordant_a_only=a_only, method="exact")
    stat = (abs(b_only - a_only) - 1.0) ** 2 / discordant
    p = float(sp_stats.chi2.sf(stat, df=1))
    return McNemarResult(statistic=float(stat), p_value=p,
                         discordant_b_only=b_only,
                         discordant_a_only=a_only, method="chisq")


def wilcoxon(a: Sequence[float], b: Sequence[float]) -> WilcoxonResult:
    """Wilcoxon signed-rank test on paired continuous scores."""
    arr_a = np.asarray(a, dtype=float)
    arr_b = np.asarray(b, dtype=float)
    if arr_a.size != arr_b.size:
        raise ValueError("wilcoxon requires equal-length inputs")
    diff = arr_a - arr_b
    nonzero = diff[diff != 0]
    if nonzero.size == 0:
        return WilcoxonResult(statistic=0.0, p_value=1.0, n_pairs=int(arr_a.size))
    res = sp_stats.wilcoxon(arr_a, arr_b, zero_method="wilcox")
    return WilcoxonResult(statistic=float(res.statistic),
                          p_value=float(res.pvalue), n_pairs=int(arr_a.size))


def pass_at_k(per_run_success: Sequence[bool]) -> float:
    """``pass@1`` over k independent runs: fraction of runs that succeeded.

    For a single task repeated k times, ``pass@1`` = mean success across the k
    runs. ``per_run_success`` is one bool per run.
    """
    arr = np.asarray(per_run_success, dtype=bool)
    if arr.size == 0:
        return 0.0
    return float(arr.mean())


def pass_pow_k(per_run_success: Sequence[bool]) -> float:
    """``pass^k`` (tau-bench reliability): 1.0 only if ALL k runs succeeded.

    Reliability — the probability that the agent succeeds on every one of k
    independent attempts. Strictly stricter than ``pass@k``.
    """
    arr = np.asarray(per_run_success, dtype=bool)
    if arr.size == 0:
        return 0.0
    return 1.0 if bool(arr.all()) else 0.0


def effect_size_hedges(a: Sequence[float], b: Sequence[float]) -> float:
    """Hedges' g: standardized mean difference between two independent samples."""
    arr_a = np.asarray(a, dtype=float)
    arr_b = np.asarray(b, dtype=float)
    na, nb = arr_a.size, arr_b.size
    if na < 2 or nb < 2:
        return float("nan")
    pooled_sd = math.sqrt(
        ((na - 1) * np.var(arr_a, ddof=1) + (nb - 1) * np.var(arr_b, ddof=1))
        / (na + nb - 2)
    )
    if pooled_sd == 0:
        return 0.0
    correction = 1.0 - 3.0 / (4.0 * (na + nb) - 9.0)
    return float((np.mean(arr_a) - np.mean(arr_b)) / pooled_sd * correction)


@dataclass(frozen=True)
class HolmCorrection:
    """One p-value after Holm-Bonferroni correction."""
    original_p: float
    adjusted_p: float
    reject: bool


def holm_bonferroni(
    pvalues: Sequence[float],
    alpha: float = 0.05,
) -> List[HolmCorrection]:
    """Holm-Bonferroni step-down multiplicity correction."""
    pvs = list(pvalues)
    m = len(pvs)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvs[i])
    out: List[HolmCorrection] = [None] * m  # type: ignore
    prev_adj = 0.0
    for rank, idx in enumerate(order):
        adj = max(prev_adj, min(1.0, pvs[idx] * (m - rank)))
        out[idx] = HolmCorrection(original_p=pvs[idx], adjusted_p=adj,
                                  reject=adj <= alpha)
        prev_adj = adj
    return out


__all__ = [
    "CIMean",
    "McNemarResult",
    "WilcoxonResult",
    "HolmCorrection",
    "bootstrap_ci",
    "paired_mcnemar",
    "wilcoxon",
    "pass_at_k",
    "pass_pow_k",
    "effect_size_hedges",
    "holm_bonferroni",
]