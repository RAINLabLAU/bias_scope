"""Confidence intervals and significance tests (PLAN.md Section 5.3).

Four short functions, each stating its formula. `run()` in `base.py` picks one
per metric: bootstrap for item-level scores, Wald for proportions,
Hedges-Olkin for effect sizes, and a permutation p-value for WEAT-family tests.

No result is ever reported without an interval where one is definable, and no
interval is ever widened or narrowed to make a reproduction pass (Section 1).
"""

from __future__ import annotations

import math
from typing import Callable, List, Sequence, Tuple

import numpy as np

# Two-sided normal quantile for a 95% interval. Spelled out rather than pulled
# from scipy so the core install stays light.
Z_95 = 1.959963984540054

DEFAULT_RESAMPLES = 2000
DEFAULT_PERMUTATIONS = 10_000


def wald_ci(p: float, n: int, z: float = Z_95) -> Tuple[float, float]:
    """
    Wald interval for a proportion.

    Formula: ``p ± z · sqrt(p(1−p)/n)``, clipped to [0, 1].

    The Wald interval is the one most bias papers report for proportions
    (CrowS-Pairs, HONEST, BBQ), which is why it is here. It is known to
    under-cover near 0 and 1; `bootstrap_ci` is the library default and this is
    reported alongside (PLAN.md Section 12).

    Parameters
    ----------
    p : float
        Observed proportion, in [0, 1].
    n : int
        Number of items the proportion was computed over. Must be positive.
    z : float
        Normal quantile. Default is the 95% two-sided value.

    Returns
    -------
    (float, float)
        Lower and upper bound.

    Examples
    --------
    >>> lo, hi = wald_ci(0.5, 100)
    >>> round(lo, 4), round(hi, 4)
    (0.402, 0.598)
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be a proportion in [0, 1], got {p}")
    if n <= 0:
        raise ValueError(f"n must be a positive integer, got {n}")

    half_width = z * math.sqrt(p * (1.0 - p) / n)
    return (max(0.0, p - half_width), min(1.0, p + half_width))


def bootstrap_ci(
    values: Sequence[float],
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_resamples: int = DEFAULT_RESAMPLES,
    confidence: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    """
    Percentile bootstrap interval over item-level scores.

    Resamples `values` with replacement `n_resamples` times, applies
    `statistic` to each resample, and returns the empirical
    ``[(1−c)/2, (1+c)/2]`` percentiles of that distribution.

    This is the library default for metrics that expose per-item scores
    (PLAN.md Section 12).

    Parameters
    ----------
    values : sequence of float
        Item-level scores. Must be non-empty.
    statistic : callable
        Applied to each resample. Default `np.mean`.
    n_resamples : int
        Number of bootstrap resamples.
    confidence : float
        Coverage, in (0, 1). Default 0.95.
    seed : int
        Seed for the resampling RNG, so the interval is reproducible.

    Returns
    -------
    (float, float)
        Lower and upper percentile bound.

    Examples
    --------
    >>> bootstrap_ci([1.0, 1.0, 1.0])
    (1.0, 1.0)
    """
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        raise ValueError("values must contain at least one score")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    if n_resamples <= 0:
        raise ValueError(f"n_resamples must be positive, got {n_resamples}")
    if not np.all(np.isfinite(array)):
        raise ValueError("values must all be finite")

    if array.size == 1:
        only = float(array[0])
        return (only, only)

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(n_resamples, array.size))
    estimates = np.array([statistic(array[row]) for row in indices], dtype=float)

    tail = (1.0 - confidence) / 2.0
    lower, upper = np.percentile(estimates, [100 * tail, 100 * (1.0 - tail)])
    return (float(lower), float(upper))


def hedges_olkin_ci(
    d: float, n1: int, n2: int, z: float = Z_95
) -> Tuple[float, float]:
    """
    Large-sample interval for a standardised mean difference.

    Formula (Hedges & Olkin 1985; Borenstein et al. 2009 eq. 4.20): the
    standard error of Cohen's *d* is

        se = sqrt( (n1 + n2) / (n1 · n2)  +  d² / (2 · (n1 + n2)) )

    and the interval is ``d ± z · se``.

    **A second form appears in the literature**, with ``2·(n1 + n2 − 2)`` in the
    second term. Both are in circulation, and they are not interchangeable at
    small n: for d = 1.6938, n1 = n2 = 8, this form gives [0.5515, 2.8361] and
    the other [0.5302, 2.8574]. `scripts/experiments/finalize_emnlp.py` uses the
    ``− 2`` form, which is where the CIs currently in `results/emnlp/` come
    from, so the two disagree. Tracked as REVIEW_LATER RL-016 and **not**
    reconciled unilaterally, because doing so would move published numbers.

    Used for WEAT/SEAT/CEAT effect sizes. **Caveat**: with the n = 8 per group
    of Caliskan's own word sets this is a formal computation more than a real
    statistical claim; report it with the sample size beside it.

    Parameters
    ----------
    d : float
        Effect size.
    n1, n2 : int
        Group sizes. Both must be positive.
    z : float
        Normal quantile. Default is the 95% two-sided value.

    Returns
    -------
    (float, float)
        Lower and upper bound. Unbounded — effect sizes are not clipped.

    Examples
    --------
    >>> lo, hi = hedges_olkin_ci(0.0, 50, 50)
    >>> round(lo, 4), round(hi, 4)
    (-0.392, 0.392)
    """
    if n1 <= 0 or n2 <= 0:
        raise ValueError(f"n1 and n2 must be positive integers, got {n1} and {n2}")
    if not math.isfinite(d):
        raise ValueError(f"d must be finite, got {d}")

    total = n1 + n2
    se = math.sqrt(total / (n1 * n2) + d**2 / (2.0 * total))
    return (d - z * se, d + z * se)


def permutation_p(
    group_a: Sequence[float],
    group_b: Sequence[float],
    n_permutations: int = DEFAULT_PERMUTATIONS,
    seed: int = 42,
) -> float:
    """
    Two-sided permutation p-value for a difference of means.

    Pools the two groups, reshuffles, splits at the original sizes, and counts
    how often the absolute difference of means is at least as large as the
    observed one. Formula:

        p = (1 + #{ |Δ*| ≥ |Δ_obs| }) / (1 + n_permutations)

    The ``+1`` in both places is the standard correction: a p-value of exactly
    zero is not a valid estimate, only an upper bound set by the number of
    permutations run.

    This is the test WEAT and SEAT define for their effect sizes.

    Parameters
    ----------
    group_a, group_b : sequence of float
        The two samples. Both must be non-empty.
    n_permutations : int
        Number of random reshuffles. Larger gives a finer resolution floor.
    seed : int
        Seed, so the p-value is reproducible.

    Returns
    -------
    float
        p-value in ``(0, 1]``.

    Examples
    --------
    >>> permutation_p([1.0, 2.0], [1.0, 2.0], n_permutations=99, seed=0)
    1.0
    """
    a = np.asarray(list(group_a), dtype=float)
    b = np.asarray(list(group_b), dtype=float)
    if a.size == 0 or b.size == 0:
        raise ValueError("group_a and group_b must both be non-empty")
    if n_permutations <= 0:
        raise ValueError(f"n_permutations must be positive, got {n_permutations}")

    observed = abs(float(a.mean()) - float(b.mean()))
    pooled = np.concatenate([a, b])
    split = a.size

    rng = np.random.default_rng(seed)
    at_least_as_extreme = 0
    for _ in range(n_permutations):
        shuffled = rng.permutation(pooled)
        delta = abs(float(shuffled[:split].mean()) - float(shuffled[split:].mean()))
        # Ties count as "at least as extreme"; this is what keeps identical
        # groups at p = 1.0 rather than p = 1/(n+1).
        if delta >= observed - 1e-12:
            at_least_as_extreme += 1

    return (1 + at_least_as_extreme) / (1 + n_permutations)


def wasserstein_1(sample_a: Sequence[float], sample_b: Sequence[float]) -> float:
    """
    Wasserstein-1 distance between two one-dimensional empirical distributions.

    Formula: ``W1(P, Q) = ∫ |F_P(t) − F_Q(t)| dt``, which for samples equals
    the mean absolute difference of their matched quantiles. Equivalently, in
    the form Huang et al. 2020 give (their eq. 1),

        W1 = E_{τ~U[0,1]} | p(S(x) > τ) − p(S(x̃) > τ) |

    Implemented by sorting both samples onto a common quantile grid, which is
    exact for equal sizes and the standard estimator otherwise. Requires no
    assumption about distribution shape — which the paper notes is the point of
    using W1 rather than comparing means.

    Parameters
    ----------
    sample_a, sample_b : sequence of float
        The two samples. Both must be non-empty.

    Returns
    -------
    float
        Distance, ``>= 0``; zero exactly when the two empirical distributions
        coincide.

    Examples
    --------
    >>> wasserstein_1([0.0, 1.0], [0.0, 1.0])
    0.0
    >>> wasserstein_1([0.0, 0.0], [1.0, 1.0])
    1.0
    >>> round(wasserstein_1([0.0, 1.0], [0.0, 0.0]), 4)
    0.5
    """
    a = np.sort(np.asarray(list(sample_a), dtype=float))
    b = np.sort(np.asarray(list(sample_b), dtype=float))
    if a.size == 0 or b.size == 0:
        raise ValueError("sample_a and sample_b must both be non-empty")
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        raise ValueError("samples must be finite")

    if a.size == b.size:
        # Equal sizes: matched quantiles, which is exact and cheaper.
        return float(np.abs(a - b).mean())

    # Unequal sizes: integrate |F_a − F_b| over the merged support directly.
    merged = np.sort(np.concatenate([a, b]))
    widths = np.diff(merged)
    cdf_a = np.searchsorted(a, merged[:-1], side="right") / a.size
    cdf_b = np.searchsorted(b, merged[:-1], side="right") / b.size
    return float(np.sum(np.abs(cdf_a - cdf_b) * widths))


__all__: List[str] = [
    "wald_ci",
    "bootstrap_ci",
    "hedges_olkin_ci",
    "permutation_p",
    "wasserstein_1",
]
