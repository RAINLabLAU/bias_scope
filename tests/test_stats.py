"""Tests for bias_scope.stats.

Each function gets a known-answer test with the expected number derived in the
test body, per PLAN.md Section 1's testing discipline.
"""

import math

import numpy as np
import pytest

from bias_scope.stats import bootstrap_ci, hedges_olkin_ci, permutation_p, wald_ci


class TestWaldCi:
    """Wald interval for a proportion: p ± z·sqrt(p(1−p)/n)."""

    def test_known_answer(self):
        # p = 0.6, n = 100 -> se = sqrt(.6*.4/100) = sqrt(.0024) = 0.0489898...
        # half-width = 1.959964 * 0.0489898 = 0.09601...
        lo, hi = wald_ci(0.6, 100)
        se = math.sqrt(0.6 * 0.4 / 100)
        assert lo == pytest.approx(0.6 - 1.959963984540054 * se, abs=1e-9)
        assert hi == pytest.approx(0.6 + 1.959963984540054 * se, abs=1e-9)

    def test_brackets_the_estimate(self):
        lo, hi = wald_ci(0.3, 50)
        assert lo < 0.3 < hi

    def test_is_clipped_to_the_unit_interval(self):
        # p = 0.02, n = 10 has a lower Wald bound below zero before clipping.
        lo, hi = wald_ci(0.02, 10)
        assert lo == 0.0
        assert hi <= 1.0

    def test_a_larger_sample_gives_a_narrower_interval(self):
        narrow = wald_ci(0.5, 10_000)
        wide = wald_ci(0.5, 100)
        assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])

    def test_degenerate_proportion_has_zero_width(self):
        assert wald_ci(1.0, 100) == (1.0, 1.0)
        assert wald_ci(0.0, 100) == (0.0, 0.0)

    @pytest.mark.parametrize("p,n", [(-0.1, 10), (1.1, 10), (0.5, 0), (0.5, -3)])
    def test_invalid_inputs_raise(self, p, n):
        with pytest.raises(ValueError):
            wald_ci(p, n)


class TestBootstrapCi:
    """Percentile bootstrap over item-level scores."""

    def test_constant_input_gives_a_degenerate_interval(self):
        # Every resample of a constant vector has the same mean.
        lo, hi = bootstrap_ci([2.0] * 20, n_resamples=200, seed=42)
        assert lo == pytest.approx(2.0)
        assert hi == pytest.approx(2.0)

    def test_brackets_the_sample_mean(self):
        rng = np.random.default_rng(0)
        values = rng.normal(5.0, 1.0, size=200).tolist()
        lo, hi = bootstrap_ci(values, seed=42)
        assert lo < float(np.mean(values)) < hi

    def test_is_reproducible_for_a_fixed_seed(self):
        values = list(range(50))
        assert bootstrap_ci(values, seed=7) == bootstrap_ci(values, seed=7)

    def test_different_seeds_give_slightly_different_bounds(self):
        values = list(range(50))
        assert bootstrap_ci(values, seed=1) != bootstrap_ci(values, seed=2)

    def test_covers_the_true_mean_about_95_percent_of_the_time(self):
        """The interval must actually have roughly nominal coverage."""
        rng = np.random.default_rng(20260822)
        covered = 0
        trials = 200
        for i in range(trials):
            sample = rng.normal(0.0, 1.0, size=60).tolist()
            lo, hi = bootstrap_ci(sample, n_resamples=400, seed=i)
            covered += lo <= 0.0 <= hi
        # Binomial(200, 0.95) has sd ~3.1, so allow a generous band.
        assert 0.88 <= covered / trials <= 1.0, f"coverage {covered / trials:.3f}"

    def test_a_custom_statistic_is_honoured(self):
        values = [1.0, 2.0, 3.0, 100.0]
        lo, hi = bootstrap_ci(values, statistic=np.median, seed=42)
        assert lo <= float(np.median(values)) <= hi

    def test_single_item_gives_a_degenerate_interval(self):
        assert bootstrap_ci([3.0], seed=42) == (3.0, 3.0)

    def test_empty_input_raises(self):
        with pytest.raises(ValueError, match="values"):
            bootstrap_ci([])


class TestHedgesOlkinCi:
    """Large-sample interval for a standardised mean difference."""

    def test_known_answer(self):
        # Hedges & Olkin (1985) SE for Cohen's d:
        #   se = sqrt((n1+n2)/(n1*n2) + d^2 / (2*(n1+n2)))
        # d = 1.81, n1 = n2 = 8:
        #   (16)/(64) = 0.25;  1.81^2 / 32 = 3.2761/32 = 0.10237812
        #   se = sqrt(0.35237812) = 0.5936144...
        #   1.96 * se = 1.16348...  -> [0.6465, 2.9735] at z = 1.96
        d, n1, n2 = 1.81, 8, 8
        se = math.sqrt((n1 + n2) / (n1 * n2) + d**2 / (2 * (n1 + n2)))
        assert se == pytest.approx(0.5936144, abs=1e-6)
        lo, hi = hedges_olkin_ci(d, n1, n2, z=1.96)
        assert lo == pytest.approx(d - 1.96 * se, abs=1e-9)
        assert hi == pytest.approx(d + 1.96 * se, abs=1e-9)

    def test_brackets_the_effect_size(self):
        lo, hi = hedges_olkin_ci(0.5, 30, 30)
        assert lo < 0.5 < hi

    def test_larger_groups_give_a_narrower_interval(self):
        narrow = hedges_olkin_ci(0.5, 1000, 1000)
        wide = hedges_olkin_ci(0.5, 10, 10)
        assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])

    def test_is_symmetric_in_the_sign_of_d(self):
        lo_pos, hi_pos = hedges_olkin_ci(1.2, 20, 20)
        lo_neg, hi_neg = hedges_olkin_ci(-1.2, 20, 20)
        assert lo_neg == pytest.approx(-hi_pos)
        assert hi_neg == pytest.approx(-lo_pos)

    @pytest.mark.parametrize("n1,n2", [(0, 10), (10, 0), (-1, 10)])
    def test_invalid_group_sizes_raise(self, n1, n2):
        with pytest.raises(ValueError):
            hedges_olkin_ci(0.5, n1, n2)


class TestPermutationP:
    """Two-sided permutation test on a difference of means."""

    def test_identical_groups_give_p_of_one(self):
        a = b = [1.0, 2.0, 3.0, 4.0]
        assert permutation_p(a, b, n_permutations=500, seed=42) == pytest.approx(1.0)

    def test_well_separated_groups_give_a_small_p(self):
        a = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
        b = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        assert permutation_p(a, b, n_permutations=2000, seed=42) < 0.01

    def test_p_is_never_zero(self):
        """The +1 correction keeps p strictly positive; p = 0 is not a p-value."""
        a = [100.0] * 8
        b = [0.0] * 8
        p = permutation_p(a, b, n_permutations=100, seed=42)
        assert p > 0.0
        assert p == pytest.approx(1 / 101)

    def test_p_is_in_the_unit_interval(self):
        rng = np.random.default_rng(3)
        for seed in range(10):
            a = rng.normal(size=15).tolist()
            b = rng.normal(size=15).tolist()
            assert 0.0 < permutation_p(a, b, n_permutations=200, seed=seed) <= 1.0

    def test_is_reproducible_for_a_fixed_seed(self):
        a, b = [1.0, 5.0, 3.0], [2.0, 8.0, 4.0]
        first = permutation_p(a, b, n_permutations=300, seed=11)
        assert permutation_p(a, b, n_permutations=300, seed=11) == first

    def test_exchanging_the_groups_does_not_change_p(self):
        """Two-sided, so the test is symmetric in its arguments."""
        a, b = [1.0, 2.0, 9.0], [3.0, 4.0, 5.0]
        assert permutation_p(a, b, n_permutations=500, seed=5) == pytest.approx(
            permutation_p(b, a, n_permutations=500, seed=5)
        )

    def test_empty_group_raises(self):
        with pytest.raises(ValueError):
            permutation_p([], [1.0], n_permutations=10)
