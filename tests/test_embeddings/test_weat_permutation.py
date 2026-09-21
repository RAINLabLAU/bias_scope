"""WEAT's permutation test (Caliskan et al. 2017).

The paper defines WEAT as an effect size **and** a one-sided permutation
p-value; v0.1.1 implemented only the effect size. Definition, verbatim:

    Let {(Xi, Yi)}i denote all the partitions of X ∪ Y into two sets of equal
    size. The one-sided p-value of the permutation test is
        Pr_i[ s(Xi, Yi, A, B) > s(X, Y, A, B) ]

where s(X, Y, A, B) = Σ_{x∈X} s(x,A,B) − Σ_{y∈Y} s(y,A,B).

The reference implementation (`sent-bias/sentbias/weat.py:82-152`) enumerates
partitions exactly when there are few enough and samples otherwise.
"""

import math

import numpy as np
import pytest

from bias_scope.embeddings_based import WEAT


def _orthogonal_sets(n=4, dim=8, separation=1.0):
    """X aligned with A, Y aligned with B — maximal separation."""
    rng = np.random.default_rng(0)
    a_axis = np.zeros(dim)
    a_axis[0] = 1.0
    b_axis = np.zeros(dim)
    b_axis[1] = 1.0
    noise = lambda: rng.normal(0, 0.01, dim)  # noqa: E731
    X = np.array([a_axis * separation + noise() for _ in range(n)])
    Y = np.array([b_axis * separation + noise() for _ in range(n)])
    A = np.array([a_axis + noise() for _ in range(n)])
    B = np.array([b_axis + noise() for _ in range(n)])
    return X, Y, A, B


class TestPermutationPValue:
    def test_exact_test_is_used_for_small_sets(self):
        """With |X| = |Y| = 4 there are C(8,4) = 70 partitions; enumerate them."""
        X, Y, A, B = _orthogonal_sets(n=4)
        details = WEAT().evaluate((X, Y), (A, B), return_details=True)
        assert details["p_value_exact"] is True
        assert details["num_partitions"] == math.comb(8, 4)

    def test_maximally_separated_sets_give_zero_strict_p_value(self):
        """The paper's strict test excludes the observed equal partition."""
        X, Y, A, B = _orthogonal_sets(n=4)
        details = WEAT().evaluate((X, Y), (A, B), return_details=True)
        assert details["p_value"] == 0.0

    def test_conservative_tie_policy_counts_equal_partitions(self):
        """May et al.'s later convention uses >= rather than the paper's >."""
        X, Y, A, B = _orthogonal_sets(n=4)
        details = WEAT().evaluate(
            (X, Y), (A, B), return_details=True, tie_policy="conservative"
        )
        assert details["p_value"] == pytest.approx(1 / math.comb(8, 4))

    def test_strict_tie_counterexample_is_zero(self):
        """X=A=e1 and Y=B=e2 has only equality and a smaller partition."""
        X = np.array([[1.0, 0.0]])
        Y = np.array([[0.0, 1.0]])
        strict = WEAT().evaluate((X, Y), (X, Y), return_details=True)
        conservative = WEAT().evaluate(
            (X, Y), (X, Y), return_details=True, tie_policy="conservative"
        )
        assert strict["p_value"] == 0.0
        assert conservative["p_value"] == pytest.approx(0.5)

    def test_identical_target_sets_are_not_significant(self):
        """Property 1 (null): identical target sets show no association.

        The observed statistic is exactly 0. Every positive partition has a
        negative complement; strict ties are excluded, so p <= 0.5. (The
        conservative May et al. convention would include the ties instead.)
        """
        rng = np.random.default_rng(1)
        shared = rng.normal(size=(8, 8))
        X, Y = shared[:4], shared[:4]
        A, B = rng.normal(size=(4, 8)), rng.normal(size=(4, 8))
        details = WEAT().evaluate((X, Y), (A, B), return_details=True)
        assert details["effect_size"] == pytest.approx(0.0, abs=1e-12)
        assert 0.0 <= details["p_value"] <= 0.5

    def test_p_value_is_in_the_unit_interval(self):
        rng = np.random.default_rng(2)
        for _ in range(5):
            X, Y, A, B = (rng.normal(size=(4, 8)) for _ in range(4))
            details = WEAT().evaluate((X, Y), (A, B), return_details=True)
            assert 0.0 <= details["p_value"] <= 1.0

    def test_sampling_is_used_when_there_are_too_many_partitions(self):
        """|X| = |Y| = 12 gives C(24,12) = 2.7M partitions; sample instead."""
        rng = np.random.default_rng(3)
        X, Y, A, B = (rng.normal(size=(12, 8)) for _ in range(4))
        details = WEAT().evaluate((X, Y), (A, B), return_details=True,
                                  n_permutation_samples=500)
        assert details["p_value_exact"] is False
        assert 0.0 < details["p_value"] <= 1.0

    def test_sampled_p_value_is_reproducible(self):
        rng = np.random.default_rng(4)
        X, Y, A, B = (rng.normal(size=(12, 8)) for _ in range(4))
        kwargs = dict(return_details=True, n_permutation_samples=200,
                      permutation_seed=7)
        first = WEAT().evaluate((X, Y), (A, B), **kwargs)["p_value"]
        second = WEAT().evaluate((X, Y), (A, B), **kwargs)["p_value"]
        assert first == second

    def test_run_seed_controls_sampled_permutations(self):
        rng = np.random.default_rng(124)
        X, Y, A, B = (rng.normal(size=(12, 6)) for _ in range(4))
        first = WEAT().run((X, Y), (A, B), seed=1, n_permutation_samples=25)
        second = WEAT().run((X, Y), (A, B), seed=2, n_permutation_samples=25)
        assert first.details["permutation_seed"] == first.protocol["permutation_seed"] == 1
        assert second.details["permutation_seed"] == second.protocol["permutation_seed"] == 2
        assert first.p_value != second.p_value

    def test_explicit_permutation_seed_overrides_run_seed(self):
        rng = np.random.default_rng(124)
        X, Y, A, B = (rng.normal(size=(12, 6)) for _ in range(4))
        first = WEAT().run(
            (X, Y), (A, B), seed=1, permutation_seed=7, n_permutation_samples=25
        )
        second = WEAT().run(
            (X, Y), (A, B), seed=2, permutation_seed=7, n_permutation_samples=25
        )
        assert first.protocol["permutation_seed"] == second.protocol["permutation_seed"] == 7
        assert first.p_value == second.p_value

    @pytest.mark.parametrize("n_samples", [0, -1, 1.5, True, False])
    def test_invalid_permutation_sample_count_is_rejected(self, n_samples):
        rng = np.random.default_rng(6)
        X, Y, A, B = (rng.normal(size=(4, 8)) for _ in range(4))
        with pytest.raises(ValueError, match="positive non-Boolean integer"):
            WEAT().evaluate(
                (X, Y), (A, B), n_permutation_samples=n_samples
            )

    def test_unequal_target_sizes_are_rejected(self):
        """Canonical WEAT requires equal target-set sizes."""
        rng = np.random.default_rng(5)
        X = rng.normal(size=(3, 8))
        Y = rng.normal(size=(5, 8))
        A, B = rng.normal(size=(4, 8)), rng.normal(size=(4, 8))
        with pytest.raises(ValueError, match="equal sizes"):
            WEAT().evaluate((X, Y), (A, B), return_details=True)

    def test_the_effect_size_is_unchanged_by_adding_the_p_value(self):
        """Regression guard: the headline number must not move."""
        rng = np.random.default_rng(42)
        sets = [rng.standard_normal((8, 16)) for _ in range(4)]
        score = WEAT().evaluate((sets[0], sets[1]), (sets[2], sets[3]))
        # The golden value frozen in tests/golden/weat.json.
        assert score == pytest.approx(-0.023887195046478658, abs=1e-12)

    def test_run_carries_the_p_value_onto_the_result(self):
        X, Y, A, B = _orthogonal_sets(n=4)
        result = WEAT().run((X, Y), (A, B))
        assert result.p_value == 0.0


class TestStandardDeviationConvention:
    """RL-008: WEAT uses the sample standard deviation (ddof=1)."""

    def test_effect_size_uses_ddof_one(self):
        """Hand-checkable case: X = {e1}, Y = {e2}, A = {e1}, B = {e2}.

        s(e1) = 1 − 0 = 1, s(e2) = 0 − 1 = −1. Pooled {1, −1}, mean 0.
          ddof=1: var = 2/1 = 2, std = √2  ->  d = 2/√2 = √2 ≈ 1.41421
          ddof=0: var = 2/2 = 1, std = 1   ->  d = 2

        The reference implementation (`sent-bias/sentbias/weat.py:175`) passes
        `ddof=1`, and on Caliskan's own GloVe-840B vectors ddof=1 reproduces the
        published 1.81 to 0.22% while ddof=0 is off by 3.50%.
        """
        X = np.array([[1.0, 0.0]])
        Y = np.array([[0.0, 1.0]])
        assert WEAT().evaluate((X, Y), (X, Y)) == pytest.approx(math.sqrt(2))
