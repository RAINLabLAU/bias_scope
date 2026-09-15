"""Tests for SEAT (Sentence Encoder Association Test).

SEAT (May et al. 2019, NAACL) is WEAT applied to sentence vectors. It inherits
WEAT's effect size unchanged, but the paper's Appendix A deliberately departs
from Caliskan on the permutation test:

  * the inequality is non-strict — ``Pr[s(Xi,Yi,A,B) >= s(X,Y,A,B)]`` — because
    "the equality has positive probability" in the nonparametric version;
  * the sampled estimate draws 99,999 partitions and counts one extra as a hit,
    so "we can never observe a p-value less than 1e-5".

BiasScope maps the first to ``tie_policy="conservative"`` and the second to
``n_permutation_samples=100_000`` (WEAT's conservative branch already adds the
observed partition), and SEAT sets both as defaults.
"""

import math

import numpy as np
import pytest
import torch

from bias_scope.embeddings_based import SEAT, WEAT


def _max_separated(n=4, dim=16, seed=0):
    """X aligned with A, Y aligned with B: the observed statistic is the max."""
    rng = np.random.default_rng(seed)
    a = np.zeros(dim)
    a[0] = 1.0
    b = np.zeros(dim)
    b[1] = 1.0
    jitter = lambda: rng.normal(0, 1e-3, dim)  # noqa: E731
    X = np.array([a + jitter() for _ in range(n)])
    Y = np.array([b + jitter() for _ in range(n)])
    A = np.array([a + jitter() for _ in range(n)])
    B = np.array([b + jitter() for _ in range(n)])
    return X, Y, A, B


class TestSEAT:
    """Test Sentence Encoder Association Test."""

    def test_basic_functionality(self):
        rng = np.random.default_rng(0)
        target1, target2 = rng.standard_normal((4, 768)), rng.standard_normal((4, 768))
        attr1, attr2 = rng.standard_normal((3, 768)), rng.standard_normal((3, 768))

        score = SEAT().evaluate((target1, target2), (attr1, attr2))

        assert isinstance(score, float)
        assert not np.isnan(score)

    def test_effect_size_equals_weat(self):
        """The magnitude (SEAT's headline number) is WEAT's, unchanged."""
        rng = np.random.default_rng(1)
        target1, target2 = rng.standard_normal((4, 768)), rng.standard_normal((4, 768))
        attr1, attr2 = rng.standard_normal((3, 768)), rng.standard_normal((3, 768))

        assert SEAT().evaluate((target1, target2), (attr1, attr2)) == WEAT().evaluate(
            (target1, target2), (attr1, attr2)
        )

    def test_sentence_embeddings(self):
        rng = np.random.default_rng(2)
        target1, target2 = rng.standard_normal((5, 768)), rng.standard_normal((5, 768))
        attr1, attr2 = rng.standard_normal((3, 768)), rng.standard_normal((3, 768))

        score = SEAT().evaluate((target1, target2), (attr1, attr2))
        assert isinstance(score, float)
        assert not np.isnan(score)

    def test_seat_different_group_sizes_are_rejected(self):
        """SEAT delegates the canonical equal-target requirement to WEAT."""
        rng = np.random.default_rng(3)
        target1, target2 = rng.standard_normal((3, 768)), rng.standard_normal((10, 768))
        attr1, attr2 = rng.standard_normal((5, 768)), rng.standard_normal((2, 768))

        with pytest.raises(ValueError, match="equal sizes"):
            SEAT().evaluate((target1, target2), (attr1, attr2))

    def test_with_torch_tensors(self):
        torch.manual_seed(0)
        target1, target2 = torch.randn(4, 768), torch.randn(4, 768)
        attr1, attr2 = torch.randn(3, 768), torch.randn(3, 768)

        score = SEAT().evaluate((target1, target2), (attr1, attr2))
        assert isinstance(score, float)

    def test_rejects_invalid_pooling(self):
        with pytest.raises(ValueError, match="pooling must be"):
            SEAT(pooling="banana")


class TestSeatPermutationConvention:
    """May et al. 2019, Appendix A."""

    def test_pvalue_counts_the_observed_partition_unlike_strict_weat(self):
        """SEAT uses ``>=`` (App. A); strict WEAT uses Caliskan's ``>``.

        With X aligned to A and Y to B, the observed partition is the unique
        maximiser of s(Xi,Yi,A,B). The strict test excludes it (p = 0); SEAT's
        non-strict test includes it, giving exactly 1 / C(2n, n).
        """
        X, Y, A, B = _max_separated(n=4)
        seat = SEAT().evaluate((X, Y), (A, B), return_details=True)
        weat_strict = WEAT().evaluate((X, Y), (A, B), return_details=True)

        assert weat_strict["p_value"] == 0.0
        assert seat["p_value"] == pytest.approx(1 / math.comb(8, 4))
        assert seat["tie_policy"] == "conservative"
        assert seat["p_value_exact"] is True

    def test_default_sample_count_matches_the_paper(self):
        """App. A: 99,999 sampled + 1 hallucinated => floor of 1 / 100,000."""
        import inspect

        default = inspect.signature(SEAT().evaluate).parameters[
            "n_permutation_samples"
        ].default
        assert default == 100_000

    def test_sampled_pvalue_never_reaches_zero(self):
        """The conservative branch always counts the observed partition."""
        rng = np.random.default_rng(4)
        X, Y, A, B = (rng.normal(size=(12, 8)) for _ in range(4))  # C(24,12) -> sampled
        details = SEAT().evaluate(
            (X, Y), (A, B), return_details=True, n_permutation_samples=200
        )
        assert details["p_value_exact"] is False
        assert details["p_value"] >= 1 / 200
        assert details["p_value"] > 0.0

    def test_strict_convention_is_still_selectable(self):
        X, Y, A, B = _max_separated(n=4)
        strict = SEAT().evaluate(
            (X, Y), (A, B), return_details=True, tie_policy="strict"
        )
        assert strict["p_value"] == 0.0

    def test_details_do_not_leak_the_weat_score_key(self):
        X, Y, A, B = _max_separated(n=3)
        details = SEAT().evaluate((X, Y), (A, B), return_details=True)
        assert "weat_score" not in details
        assert details["seat_score"] == details["effect_size"]


class TestSeatRun:
    def test_run_carries_the_pvalue(self):
        X, Y, A, B = _max_separated(n=4)
        result = SEAT().run((X, Y), (A, B))
        assert result.p_value == pytest.approx(1 / math.comb(8, 4))
        assert result.score == pytest.approx(SEAT().evaluate((X, Y), (A, B)))

    def test_run_threads_and_records_the_permutation_seed(self):
        """``run(seed=s)`` must reach the permutation RNG (as ``WEAT.run`` does)
        and record it in the protocol; otherwise sampled p-values ignore it."""
        rng = np.random.default_rng(6)
        X, Y, A, B = (rng.normal(size=(12, 6)) for _ in range(4))

        result = SEAT().run((X, Y), (A, B), seed=3, n_permutation_samples=200)
        direct = SEAT().evaluate(
            (X, Y), (A, B), return_details=True,
            permutation_seed=3, n_permutation_samples=200,
        )
        assert result.protocol["permutation_seed"] == 3
        assert result.p_value == direct["p_value"]

    def test_explicit_permutation_seed_overrides_run_seed(self):
        rng = np.random.default_rng(6)
        X, Y, A, B = (rng.normal(size=(12, 6)) for _ in range(4))
        first = SEAT().run(
            (X, Y), (A, B), seed=1, permutation_seed=7, n_permutation_samples=200
        )
        second = SEAT().run(
            (X, Y), (A, B), seed=2, permutation_seed=7, n_permutation_samples=200
        )
        assert first.protocol["permutation_seed"] == second.protocol["permutation_seed"] == 7
        assert first.p_value == second.p_value
