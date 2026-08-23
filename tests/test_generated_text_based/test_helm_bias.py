"""HELM's demographic-representation and stereotypical-association metrics.

Both are `TVD(group distribution, uniform)` with per-group word-list-size
normalisation (Liang et al. 2022; `helm/benchmark/metrics/bias_metrics.py`).
Expected values are derived in each test body.
"""

import pytest

from bias_scope.generated_text_based import (
    DemographicRepresentation,
    StereotypeRuleHitRate,
    StereotypicalAssociations,
)
from bias_scope.generated_text_based._helm import group_counts_to_bias

GENDER = {"male": ["he", "him", "man"], "female": ["she", "her", "woman"]}


class TestGroupCountsToBias:
    def test_equal_representation_is_zero(self):
        assert group_counts_to_bias([5, 5], [1, 1]) == 0.0

    def test_one_group_only_gives_the_two_group_maximum(self):
        """p = [1, 0], u = [.5, .5] -> L1 = 1 -> TVD = 0.5."""
        assert group_counts_to_bias([7, 0], [1, 1]) == pytest.approx(0.5)

    def test_word_list_size_normalisation_is_applied(self):
        """The step most likely to be dropped in a reimplementation.

        Raw counts [10, 5] look 2:1 skewed. But the first group's list has 10
        words and the second's has 5, so normalised counts are [1.0, 1.0] and
        the distribution is uniform: TVD = 0.
        """
        assert group_counts_to_bias([10, 5], [10, 5]) == pytest.approx(0.0)
        # Without the normalisation this would be 2/3 vs 1/3 -> TVD = 1/6.
        assert group_counts_to_bias([10, 5], [1, 1]) == pytest.approx(1 / 6)

    def test_three_groups_hand_computed(self):
        """counts [6,3,3] / sizes [1,1,1] -> p = [.5,.25,.25], u = 1/3.

        L1 = |1/3-1/2| + |1/3-1/4| + |1/3-1/4| = 1/6 + 1/12 + 1/12 = 1/3
        TVD = 1/6
        """
        assert group_counts_to_bias([6, 3, 3], [1, 1, 1]) == pytest.approx(1 / 6)

    def test_no_mentions_returns_none_not_zero(self):
        """HELM drops the instance; reporting 0 would claim 'unbiased'."""
        assert group_counts_to_bias([0, 0, 0], [1, 1, 1]) is None

    def test_score_is_in_the_unit_interval(self):
        for counts in ([1, 0], [3, 1], [100, 1], [1, 1, 1, 50]):
            value = group_counts_to_bias(counts, [1] * len(counts))
            assert 0.0 <= value < 1.0

    @pytest.mark.parametrize("counts,sizes", [([1], []), ([1, 2], [1]), ([], [])])
    def test_mismatched_or_empty_inputs_raise(self, counts, sizes):
        with pytest.raises(ValueError):
            group_counts_to_bias(counts, sizes)

    def test_a_zero_sized_word_list_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            group_counts_to_bias([1, 1], [1, 0])


class TestDemographicRepresentation:
    def test_balanced_generations_score_zero(self):
        result = DemographicRepresentation().evaluate(
            generations=["he saw her", "she saw him"],
            group_lexicons=GENDER,
            return_details=True,
        )
        assert result["bias_score"] == pytest.approx(0.0)

    def test_single_group_generations_score_the_maximum(self):
        result = DemographicRepresentation().evaluate(
            generations=["he saw him and the man"],
            group_lexicons=GENDER,
            return_details=True,
        )
        assert result["bias_score"] == pytest.approx(0.5)

    def test_hand_computed_skew(self):
        """3 male tokens, 1 female; both lists have 3 words.

        normalised = [3/3, 1/3] = [1, 1/3]; p = [0.75, 0.25]
        L1 = |0.5-0.75| + |0.5-0.25| = 0.5  ->  TVD = 0.25
        """
        result = DemographicRepresentation().evaluate(
            generations=["he him man she"],
            group_lexicons=GENDER,
            return_details=True,
        )
        assert result["bias_score"] == pytest.approx(0.25)

    def test_entropy_and_gini_are_kept_in_details(self):
        """PLAN.md 5.2: keep the v0.1.1 diagnostics, just not as the score."""
        result = DemographicRepresentation().evaluate(
            generations=["he saw her"], group_lexicons=GENDER, return_details=True,
        )
        assert "entropy" in result["diversity"]
        assert "gini_impurity" in result["diversity"]

    def test_no_group_mentions_is_reported_not_scored_as_unbiased(self):
        result = DemographicRepresentation().evaluate(
            generations=["the weather is fine"],
            group_lexicons=GENDER,
            return_details=True,
        )
        assert result["bias_score"] is None
        assert result["undefined_reason"]

    def test_run_guards_reject_an_undefined_score(self):
        """A None score must not reach a BiasResult as a number."""
        from bias_scope.base import BiasScopeError

        with pytest.raises(BiasScopeError):
            DemographicRepresentation().run(
                generations=["nothing here"], group_lexicons=GENDER,
            )


class TestStereotypicalAssociations:
    def test_balanced_co_occurrence_scores_zero(self):
        """Each target co-occurs equally with both groups."""
        result = StereotypicalAssociations().evaluate(
            generations=["he is smart", "she is smart"],
            group_lexicons=GENDER,
            target_words=["smart"],
            return_details=True,
        )
        assert result["bias_score"] == pytest.approx(0.0)

    def test_one_sided_co_occurrence_scores_the_maximum(self):
        result = StereotypicalAssociations().evaluate(
            generations=["he is smart", "he is smart"],
            group_lexicons=GENDER,
            target_words=["smart"],
            return_details=True,
        )
        assert result["bias_score"] == pytest.approx(0.5)

    def test_co_occurrence_is_a_within_text_product(self):
        """HELM multiplies counts within a text (`bias_metrics.py:169-172`).

        "he him smart" -> 2 male tokens x 1 target = 2, not 1.
        Female count is 0, so p = [1, 0] and TVD = 0.5 either way; the count
        itself is asserted so the product rule is pinned.
        """
        result = StereotypicalAssociations().evaluate(
            generations=["he him smart"],
            group_lexicons=GENDER,
            target_words=["smart"],
            return_details=True,
        )
        assert result["co_occurrences"]["smart"]["male"] == 2

    def test_the_score_is_a_mean_over_target_words(self):
        """One target perfectly skewed, one perfectly balanced -> mean of 0.5 and 0."""
        result = StereotypicalAssociations().evaluate(
            generations=["he is smart", "he is kind", "she is kind"],
            group_lexicons=GENDER,
            target_words=["smart", "kind"],
            return_details=True,
        )
        assert result["per_target"]["smart"] == pytest.approx(0.5)
        assert result["per_target"]["kind"] == pytest.approx(0.0)
        assert result["bias_score"] == pytest.approx(0.25)

    def test_targets_with_no_co_occurrence_are_dropped_not_zeroed(self):
        """HELM filters None scores before the mean (`bias_metrics.py:180`)."""
        result = StereotypicalAssociations().evaluate(
            generations=["he is smart"],
            group_lexicons=GENDER,
            target_words=["smart", "never_appears"],
            return_details=True,
        )
        assert "never_appears" not in result["per_target"]
        assert result["bias_score"] == pytest.approx(0.5)

    def test_all_targets_absent_gives_an_undefined_score(self):
        result = StereotypicalAssociations().evaluate(
            generations=["nothing relevant"],
            group_lexicons=GENDER,
            target_words=["smart"],
            return_details=True,
        )
        assert result["bias_score"] is None

    def test_text_order_does_not_change_the_score(self):
        """Property 4 over generations."""
        kwargs = dict(group_lexicons=GENDER, target_words=["smart", "kind"],
                      return_details=True)
        forward = StereotypicalAssociations().evaluate(
            generations=["he is smart", "she is kind"], **kwargs)
        reverse = StereotypicalAssociations().evaluate(
            generations=["she is kind", "he is smart"], **kwargs)
        assert forward["bias_score"] == pytest.approx(reverse["bias_score"])


class TestStereotypeRuleHitRate:
    """The v0.1.1 rule matcher, preserved under its own name."""

    def test_it_still_reports_hit_rates(self):
        result = StereotypeRuleHitRate().evaluate(
            generations=["the nurse was gentle", "the doctor was firm"],
            stereotype_rules=[{"name": "nurse_gentle",
                               "group_terms": ["nurse"],
                               "attribute_terms": ["gentle"]}],
            return_details=True,
        )
        assert result["overall"]["any_hit_generations"] == 1

    def test_it_is_registered_as_original(self):
        info = StereotypeRuleHitRate.info
        assert info.fidelity == "original"
        assert "HELM" not in info.reference
