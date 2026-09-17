"""Tests for BiasMetric.run() and its runtime guards (PLAN.md 5.3, Section 1).

The guards are the point of this file. Section 1 says a guard failure "must
never be silenced", so each one gets a test that proves it fires.
"""

import math
from dataclasses import replace

import pytest

from bias_scope.base import BiasScopeError, GeneratedTextMetric
from bias_scope.metadata import MetricInfo
from bias_scope.result import BiasResult, from_dict

INFO = MetricInfo(
    name="Fake",
    family="generated_text",
    access=("completions",),
    neutral_value=0.5,
    direction="higher_more_biased",
    value_range=(0.0, 1.0),
    fidelity="faithful",
    reference="none — test fixture",
)


class FakeMetric(GeneratedTextMetric):
    """A metric with item-level scores, for exercising run()."""

    info = INFO

    def __init__(self, per_item=None, score=None, extra=None):
        self._per_item = [0.1, 0.9, 0.5, 0.7] if per_item is None else per_item
        self._score = score
        self._extra = extra or {}

    def evaluate(self, return_details: bool = False):
        score = (
            self._score
            if self._score is not None
            else sum(self._per_item) / len(self._per_item)
        )
        if not return_details:
            return score
        return {"bias_score": score, "per_item": list(self._per_item), **self._extra}


class TestRunHappyPath:
    def test_returns_a_biasresult(self):
        result = FakeMetric().run()
        assert isinstance(result, BiasResult)
        assert result.metric == "FakeMetric"

    def test_score_matches_evaluate(self):
        metric = FakeMetric()
        assert metric.run().score == pytest.approx(metric.evaluate())

    def test_n_is_the_number_of_items_scored(self):
        assert FakeMetric(per_item=[0.1, 0.2, 0.3]).run().n == 3

    def test_per_item_is_carried_through(self):
        assert FakeMetric(per_item=[0.2, 0.4]).run().per_item == [0.2, 0.4]

    def test_bootstrap_is_the_default_and_brackets_the_score(self):
        result = FakeMetric().run()
        assert result.ci_method == "bootstrap"
        assert result.ci[0] <= result.score <= result.ci[1]

    def test_wald_is_available_for_proportions(self):
        result = FakeMetric().run(ci="wald")
        assert result.ci_method == "wald"
        assert result.ci[0] <= result.score <= result.ci[1]

    def test_ci_none_gives_no_interval(self):
        result = FakeMetric().run(ci="none")
        assert result.ci is None
        assert result.ci_method == "none"

    def test_protocol_records_the_seed_and_the_metric(self):
        protocol = FakeMetric().run(seed=7).protocol
        assert protocol["seed"] == 7
        assert protocol["metric"] == "FakeMetric"
        assert len(protocol["hash"]) == 12

    def test_protocol_kwargs_reach_the_protocol(self):
        result = FakeMetric().run(protocol_kwargs={"model_id": "gpt2", "dtype": "bf16"})
        assert result.protocol["model_id"] == "gpt2"
        assert result.protocol["dtype"] == "bf16"

    def test_run_is_reproducible_for_a_fixed_seed(self):
        assert FakeMetric().run(seed=3).ci == FakeMetric().run(seed=3).ci

    def test_breakdown_is_picked_up(self):
        result = FakeMetric(extra={"breakdown": {"male": 0.8, "female": 0.2}}).run()
        assert result.breakdown == {"male": 0.8, "female": 0.2}

    def test_details_are_preserved(self):
        result = FakeMetric(extra={"tie_rate": 0.25}).run()
        assert result.details["tie_rate"] == 0.25

    def test_to_dict_round_trips(self):
        original = FakeMetric().run()
        restored = from_dict(original.to_dict())
        assert restored.score == original.score
        assert restored.n == original.n
        assert restored.ci == original.ci
        assert restored.info == original.info

    def test_normalized_deviation_is_signed_from_neutral(self):
        # neutral 0.5, range [0, 1], higher_more_biased -> scale = 0.5
        assert FakeMetric(score=0.75).run().normalized_deviation() == pytest.approx(0.5)
        assert FakeMetric(score=0.5).run().normalized_deviation() == pytest.approx(0.0)
        assert FakeMetric(score=0.25).run().normalized_deviation() == pytest.approx(-0.5)

    def test_repr_names_the_metric_and_fidelity(self):
        text = repr(FakeMetric().run())
        assert "FakeMetric" in text
        assert "faithful" in text


class TestRuntimeGuards:
    """Section 1: a guard failure raises BiasScopeError naming the metric."""

    def test_score_above_value_range_raises(self):
        with pytest.raises(BiasScopeError, match="outside the declared value_range"):
            FakeMetric(score=1.5).run()

    def test_score_below_value_range_raises(self):
        with pytest.raises(BiasScopeError, match="outside the declared value_range"):
            FakeMetric(score=-0.2).run()

    def test_non_finite_score_raises(self):
        with pytest.raises(BiasScopeError, match="not finite"):
            FakeMetric(score=float("nan")).run()

    def test_infinite_score_raises(self):
        with pytest.raises(BiasScopeError, match="not finite"):
            FakeMetric(score=float("inf")).run()

    def test_zero_items_raises(self):
        class NoItems(FakeMetric):
            def evaluate(self, return_details: bool = False):
                return {"bias_score": 0.5} if return_details else 0.5

        with pytest.raises(BiasScopeError, match="n must be positive"):
            NoItems().run()

    def test_the_error_names_the_metric(self):
        with pytest.raises(BiasScopeError, match="FakeMetric"):
            FakeMetric(score=99.0).run()

    def test_a_ci_that_does_not_bracket_the_score_raises(self):
        class BadInterval(FakeMetric):
            def _interval(self, score, per_item, n, ci, seed):
                return (0.9, 0.95), "bootstrap", None

        with pytest.raises(BiasScopeError, match="does not bracket"):
            BadInterval(score=0.1).run()

    def test_a_result_with_no_headline_score_raises(self):
        class Unscored(FakeMetric):
            def evaluate(self, return_details: bool = False):
                return {"something_else": 1.0, "per_item": [1.0]}

        with pytest.raises(BiasScopeError, match="headline score"):
            Unscored().run()

    def test_a_non_numeric_non_dict_result_raises(self):
        class Weird(FakeMetric):
            def evaluate(self, return_details: bool = False):
                return "not a score"

        with pytest.raises(BiasScopeError, match="float or a dict"):
            Weird().run()


class TestMetricsWithoutInfo:
    """A metric that has not declared MetricInfo is reported, not guessed at."""

    def test_placeholder_info_is_unaudited(self):
        class Bare(GeneratedTextMetric):
            def evaluate(self, return_details: bool = False):
                return {"bias_score": 0.5, "per_item": [0.5]} if return_details else 0.5

        result = Bare().run()
        assert result.info.fidelity == "unaudited"
        assert "no MetricInfo yet" in result.info.deviation_note

    def test_placeholder_range_is_unbounded_so_no_guard_false_positive(self):
        class Bare(GeneratedTextMetric):
            def evaluate(self, return_details: bool = False):
                return {"bias_score": 1e9, "per_item": [1e9]} if return_details else 1e9

        assert math.isfinite(Bare().run().score)


class TestDegenerateIntervals:
    """A model whose per-item scores are all equal must not be rejected.

    The bootstrap interval collapses to [v, v] and the score can differ from
    both endpoints by an ULP purely from summation order, which the strict
    bracketing guard read as a failure. The slack is relative and tiny — a real
    gap still raises.
    """

    def test_a_perfectly_consistent_metric_passes_the_guard(self):
        from bias_scope.prompts_based.first_person_fairness import FirstPersonFairness

        entries = [{"prompt": f"p{i}", "response_a": "a", "response_b": "b"}
                   for i in range(6)]
        result = FirstPersonFairness().run(
            pairs=entries,
            rate_fn=lambda p: ({"A": 0.8, "C": 0.2} if "Response 1: a" in p
                               else {"B": 0.8, "C": 0.2}),
        )
        assert result.score == pytest.approx(0.8)
        assert result.ci == pytest.approx((0.8, 0.8))

    def test_a_real_bracketing_failure_still_raises(self):
        """The slack is 1e-9 relative; a gap of 0.1 is not float noise."""
        result = FakeMetric().run()
        broken = replace(result, score=result.ci[1] + 0.1)
        with pytest.raises(BiasScopeError, match="does not bracket"):
            GeneratedTextMetric._check_guards(broken)


class TestMetricNamedScoreKey:
    """Some metrics name their headline number after themselves rather than
    using one of the four documented keys: CrowS-Pairs returns
    "crows_pairs_score", AUL "aul_score", AULA "aula_score", CEAT
    "ceat_score". Until run() accepts that, BiasSuite records all four as
    skipped and they are unreachable through the suite and the agent, while
    still working when evaluate() is called directly (REVIEW_LATER RL-041).
    """

    def test_a_single_name_suffixed_score_key_is_accepted(self):
        class NamedScore(FakeMetric):
            def evaluate(self, return_details: bool = False):
                return {"crows_pairs_score": 0.62, "num_pairs": 4, "per_item": [0.62]}

        assert NamedScore().run().score == pytest.approx(0.62)

    def test_a_documented_key_still_wins_over_a_suffixed_one(self):
        """The four documented keys stay authoritative; the suffix is a fallback."""

        class Both(FakeMetric):
            def evaluate(self, return_details: bool = False):
                return {"aul_score": 0.9, "bias_score": 0.1, "per_item": [0.1]}

        assert Both().run().score == pytest.approx(0.1)

    def test_two_suffixed_keys_are_ambiguous_and_still_raise(self):
        """Guessing between two candidates would be exactly the fabrication
        PLAN.md Section 1 forbids."""

        class Ambiguous(FakeMetric):
            def evaluate(self, return_details: bool = False):
                return {"a_score": 0.1, "b_score": 0.2, "per_item": [0.1]}

        with pytest.raises(BiasScopeError, match="headline score"):
            Ambiguous().run()

    def test_a_non_numeric_suffixed_key_is_not_mistaken_for_a_score(self):
        class Textual(FakeMetric):
            def evaluate(self, return_details: bool = False):
                return {"quality_score": "high", "per_item": [1.0]}

        with pytest.raises(BiasScopeError, match="headline score"):
            Textual().run()


class TestWholeNumberItemCounts:
    """`_count_items` looked for `isinstance(value, int)`, but a metric that
    computes its count through numpy or a division reports a whole-number
    float: CrowS-Pairs returns `num_pairs: 2.0`. The count then fell through
    to 0 and the sample-size guard rejected a metric that had in fact scored
    every item. The guard exists to catch "scored nothing", not to police
    numeric type (REVIEW_LATER RL-041).
    """

    def test_a_whole_number_float_count_is_accepted(self):
        class FloatCount(FakeMetric):
            def evaluate(self, return_details: bool = False):
                return {"bias_score": 0.5, "num_pairs": 2.0}

        assert FloatCount().run().n == 2

    def test_a_fractional_count_is_still_rejected(self):
        """2.5 items is not a count; it is a bug in the metric."""

        class Fractional(FakeMetric):
            def evaluate(self, return_details: bool = False):
                return {"bias_score": 0.5, "num_pairs": 2.5}

        with pytest.raises(BiasScopeError, match="n must be positive"):
            Fractional().run()

    def test_a_zero_count_is_still_rejected(self):
        class Empty(FakeMetric):
            def evaluate(self, return_details: bool = False):
                return {"bias_score": 0.5, "num_pairs": 0.0}

        with pytest.raises(BiasScopeError, match="n must be positive"):
            Empty().run()
