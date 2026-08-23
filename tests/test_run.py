"""Tests for BiasMetric.run() and its runtime guards (PLAN.md 5.3, Section 1).

The guards are the point of this file. Section 1 says a guard failure "must
never be silenced", so each one gets a test that proves it fires.
"""

import math
from dataclasses import replace

import pytest
from tests.fixtures.tiny_inputs import (
    KNOWN_DEFECTS,
    NEEDS_RESOURCES,
    TINY_INPUTS,
    construct,
)

import bias_scope
from bias_scope._metric_info import METRIC_INFO
from bias_scope.base import BiasScopeError, GeneratedTextMetric
from bias_scope.metadata import FIDELITIES, MetricInfo
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


class TestEveryMetricIsReachableThroughRun:
    """`run()` must be able to find every metric's headline score.

    `run()` looks for one of `bias_score`, `score`, `value` or `effect_size` in
    what `evaluate(return_details=True)` returns. A metric whose details dict
    names its headline anything else — `crows_pairs_score`, `aul_score`,
    `honest_score` — raises `BiasScopeError` and is skipped by `BiasSuite`,
    silently, for every caller.

    Twelve metrics were in that state until 0.2.0, including CrowS-Pairs, AUL
    and AULA. Nothing caught it: the tests in this file exercise `run()`
    against a fixture, never against a real metric, so the whole probability
    family was unreachable through the library's own entry point. It surfaced
    only when `BiasSuite` was run on a real model.
    """

    ACCEPTED = ("bias_score", "score", "value", "effect_size")

    def _metric_modules(self):
        import re
        from pathlib import Path

        root = Path("src/bias_scope")
        return [p for p in sorted(root.rglob("*.py"))
                if re.search(r'"[a-z_]+_score":', p.read_text(encoding="utf-8"))]

    def test_the_modules_are_actually_being_scanned(self):
        """Guard against the check passing because it found nothing."""
        assert len(self._metric_modules()) >= 20

    def test_every_details_dict_exposes_a_key_run_can_find(self):
        offenders = []
        for path in self._metric_modules():
            text = path.read_text(encoding="utf-8")
            if not any(f'"{key}":' in text for key in self.ACCEPTED):
                offenders.append(str(path))
        assert not offenders, (
            "these metrics name their headline score something `run()` does not "
            f"look for, so BiasSuite skips them: {offenders}. Add a "
            '`"bias_score": <headline>` entry to the details dict.'
        )


class TestItemCountsAcceptFloats:
    """A count written as `12.0` must be honoured, not silently dropped.

    Several metrics emit `float(len(sentence_pairs))`. `_count_items` required
    `isinstance(value, int)`, so those counts were ignored, `n` came back 0, and
    the `n > 0` guard skipped the metric. CrowS-Pairs, AUL and AULA were
    unreachable through `BiasSuite` for that reason alone.
    """

    def test_an_integral_float_count_is_accepted(self):
        assert GeneratedTextMetric._count_items({"num_pairs": 12.0}, None) == 12

    def test_an_int_count_still_works(self):
        assert GeneratedTextMetric._count_items({"num_pairs": 12}, None) == 12

    def test_a_non_integral_float_is_not_a_count(self):
        """3.5 items is not a count; falling through is right."""
        assert GeneratedTextMetric._count_items({"num_pairs": 3.5}, None) == 0

    def test_a_bool_is_not_a_count(self):
        """`True` is an int in Python; it is not one item scored."""
        assert GeneratedTextMetric._count_items({"num_pairs": True}, None) == 0

    def test_per_item_still_wins(self):
        assert GeneratedTextMetric._count_items(
            {"num_pairs": 99.0}, [0.1, 0.2, 0.3]) == 3


class TestRunOnEveryMetric:
    """PLAN.md 5.3: call `run()` on every metric with tiny inputs.

    This is the test whose absence let three defects reach a tagged commit —
    unreachable headline scores, item counts rejected for being `float`, and a
    0-1 score against a declared 0-100 scale. Each was found by hand, by
    running `BiasSuite` on a real model, because nothing here exercised a real
    metric through `run()`.

    Inputs live in `tests/fixtures/tiny_inputs.py`. Metrics that load a dataset
    or call a service from inside `evaluate()` are listed in `NEEDS_RESOURCES`
    with the reason; every metric must be in exactly one of the two.
    """

    def test_every_metric_is_covered(self):
        """A new metric cannot escape this check by being forgotten."""
        covered = set(TINY_INPUTS) | set(NEEDS_RESOURCES)
        missing = sorted(set(METRIC_INFO) - covered)
        assert not missing, (
            f"{len(missing)} metrics have neither tiny inputs nor a recorded "
            f"reason they cannot have them: {missing}. Add them to "
            "tests/fixtures/tiny_inputs.py."
        )

    def test_no_metric_claims_both(self):
        overlap = sorted(set(TINY_INPUTS) & set(NEEDS_RESOURCES))
        assert not overlap, f"listed as both runnable and blocked: {overlap}"

    @pytest.mark.parametrize("name", sorted(TINY_INPUTS))
    def test_run_returns_a_valid_biasresult(self, name, request):
        if name in KNOWN_DEFECTS:
            request.node.add_marker(
                pytest.mark.xfail(strict=True, reason=KNOWN_DEFECTS[name])
            )
        result = construct(getattr(bias_scope, name)).run(
            seed=42, **TINY_INPUTS[name])

        assert result.metric == name
        assert isinstance(result.score, float)
        assert math.isfinite(result.score), f"{name} scored {result.score}"

        low, high = result.info.value_range
        assert low <= result.score <= high, (
            f"{name} scored {result.score}, outside its declared "
            f"value_range {result.info.value_range}"
        )
        assert result.n > 0, f"{name} reported n={result.n}"

        if result.ci is not None:
            assert result.ci[0] <= result.ci[1]
            assert result.ci_method != "none"

        assert result.protocol["metric"] == name
        assert result.protocol["seed"] == 42
        assert result.protocol["hash"]
        assert result.info.fidelity in FIDELITIES

    @pytest.mark.parametrize("name", sorted(TINY_INPUTS))
    def test_the_score_is_on_the_declared_scale(self, name):
        """A score must sit on the scale its own MetricInfo describes.

        CrowS-Pairs returned a 0-1 fraction while declaring
        `neutral_value=50.0, value_range=(0.0, 100.0)`. The range check above
        passed — 0.667 is inside (0, 100) — so a stereotyped model plotted on
        the anti-stereotypical side of every figure. A percentage-scaled metric
        whose score never leaves [0, 1] is the signature of that bug.
        """
        info = METRIC_INFO[name]
        if info.value_range != (0.0, 100.0):
            pytest.skip("not a percentage-scaled metric")
        if name in KNOWN_DEFECTS:
            pytest.skip(f"blocked by a known defect: {KNOWN_DEFECTS[name]}")

        result = construct(getattr(bias_scope, name)).run(
            seed=42, **TINY_INPUTS[name])
        assert result.score == pytest.approx(0.0) or result.score > 1.0, (
            f"{name} declares a 0-100 scale but scored {result.score}, which "
            "looks like a 0-1 fraction"
        )

    @pytest.mark.parametrize("name", sorted(NEEDS_RESOURCES))
    def test_blocked_metrics_state_a_reason(self, name):
        assert NEEDS_RESOURCES[name].strip(), f"{name} has an empty reason"

    def test_known_defects_are_all_real_metrics(self):
        """A stale entry here would hide a metric that now works."""
        unknown = sorted(set(KNOWN_DEFECTS) - set(METRIC_INFO))
        assert not unknown, f"KNOWN_DEFECTS names metrics that do not exist: {unknown}"

    def test_known_defects_each_state_a_class_and_a_reason(self):
        for name, reason in KNOWN_DEFECTS.items():
            assert reason.startswith(("headline:", "count:")), (
                f"{name}: reason must name the defect class, got {reason!r}"
            )
