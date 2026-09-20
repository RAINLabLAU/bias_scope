"""PLAN.md 5.3: call `run()` on every metric with tiny inputs.

Carried over from the August `v0.2-metrics-and-framework` branch when it was
merged (2026-09-20). Inputs live in `tests/fixtures/tiny_inputs.py`; metrics
that load a dataset or call a service from inside `evaluate()` are listed in
`NEEDS_RESOURCES` with the reason, and metrics whose paper defines no scalar
in `NO_SCALAR_BY_DESIGN`. Every metric must be in exactly one of them.
"""

import math

import pytest
from tests.fixtures.tiny_inputs import (
    KNOWN_DEFECTS,
    NEEDS_RESOURCES,
    NO_SCALAR_BY_DESIGN,
    TINY_INPUTS,
    construct,
)

import bias_scope
from bias_scope._metric_info import METRIC_INFO
from bias_scope.base import BiasScopeError
from bias_scope.metadata import FIDELITIES


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
        covered = (set(TINY_INPUTS) | set(NEEDS_RESOURCES)
                   | set(NO_SCALAR_BY_DESIGN))
        missing = sorted(set(METRIC_INFO) - covered)
        assert not missing, (
            f"{len(missing)} metrics have neither tiny inputs nor a recorded "
            f"reason they cannot have them: {missing}. Add them to "
            "tests/fixtures/tiny_inputs.py."
        )

    def test_no_metric_claims_both(self):
        overlap = sorted(set(TINY_INPUTS) & set(NEEDS_RESOURCES))
        assert not overlap, f"listed as both runnable and blocked: {overlap}"

    @pytest.mark.parametrize("name", sorted(set(TINY_INPUTS) - set(NO_SCALAR_BY_DESIGN)))
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

    @pytest.mark.parametrize("name", sorted(NO_SCALAR_BY_DESIGN))
    def test_metrics_without_a_scalar_are_refused_not_faked(self, name):
        """`run()` must refuse, and must not invent a number.

        Cheng et al. define no scalar for MarkedPersons. Giving it one would
        fabricate a metric the paper does not have (PLAN.md Section 1), so the
        correct behaviour is the loud refusal `run()` already gives — and
        `evaluate()` still returns the full ranked word list.
        """
        metric = construct(getattr(bias_scope, name))
        with pytest.raises(BiasScopeError, match="headline score"):
            metric.run(seed=42, **TINY_INPUTS[name])

        details = metric.evaluate(return_details=True, **TINY_INPUTS[name])
        assert details, f"{name} must still be usable through evaluate()"
        assert NO_SCALAR_BY_DESIGN[name].strip()

    def test_no_metric_is_both_defective_and_by_design(self):
        overlap = sorted(set(KNOWN_DEFECTS) & set(NO_SCALAR_BY_DESIGN))
        assert not overlap, (
            f"{overlap} are listed both as defects and as deliberate; a metric "
            "with no scalar by design is not a bug to fix"
        )

    def test_known_defects_are_all_real_metrics(self):
        """A stale entry here would hide a metric that now works."""
        unknown = sorted(set(KNOWN_DEFECTS) - set(METRIC_INFO))
        assert not unknown, f"KNOWN_DEFECTS names metrics that do not exist: {unknown}"

    def test_known_defects_each_state_a_class_and_a_reason(self):
        for name, reason in KNOWN_DEFECTS.items():
            assert reason.startswith(("headline:", "count:")), (
                f"{name}: reason must name the defect class, got {reason!r}"
            )
