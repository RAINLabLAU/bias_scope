"""Differential test: our parity formulas against `fairlearn`, the reference's own.

`DecodingTrust/src/dt/perspectives/fairness/score_calculation_script.py` calls
`fairlearn.metrics.demographic_parity_difference` and
`equalized_odds_difference` directly, so fairlearn *is* the reference
implementation for those two numbers. BiasScope reimplements them to keep
fairlearn out of the runtime dependencies (PLAN.md Section 1, light core), which
means the reimplementation has to be shown equal rather than assumed equal.

fairlearn is a dev-only cross-check. The test skips when it is absent.
"""

import numpy as np
import pytest

from bias_scope.prompts_based.decodingtrust import (
    demographic_parity_difference,
    equalized_odds_difference,
)

fairlearn_metrics = pytest.importorskip(
    "fairlearn.metrics", reason="fairlearn is a dev-only oracle; pip install fairlearn"
)

TOLERANCE = 1e-12
N_RANDOM_INPUTS = 2000


def test_agrees_with_fairlearn_on_random_inputs():
    """Exact agreement across 2000 seeded draws, 2-4 groups, 4-150 records.

    The draws deliberately include degenerate groups — ones with no positive or
    no negative records — because that is where the two could differ: fairlearn
    scores an undefined rate 0.0, and BiasScope matches it so the benchmark's
    published numbers reproduce.
    """
    rng = np.random.default_rng(20260823)
    worst_dpd = worst_eod = 0.0
    compared = 0

    for _ in range(N_RANDOM_INPUTS):
        n = int(rng.integers(4, 151))
        k = int(rng.integers(2, 5))
        labels = rng.integers(0, 2, n)
        predictions = rng.integers(0, 2, n)
        groups = rng.integers(0, k, n)

        theirs_eod = fairlearn_metrics.equalized_odds_difference(
            labels, predictions, sensitive_features=groups)
        if np.isnan(theirs_eod):
            continue
        theirs_dpd = fairlearn_metrics.demographic_parity_difference(
            labels, predictions, sensitive_features=groups)

        worst_dpd = max(worst_dpd, abs(theirs_dpd - demographic_parity_difference(
            list(labels), list(predictions), list(groups))))
        worst_eod = max(worst_eod, abs(theirs_eod - equalized_odds_difference(
            list(labels), list(predictions), list(groups))))
        compared += 1

    assert compared > N_RANDOM_INPUTS // 2, (
        f"only {compared} of {N_RANDOM_INPUTS} draws were comparable"
    )
    assert worst_dpd < TOLERANCE, f"demographic parity differs by {worst_dpd:.3e}"
    assert worst_eod < TOLERANCE, f"equalized odds differs by {worst_eod:.3e}"
