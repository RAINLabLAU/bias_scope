"""Differential test: the in-library chi-square against `scipy.stats`.

`chi_square_2xk` and `chi_square_p_value` are written out by hand so the core
install stays free of scipy (PLAN.md Section 1). Two metrics depend on them —
`DisCo`, whose significance test they *are*, and `TrustLLMDisparagement`, whose
reference calls `scipy.stats.chi2_contingency` directly. Both cases mean the
hand-written version has to be shown equal to scipy rather than assumed equal.

scipy is a dev-only oracle. The test skips when it is absent.
"""

import numpy as np
import pytest

from bias_scope.probability_based.disco import chi_square_2xk, chi_square_p_value

stats = pytest.importorskip(
    "scipy.stats", reason="scipy is a dev-only oracle; pip install scipy"
)

TOLERANCE = 1e-9
N_RANDOM_INPUTS = 500


def _reference(table):
    """scipy's uncorrected Pearson statistic, df and p for a contingency table."""
    statistic, p, df, _ = stats.chi2_contingency(np.array(table), correction=False)
    return float(statistic), int(df), float(p)


class TestAgainstScipy:
    def test_random_tables_agree_on_statistic_df_and_p(self):
        """500 seeded tables, 2-5 rows by 2-6 columns, counts 0-40."""
        rng = np.random.default_rng(20260823)
        worst = {"statistic": 0.0, "p": 0.0}
        compared = 0

        for _ in range(N_RANDOM_INPUTS):
            rows = int(rng.integers(2, 6))
            cols = int(rng.integers(2, 7))
            table = rng.integers(0, 41, (rows, cols))
            # scipy rejects a table with an all-zero row or column; so would
            # any sensible reading of it, and `chi_square_2xk` returns 0 there.
            if (table.sum(axis=0) == 0).any() or (table.sum(axis=1) == 0).any():
                continue

            theirs_statistic, theirs_df, theirs_p = _reference(table)
            ours_statistic, ours_df = chi_square_2xk(table.tolist())
            ours_p = chi_square_p_value(ours_statistic, ours_df)

            assert ours_df == theirs_df
            worst["statistic"] = max(
                worst["statistic"], abs(ours_statistic - theirs_statistic))
            worst["p"] = max(worst["p"], abs(ours_p - theirs_p))
            compared += 1

        assert compared > N_RANDOM_INPUTS // 2, f"only {compared} usable tables"
        assert worst["statistic"] < TOLERANCE, f"statistic differs by {worst['statistic']:.3e}"
        assert worst["p"] < TOLERANCE, f"p-value differs by {worst['p']:.3e}"

    def test_the_2x4_shape_trustllm_disparagement_produces(self):
        """`p_value` crosstabs sex or race against four salary brackets, so the
        table is never 2x2 and scipy applies no Yates correction. Pinned,
        because a corrected statistic would silently disagree."""
        table = [[12, 30, 8, 4], [25, 11, 9, 6]]
        theirs_statistic, theirs_df, theirs_p = _reference(table)
        ours_statistic, ours_df = chi_square_2xk(table)
        assert ours_df == theirs_df == 3
        assert ours_statistic == pytest.approx(theirs_statistic, abs=TOLERANCE)
        assert chi_square_p_value(ours_statistic, ours_df) == pytest.approx(
            theirs_p, abs=TOLERANCE)
        # And the default-corrected call agrees too, since correction only
        # applies to 2x2 — so DisCo and TrustLLM are unaffected by the choice.
        assert stats.chi2_contingency(np.array(table))[0] == pytest.approx(
            ours_statistic, abs=TOLERANCE)

    def test_a_table_with_no_association_gives_p_one(self):
        table = [[10, 10], [20, 20]]
        statistic, df = chi_square_2xk(table)
        assert statistic == pytest.approx(0.0, abs=TOLERANCE)
        assert chi_square_p_value(statistic, df) == pytest.approx(1.0)
