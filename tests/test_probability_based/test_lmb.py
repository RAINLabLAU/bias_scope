"""Tests for Language Model Bias (LMB)."""

import json
import math

import numpy as np
import pytest

from bias_scope.probability_based import LMB


class TestLMB:
    """Test LMB metric."""

    def test_basic_functionality(self):
        """Test with biased predictions."""
        lmb = LMB()

        # Biased: stereotypes get higher prob (lower perplexity)
        def biased_predict(sentence, pos):
            if "Women" in sentence:
                return 0.8  # High prob -> low perplexity
            return 0.4  # Lower prob -> higher perplexity

        pairs = [
            (["Women", "work"], ["Men", "work"]),
            (["Women", "cook"], ["Men", "cook"]),
        ]

        result = lmb.evaluate(pairs, biased_predict)

        # Check all required keys
        assert "t_stat" in result
        assert "p_value" in result
        assert "mean_pp_s1" in result
        assert "mean_pp_s2" in result
        assert "mean_diff" in result
        assert "effect_size" in result
        assert "n" in result
        assert "outliers_removed" in result
        assert "alpha" in result

        # Stereotypes should have lower perplexity
        assert result["mean_pp_s1"] < result["mean_pp_s2"]
        assert result["mean_diff"] < 0

    def test_perplexity_computed_correctly(self):
        """Test perplexity calculation for simple case."""
        lmb = LMB()

        # All tokens have prob 0.5
        def constant_predict(sentence, pos):
            return 0.5

        sentence = ["A", "B"]

        pp = lmb._compute_perplexity(sentence, constant_predict)

        # PP = exp(- (1/2) * (log(0.5) + log(0.5)))
        #    = exp(- (1/2) * 2*log(0.5))
        #    = exp(-log(0.5))
        #    = exp(log(2))
        #    = 2.0
        expected = 2.0

        assert pytest.approx(pp, abs=1e-5) == expected

    def test_paired_evaluation_means(self):
        """Test that mean perplexities are computed correctly."""
        lmb = LMB()

        # S1 always gets prob 0.8, S2 always gets 0.5
        def deterministic_predict(sentence, pos):
            if "S1" in sentence:
                return 0.8
            return 0.5

        pairs = [
            (["S1", "token"], ["S2", "token"]),
            (["S1", "word"], ["S2", "word"]),
        ]

        result = lmb.evaluate(pairs, deterministic_predict, outlier_strategy="none")

        # Compute expected PPs
        # S1: PP = exp(-log(0.8)) = 1/0.8 = 1.25
        # S2: PP = exp(-log(0.5)) = 2.0

        expected_pp_s1 = 1 / 0.8
        expected_pp_s2 = 2.0

        assert pytest.approx(result["mean_pp_s1"], abs=1e-5) == expected_pp_s1
        assert pytest.approx(result["mean_pp_s2"], abs=1e-5) == expected_pp_s2

    def test_outlier_removal_percentile(self):
        """Test outlier removal with percentile strategy."""
        lmb = LMB()

        # Create data with one extreme outlier
        call_count = [0]

        def outlier_predict(sentence, pos):
            idx = call_count[0] // 2  # Pair index
            call_count[0] += 1

            if "S1" in sentence:
                if idx == 4:  # 5th pair has extreme value
                    return 0.001  # Very low prob -> very high PP
                return 0.5
            else:
                return 0.5

        # 10 pairs: pair 4 (0-indexed) will be outlier
        pairs = [
            (["S1", "a"], ["S2", "a"]),
            (["S1", "b"], ["S2", "b"]),
            (["S1", "c"], ["S2", "c"]),
            (["S1", "d"], ["S2", "d"]),
            (["S1", "e"], ["S2", "e"]),  # Outlier
            (["S1", "f"], ["S2", "f"]),
            (["S1", "g"], ["S2", "g"]),
            (["S1", "h"], ["S2", "h"]),
            (["S1", "i"], ["S2", "i"]),
            (["S1", "j"], ["S2", "j"]),
        ]

        result = lmb.evaluate(
            pairs, outlier_predict, outlier_strategy="percentile", outlier_percentile=5.0
        )

        # Should remove some outliers
        assert result["outliers_removed"] >= 1
        assert result["n"] < 10

    def test_outlier_removal_none(self):
        """Test no outlier removal with 'none' strategy."""
        lmb = LMB()

        def mock_predict(sentence, pos):
            return 0.5

        pairs = [
            (["A", "a"], ["B", "a"]),
            (["A", "b"], ["B", "b"]),
        ]

        result = lmb.evaluate(pairs, mock_predict, outlier_strategy="none")

        # No outliers removed
        assert result["outliers_removed"] == 0
        assert result["n"] == 2

    def test_t_test_identical_arrays(self):
        """Test t-test with identical arrays gives t≈0, p≈1."""
        lmb = LMB()

        # Identical probabilities for both sentences
        def identical_predict(sentence, pos):
            return 0.6

        pairs = [
            (["A", "a"], ["B", "a"]),
            (["A", "b"], ["B", "b"]),
            (["A", "c"], ["B", "c"]),
        ]

        result = lmb.evaluate(pairs, identical_predict, outlier_strategy="none")

        # No difference -> t_stat ≈ 0, p_value ≈ 1
        assert pytest.approx(result["t_stat"], abs=0.01) == 0.0
        assert result["p_value"] > 0.9

    def test_empty_pairs_raises_error(self):
        """Test empty sentence pairs raises error."""
        lmb = LMB()

        def mock_predict(sentence, pos):
            return 0.5

        with pytest.raises(ValueError, match="cannot be empty"):
            lmb.evaluate([], mock_predict)

    def test_different_length_raises_error(self):
        """Test sentences with different lengths raise error."""
        lmb = LMB()

        def mock_predict(sentence, pos):
            return 0.5

        pairs = [(["A", "B"], ["C", "D", "E"])]

        with pytest.raises(ValueError, match="same length"):
            lmb.evaluate(pairs, mock_predict)

    def test_empty_sentence_raises_error(self):
        """Test empty sentence raises error."""
        lmb = LMB()

        def mock_predict(sentence, pos):
            return 0.5

        pairs = [([], [])]

        with pytest.raises(ValueError, match="cannot be empty"):
            lmb.evaluate(pairs, mock_predict)

    def test_invalid_probability_raises_error(self):
        """Test invalid probability raises error."""
        lmb = LMB()

        def bad_predict(sentence, pos):
            return 1.5  # > 1

        pairs = [(["A", "B"], ["C", "D"])]

        with pytest.raises(ValueError, match="Invalid probability"):
            lmb.evaluate(pairs, bad_predict)

    def test_zero_probability_raises_error(self):
        """Test zero probability raises error."""
        lmb = LMB()

        def zero_predict(sentence, pos):
            return 0.0  # log(0) is undefined

        pairs = [(["A", "B"], ["C", "D"])]

        with pytest.raises(ValueError, match="Invalid probability"):
            lmb.evaluate(pairs, zero_predict)

    def test_invalid_outlier_strategy_raises_error(self):
        """Test invalid outlier strategy raises error."""
        lmb = LMB()

        def mock_predict(sentence, pos):
            return 0.5

        pairs = [(["A", "B"], ["C", "D"])]

        with pytest.raises(ValueError, match="outlier_strategy must be"):
            lmb.evaluate(pairs, mock_predict, outlier_strategy="invalid")

    def test_invalid_outlier_percentile_raises_error(self):
        """Test invalid outlier percentile raises error."""
        lmb = LMB()

        def mock_predict(sentence, pos):
            return 0.5

        pairs = [(["A", "B"], ["C", "D"])]

        with pytest.raises(ValueError, match="outlier_percentile must be in"):
            lmb.evaluate(pairs, mock_predict, outlier_percentile=60)

    def test_invalid_alpha_raises_error(self):
        """Test invalid alpha raises error."""
        lmb = LMB()

        def mock_predict(sentence, pos):
            return 0.5

        pairs = [(["A", "B"], ["C", "D"])]

        with pytest.raises(ValueError, match="alpha must be in"):
            lmb.evaluate(pairs, mock_predict, alpha=1.5)

    def test_n_equals_one_raises_error(self):
        """Test that n=1 after filtering raises error."""
        lmb = LMB()

        # Simple case: only provide 1 pair
        def mock_predict(sentence, pos):
            return 0.5

        pairs = [(["A"], ["B"])]

        # With no outlier removal, we still only have 1 pair
        with pytest.raises(ValueError, match="Cannot perform t-test with only 1 pair"):
            lmb.evaluate(pairs, mock_predict, outlier_strategy="none")

    def test_all_outliers_removed_raises_error(self):
        """Test that removing all pairs raises error."""
        lmb = LMB()

        # Extreme outlier removal
        def mock_predict(sentence, pos):
            return 0.5

        pairs = [(["A", "B"], ["C", "D"])]

        # With only 1 pair and aggressive outlier removal
        with pytest.raises(ValueError, match="No pairs remaining|Cannot perform t-test"):
            lmb.evaluate(
                pairs, mock_predict, outlier_strategy="percentile", outlier_percentile=49
            )

    def test_effect_size_computed(self):
        """Test that effect size (Cohen's d) is computed."""
        lmb = LMB()

        # Add variance by varying probability by position
        def mock_predict(sentence, pos):
            base_prob = 0.8 if "S1" in sentence else 0.5
            # Add small position-dependent variation
            return base_prob + pos * 0.01

        pairs = [
            (["S1", "a"], ["S2", "a"]),
            (["S1", "b"], ["S2", "b"]),
            (["S1", "c"], ["S2", "c"]),
        ]

        result = lmb.evaluate(pairs, mock_predict, outlier_strategy="none")

        # Effect size should be non-zero
        assert result["effect_size"] != 0.0
        assert isinstance(result["effect_size"], float)
        # Should be negative (S1 has lower PP)
        assert result["effect_size"] < 0

    def test_result_structure(self):
        """Test result dictionary has correct structure."""
        lmb = LMB()

        def mock_predict(sentence, pos):
            return 0.5

        pairs = [
            (["A", "a"], ["B", "a"]),
            (["A", "b"], ["B", "b"]),
        ]

        result = lmb.evaluate(pairs, mock_predict)

        # Check all required keys present
        required_keys = [
            "t_stat",
            "p_value",
            "mean_pp_s1",
            "mean_pp_s2",
            "mean_diff",
            "effect_size",
            "n",
            "outliers_removed",
            "alpha",
        ]

        for key in required_keys:
            assert key in result

        # Check types
        assert isinstance(result["t_stat"], float)
        assert isinstance(result["p_value"], float)
        assert isinstance(result["mean_pp_s1"], float)
        assert isinstance(result["mean_pp_s2"], float)
        assert isinstance(result["mean_diff"], float)
        assert isinstance(result["effect_size"], float)
        assert isinstance(result["n"], int)
        assert isinstance(result["outliers_removed"], int)
        assert isinstance(result["alpha"], float)

        # Check ranges
        assert 0.0 <= result["p_value"] <= 1.0
        assert result["n"] >= 0
        assert result["outliers_removed"] >= 0
        assert 0.0 < result["alpha"] < 1.0

    def test_statistical_significance_detection(self):
        """Test that large differences are detected as significant."""
        lmb = LMB()

        # Large systematic difference with some variance
        call_count = [0]

        def biased_predict(sentence, pos):
            call_count[0] += 1
            # Add small variance to avoid zero std
            noise = (call_count[0] % 7) * 0.001

            if "S1" in sentence:
                return 0.9 + noise  # Very high prob -> low PP
            return 0.2 + noise  # Low prob -> high PP

        pairs = [
            (["S1", str(i)], ["S2", str(i)])
            for i in range(20)  # Many pairs for power
        ]

        result = lmb.evaluate(pairs, biased_predict, outlier_strategy="none", alpha=0.05)

        # Should detect significant difference
        assert result["p_value"] < 0.05
        # Effect size should be substantial
        assert abs(result["effect_size"]) > 0.5

    def test_t_distribution_approximation(self):
        """Test t-distribution p-value approximation is reasonable."""
        lmb = LMB()

        # Known case: with many samples, moderate difference
        # Add variance to make test realistic
        call_count = [0]

        def slightly_biased(sentence, pos):
            call_count[0] += 1
            # Add position and call-dependent variance
            noise = (call_count[0] % 5) * 0.01 + pos * 0.005

            if "S1" in sentence:
                return min(0.9, 0.7 + noise)
            return min(0.9, 0.6 + noise)

        pairs = [(["S1", str(i)], ["S2", str(i)]) for i in range(30)]

        result = lmb.evaluate(pairs, slightly_biased, outlier_strategy="none")

        # P-value should be in reasonable range (not exactly 0 or 1)
        # With moderate difference and variance, likely significant
        assert 0.0 <= result["p_value"] <= 1.0

    def test_t_distribution_matches_known_reference_value(self):
        """Small-df p-value should stay close to a standard reference value."""
        lmb = LMB()

        # Reference: two-tailed p-value for t=2.0 with df=10 is about 0.07339.
        p_value = lmb._t_distribution_pvalue(2.0, 10)

        assert p_value == pytest.approx(0.07339, abs=1e-3)

    def test_t_distribution_characterization_against_scipy(self):
        """LMB p-values should stay close to scipy.stats.t.sf references."""
        scipy_stats = pytest.importorskip("scipy.stats")
        lmb = LMB()

        dfs = [5, 10, 20, 30]
        t_values = [0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]

        max_abs_error = 0.0
        worst_case = None

        for df in dfs:
            for t_abs in t_values:
                expected = float(2.0 * scipy_stats.t.sf(t_abs, df))
                observed = lmb._t_distribution_pvalue(t_abs, df)
                abs_error = abs(observed - expected)

                if abs_error > max_abs_error:
                    max_abs_error = abs_error
                    worst_case = (df, t_abs, observed, expected)

        assert max_abs_error <= 1e-4, (
            f"Max abs error {max_abs_error} exceeded tolerance at {worst_case}"
        )

    def test_perplexity_increases_with_lower_prob(self):
        """Test that perplexity increases when probabilities decrease."""
        lmb = LMB()

        high_prob_sentence = ["A", "B"]
        low_prob_sentence = ["C", "D"]

        def varied_predict(sentence, pos):
            if sentence[0] in ["A"]:
                return 0.9  # High prob
            return 0.1  # Low prob

        pp_high = lmb._compute_perplexity(high_prob_sentence, varied_predict)
        pp_low = lmb._compute_perplexity(low_prob_sentence, varied_predict)

        # Lower probability -> higher perplexity
        assert pp_low > pp_high

    # === A) Common validation gaps ===

    def test_predict_function_not_callable_raises_error(self):
        """Test that non-callable predict_function raises TypeError."""
        lmb = LMB()

        pairs = [(["A", "B"], ["C", "D"])]

        # Test with None
        with pytest.raises(TypeError, match="callable"):
            lmb.evaluate(pairs, None)

        # Test with non-callable value
        with pytest.raises(TypeError, match="callable"):
            lmb.evaluate(pairs, 123)

    def test_predict_function_returning_nan_raises_error(self):
        """Test that predict_function returning NaN raises error."""
        lmb = LMB()

        def nan_predict(sentence, pos):
            return np.nan

        pairs = [(["A", "B"], ["C", "D"])]

        with pytest.raises(ValueError, match="Invalid probability"):
            lmb.evaluate(pairs, nan_predict)

    def test_predict_function_returning_inf_raises_error(self):
        """Test that predict_function returning Inf raises error."""
        lmb = LMB()

        def inf_predict(sentence, pos):
            return np.inf

        pairs = [(["A", "B"], ["C", "D"])]

        with pytest.raises(ValueError, match="Invalid probability"):
            lmb.evaluate(pairs, inf_predict)

    # === C) LMB outlier trimming robustness ===

    def test_outlier_percentile_zero_behavior(self):
        """Test that outlier_percentile near 0 is rejected or handled clearly."""
        lmb = LMB()

        def mock_predict(sentence, pos):
            return 0.5

        pairs = [
            (["A", "a"], ["B", "a"]),
            (["A", "b"], ["B", "b"]),
        ]

        # outlier_percentile must be in (0, 50)
        with pytest.raises(ValueError, match="outlier_percentile must be in"):
            lmb.evaluate(pairs, mock_predict, outlier_percentile=0)

    def test_outlier_percentile_very_small_deterministic(self):
        """Test that very small outlier_percentile behaves deterministically."""
        lmb = LMB()

        def mock_predict(sentence, pos):
            # Create predictable variance
            if "A" in sentence:
                return 0.7 + pos * 0.01
            return 0.5 + pos * 0.01

        pairs = [
            (["A", "a"], ["B", "a"]),
            (["A", "b"], ["B", "b"]),
            (["A", "c"], ["B", "c"]),
        ]

        # Very small percentile should remove very few/no outliers
        result = lmb.evaluate(
            pairs, mock_predict, outlier_strategy="percentile", outlier_percentile=0.1
        )

        # Should not remove all pairs
        assert result["n"] >= 2
        # Should have meaningful results
        assert isinstance(result["t_stat"], float)
        assert isinstance(result["p_value"], float)

    def test_small_n_with_outlier_trimming_fails_gracefully(self):
        """Test that outlier trimming with small n handles edge cases gracefully."""
        lmb = LMB()

        # Test 1: n=2 with moderate outlier removal should work (not remove all)
        def mock_predict(sentence, pos):
            if "A" in sentence:
                return 0.7 + pos * 0.01
            return 0.5 + pos * 0.01

        pairs = [
            (["A", "a"], ["B", "a"]),
            (["A", "b"], ["B", "b"]),
        ]

        # Should work with small percentile
        result = lmb.evaluate(
            pairs, mock_predict, outlier_strategy="percentile", outlier_percentile=5
        )
        assert result["n"] == 2  # No outliers removed

        # Test 2: Only 1 pair should always fail
        single_pair = [(["A"], ["B"])]
        
        with pytest.raises(ValueError, match="Cannot perform t-test with only 1 pair"):
            lmb.evaluate(single_pair, mock_predict, outlier_strategy="none")

    # === D) Schema/serialization ===

    def test_result_json_serializable(self):
        """Test that result dictionary is JSON serializable."""
        lmb = LMB()

        def mock_predict(sentence, pos):
            if "S1" in sentence:
                return 0.8 + pos * 0.01
            return 0.5 + pos * 0.01

        pairs = [
            (["S1", "a"], ["S2", "a"]),
            (["S1", "b"], ["S2", "b"]),
            (["S1", "c"], ["S2", "c"]),
        ]

        result = lmb.evaluate(pairs, mock_predict, outlier_strategy="none")

        # Should not raise
        json_str = json.dumps(result)
        assert isinstance(json_str, str)

        # Should round-trip
        reloaded = json.loads(json_str)
        assert reloaded["t_stat"] == result["t_stat"]
        assert reloaded["p_value"] == result["p_value"]
        assert reloaded["n"] == result["n"]

    # === E) Outlier removal correctness (sigma strategy must actually run) ===

    def test_sigma_outlier_removal_actually_removes_outliers(self):
        """Before the fix, outlier_strategy='sigma' (the default, documented
        as the paper's rule) computed [mean-3*std, mean+3*std] but never
        applied it -- it was silently identical to 'none'. An extreme
        outlier pair must be dropped."""
        lmb = LMB()

        def predict(sentence, pos):
            if "OUTLIER" in sentence:
                return 0.0001  # astronomically low prob -> huge perplexity
            return 0.5

        pairs = [
            (["Women", "work"], ["Men", "work"]),
            (["Women", "cook"], ["Men", "cook"]),
            (["Women", "lead"], ["Men", "lead"]),
            (["Women", "code"], ["Men", "code"]),
            (["OUTLIER", "sentence"], ["Men", "normal"]),
        ]

        result_sigma = lmb.evaluate(pairs, predict, outlier_strategy="sigma")
        result_none = lmb.evaluate(pairs, predict, outlier_strategy="none")

        assert result_none["outliers_removed"] == 0
        assert result_none["n"] == 5
        # The whole point of the fix: sigma must differ from none here.
        assert result_sigma["outliers_removed"] >= 1
        assert result_sigma["n"] < 5

    def test_sigma_outlier_removal_keeps_uniform_data_unchanged(self):
        """With no real outliers, sigma removal should be a no-op (not an
        off-by-something that drops normal pairs)."""
        lmb = LMB()

        def predict(sentence, pos):
            return 0.6 if "Women" in sentence else 0.5

        pairs = [
            (["Women", "work"], ["Men", "work"]),
            (["Women", "cook"], ["Men", "cook"]),
            (["Women", "lead"], ["Men", "lead"]),
            (["Women", "code"], ["Men", "code"]),
        ]
        result = lmb.evaluate(pairs, predict, outlier_strategy="sigma")
        assert result["outliers_removed"] == 0
        assert result["n"] == 4

    # === F) p-value accuracy for df > 30 (the normal-approximation branch) ===

    def test_normal_approximation_pvalue_matches_scipy_reference(self):
        """Before the fix, _normal_cdf computed 0.5*(1+erf(x)) instead of
        0.5*(1+erf(x/sqrt(2))) -- the wrong function entirely -- so every
        p-value computed via the df>30 branch was wrong by up to an order
        of magnitude. Reference values below are scipy.stats.t.sf(t,df)*2,
        computed independently (not by importing scipy here, since it is
        not a declared runtime or test dependency)."""
        lmb = LMB()

        # (t, df, expected p-value from scipy.stats.t.cdf, tolerance)
        # Tolerances reflect the normal approximation's genuine (small,
        # expected) error vs. the exact t-distribution -- not a bug. Before
        # the fix, the error at these points was ~0.05 (roughly 10x too
        # small), not ~0.01.
        cases = [
            (1.96, 31, 0.05904, 0.01),
            (2.5, 31, 0.01792, 0.01),
            (1.96, 50, 0.05558, 0.01),
            (1.96, 254, 0.05109, 0.005),
        ]
        for t_val, df, expected_p, tol in cases:
            got = lmb._t_distribution_pvalue(t_val, df)
            assert got == pytest.approx(expected_p, abs=tol), (t_val, df)

    def test_normal_approximation_pvalue_continuous_across_df_30_boundary(self):
        """The exact (incomplete-beta) branch (df<=30) and the normal
        approximation (df>30) must agree closely near their boundary --
        the old bug made them disagree by roughly 10x at df=31."""
        lmb = LMB()
        p_at_30 = lmb._t_distribution_pvalue(1.96, 30)
        p_at_31 = lmb._t_distribution_pvalue(1.96, 31)
        assert abs(p_at_30 - p_at_31) < 0.01

    # === G) run()'s headline score must be the paper's t-value ===

    def test_return_details_exposes_bias_score_as_t_stat(self):
        lmb = LMB()

        def predict(sentence, pos):
            return 0.3 if "Women" in sentence else 0.7

        pairs = [
            (["Women", "work"], ["Men", "work"]),
            (["Women", "cook"], ["Men", "cook"]),
            (["Women", "lead"], ["Men", "lead"]),
        ]
        result = lmb.evaluate(pairs, predict, outlier_strategy="none")
        assert result["bias_score"] == result["t_stat"]

    def test_run_uses_t_stat_as_headline_not_effect_size(self):
        """Before the fix, run()'s _split_result matched "effect_size"
        (Cohen's d) before finding any other recognised key, since LMB's
        dict had no 'bias_score'/'score'/'value'. The paper reports the
        t-value; run() must report that, not Cohen's d."""
        lmb = LMB()

        # Per-pair probabilities vary (not just by "Women" vs "Men") so the
        # paired differences have real variance and t_stat is finite --
        # a constant difference across pairs gives std_diff=0 and an
        # infinite t_stat, which is a separate, correct degenerate case but
        # not what this test is checking.
        probs = {
            "Women": 0.3, "work": 0.55, "Men": 0.7, "cook": 0.6,
            "lead": 0.65,
        }

        def predict(sentence, pos):
            return probs[sentence[pos]]

        pairs = [
            (["Women", "work"], ["Men", "work"]),
            (["Women", "cook"], ["Men", "cook"]),
            (["Women", "lead"], ["Men", "lead"]),
        ]
        details = lmb.evaluate(pairs, predict, outlier_strategy="none")
        assert math.isfinite(details["t_stat"])
        result = lmb.run(pairs, predict, outlier_strategy="none", ci="none")
        assert result.score == details["t_stat"]
        assert result.score != details["effect_size"]
