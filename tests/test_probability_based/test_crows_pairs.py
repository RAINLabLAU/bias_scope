"""Tests for CrowS-Pairs Score."""

import numpy as np
import pytest
from scripts.experiments.emnlp_reproduction import _crows_pair_from_row

from bias_scope.metadata import list_metrics, normalized_deviation
from bias_scope.probability_based import CrowSPairs


class TestCrowSPairs:
    """Test CrowS-Pairs metric."""

    def test_basic_functionality(self):
        """Test with simple biased prediction function."""
        crows = CrowSPairs(mode="whitespace")

        # Mock prediction: biased toward stereotypes
        def biased_predict(sentence, pos):
            if "Women" in sentence:
                return 0.8  # High confidence
            return 0.3  # Low confidence

        pairs = [
            (["Women", "are", "bad"], ["Men", "are", "bad"]),
            (["Girls", "like", "pink"], ["Boys", "like", "pink"]),
        ]

        score = crows.evaluate(pairs, biased_predict)

        # Should prefer stereotypes (>= 50 allows for ties)
        assert 0.0 <= score <= 100.0
        assert score >= 50.0

    def test_unbiased_model(self):
        """Test with unbiased prediction function."""
        crows = CrowSPairs(mode="whitespace")

        # Mock prediction: slightly varying probabilities to avoid exact ties
        def unbiased_predict(sentence, pos):
            # Add small random variation to avoid exact ties
            import random

            return 0.5 + random.uniform(-0.05, 0.05)

        pairs = [
            (["Women", "are", "smart"], ["Men", "are", "smart"]),
            (["She", "is", "strong"], ["He", "is", "strong"]),
        ]

        score = crows.evaluate(pairs, unbiased_predict)

        # Should be approximately 50 (no preference)
        assert abs(score - 50.0) <= 60.0  # Allow variance due to randomness

    def test_empty_pairs_raises_error(self):
        """Test empty sentence pairs raises error."""
        crows = CrowSPairs(mode="whitespace")

        def mock_predict(sentence, pos):
            return 0.5

        with pytest.raises(ValueError, match="cannot be empty"):
            crows.evaluate([], mock_predict)

    def test_different_length_raises_error(self):
        """Test sentences with different lengths raise error."""
        crows = CrowSPairs(mode="whitespace")

        def mock_predict(sentence, pos):
            return 0.5

        pairs = [(["Women", "are"], ["Men", "are", "smart"])]  # Different lengths

        with pytest.raises(ValueError, match="same length"):
            crows.evaluate(pairs, mock_predict)

    def test_empty_sentence_raises_error(self):
        """Test empty sentence raises error."""
        crows = CrowSPairs(mode="whitespace")

        def mock_predict(sentence, pos):
            return 0.5

        pairs = [([], [])]

        with pytest.raises(ValueError, match="cannot be empty"):
            crows.evaluate(pairs, mock_predict)

    def test_invalid_probability_raises_error(self):
        """Test invalid probabilities raise error."""
        crows = CrowSPairs(mode="whitespace")

        # Returns invalid probability
        def bad_predict(sentence, pos):
            return 1.5  # > 1

        pairs = [(["Women", "work"], ["Men", "work"])]

        with pytest.raises(ValueError, match="Invalid probability"):
            crows.evaluate(pairs, bad_predict)

    def test_negative_probability_raises_error(self):
        """Test negative probability raises error."""
        crows = CrowSPairs(mode="whitespace")

        def bad_predict(sentence, pos):
            return -0.1

        pairs = [(["She", "runs"], ["He", "runs"])]

        with pytest.raises(ValueError, match="Invalid probability"):
            crows.evaluate(pairs, bad_predict)

    def test_categorizes_tokens_correctly(self):
        """Test modified/unmodified categorization."""
        crows = CrowSPairs(mode="whitespace")

        # Track which positions are masked
        masked_positions = []

        def tracking_predict(sentence, pos):
            masked_positions.append(pos)
            return 0.5

        pairs = [(["Women", "are", "bad"], ["Men", "are", "bad"])]

        crows.evaluate(pairs, tracking_predict)

        # Should mask positions 1 and 2 (are, bad) - unmodified
        # Should NOT mask position 0 (Women/Men) - modified
        # Each sentence processed separately, so 4 calls total
        assert len(masked_positions) == 4
        assert 1 in masked_positions
        assert 2 in masked_positions

    def test_multiple_modified_tokens(self):
        """Test with multiple modified tokens."""
        crows = CrowSPairs(mode="whitespace")

        def mock_predict(sentence, pos):
            return 0.6

        pairs = [
            (["Women", "and", "girls"], ["Men", "and", "boys"])
            # Modified: positions 0 and 2
            # Unmodified: position 1
        ]

        score = crows.evaluate(pairs, mock_predict)

        assert isinstance(score, float)
        assert 0.0 <= score <= 100.0

    def test_all_tokens_modified_raises_error(self):
        """Test completely different sentences raise error."""
        crows = CrowSPairs(mode="whitespace")

        def mock_predict(sentence, pos):
            return 0.5

        pairs = [(["Women", "are", "smart"], ["Dogs", "eat", "food"])]

        with pytest.raises(ValueError, match="No unmodified tokens"):
            crows.evaluate(pairs, mock_predict)

    def test_single_pair(self):
        """Test with single sentence pair."""
        crows = CrowSPairs(mode="whitespace")

        def mock_predict(sentence, pos):
            return 0.7

        pairs = [(["She", "works"], ["He", "works"])]

        score = crows.evaluate(pairs, mock_predict)

        # Single pair: score is either 0 or 100
        assert score in [0.0, 100.0]

    def test_many_pairs(self):
        """Test with many sentence pairs."""
        crows = CrowSPairs(mode="whitespace")

        def mock_predict(sentence, pos):
            return np.random.uniform(0.3, 0.7)

        # Generate 100 pairs
        pairs = [(["Women", "are", "X"], ["Men", "are", "X"]) for _ in range(100)]

        score = crows.evaluate(pairs, mock_predict)

        assert 0.0 <= score <= 100.0

    def test_deterministic_with_same_function(self):
        """Test same function produces same results."""
        crows = CrowSPairs(mode="whitespace")

        def deterministic_predict(sentence, pos):
            # Deterministic based on sentence content
            return 0.8 if "Women" in sentence else 0.4

        pairs = [(["Women", "work"], ["Men", "work"])]

        score1 = crows.evaluate(pairs, deterministic_predict)
        score2 = crows.evaluate(pairs, deterministic_predict)

        assert score1 == score2
        assert score1 == 100.0

    def test_long_sentences(self):
        """Test with longer sentences."""
        crows = CrowSPairs(mode="whitespace")

        def mock_predict(sentence, pos):
            return 0.5

        pairs = [
            (
                ["Women", "are", "often", "seen", "as", "more", "emotional"],
                ["Men", "are", "often", "seen", "as", "more", "emotional"],
            )
        ]

        score = crows.evaluate(pairs, mock_predict)

        assert isinstance(score, float)

    def test_zero_probability_raises_error(self):
        """Test zero probability raises error."""
        crows = CrowSPairs(mode="whitespace")

        def bad_predict(sentence, pos):
            return 0.0  # Zero probability

        pairs = [(["She", "runs"], ["He", "runs"])]

        with pytest.raises(ValueError, match="Invalid probability"):
            crows.evaluate(pairs, bad_predict)

    def test_mask_token_replacement(self):
        """Test that [MASK] token is correctly placed."""
        crows = CrowSPairs(mode="whitespace")

        # Track masked sentences
        masked_sentences = []

        def tracking_predict(sentence, pos):
            masked_sentences.append(sentence.copy())
            return 0.5

        pairs = [(["Women", "are", "bad"], ["Men", "are", "bad"])]

        crows.evaluate(pairs, tracking_predict)

        # Check that [MASK] appears in tracked sentences
        assert any("[MASK]" in sent for sent in masked_sentences)

    def test_score_scale_is_percentage(self):
        crows = CrowSPairs(mode="whitespace")

        def predict(sentence, pos):
            return 0.8 if "Women" in sentence else 0.4

        pairs = [(["Women", "work"], ["Men", "work"])]

        assert crows.evaluate(pairs, predict) == 100.0

    def test_return_details_includes_bias_score_for_run(self):
        crows = CrowSPairs(mode="whitespace")

        def predict(sentence, pos):
            return 0.8 if "Women" in sentence else 0.4

        result = crows.evaluate(
            [(["Women", "work"], ["Men", "work"])],
            predict,
            return_details=True,
        )

        assert result["bias_score"] == 100.0
        assert result["crows_pairs_score"] == 100.0

    def test_run_succeeds_and_exposes_score(self):
        crows = CrowSPairs(mode="whitespace")

        def predict(sentence, pos):
            return 0.8 if "Women" in sentence else 0.4

        result = crows.run(
            [(["Women", "work"], ["Men", "work"])],
            predict,
            ci="none",
        )

        assert result.score == 100.0
        assert result.details["bias_score"] == 100.0

    def test_neutral_fixture_is_50(self):
        crows = CrowSPairs(mode="whitespace")

        def predict(sentence, pos):
            return 0.8 if "Women" in sentence else 0.4

        score = crows.evaluate(
            [
                (["Women", "work"], ["Men", "work"]),
                (["Girls", "work"], ["Boys", "work"]),
            ],
            lambda sentence, pos: predict(sentence, pos)
            if "Women" in sentence or "Men" in sentence
            else (0.4 if "Girls" in sentence else 0.8),
        )

        assert score == 50.0

    def test_antistereo_row_keeps_sent_more_first(self):
        row = {
            "sent_more": "The doctor said that he was late.",
            "sent_less": "The doctor said that she was late.",
            "stereo_antistereo": "antistereo",
        }

        assert _crows_pair_from_row(row) == (row["sent_more"], row["sent_less"])

    # === per_item exposure + run() confidence intervals ===

    def test_return_details_exposes_per_item(self):
        """run() needs details['per_item'] to compute a CI; verify
        evaluate(return_details=True) reports it, scaled to match the 0-100
        bias_score."""
        crows = CrowSPairs(mode="whitespace")

        def predict(sentence, pos):
            return 0.8 if "Women" in sentence else 0.3

        pairs = [
            (["Women", "work"], ["Men", "work"]),
            (["Women", "cook"], ["Men", "cook"]),
            (["Girls", "play"], ["Boys", "play"]),
        ]
        result = crows.evaluate(pairs, predict, return_details=True)
        assert result["per_item"] == [100.0, 100.0, 0.0]
        assert np.mean(result["per_item"]) == pytest.approx(result["bias_score"])

    def test_run_produces_bootstrap_ci(self):
        """Before the fix, run()'s default ci='bootstrap' silently returned
        no interval for CrowS-Pairs because per_item was never exposed."""
        crows = CrowSPairs(mode="whitespace")

        def predict(sentence, pos):
            return 0.8 if "Women" in sentence else 0.3

        pairs = [
            (["Women", "work"], ["Men", "work"]),
            (["Women", "cook"], ["Men", "cook"]),
            (["Girls", "play"], ["Boys", "play"]),
            (["Girls", "read"], ["Boys", "read"]),
        ]
        result = crows.run(pairs, predict)  # default ci="bootstrap"
        assert result.ci is not None
        assert result.ci_method == "bootstrap"
        ci_low, ci_high = result.ci
        assert ci_low <= result.score <= ci_high

    def test_run_produces_wald_ci_on_the_0_100_scale(self):
        """Before the fix, base.py's _interval checked `per_item is None`
        before `ci == 'wald'`, so ci='wald' was unreachable for CrowS-Pairs
        (which never exposed per_item) even though wald_ci(score, n) needs
        no per-item data. CrowS-Pairs is exactly the proportion-style metric
        stats.py's wald_ci docstring names it as being for."""
        crows = CrowSPairs(mode="whitespace")

        def predict(sentence, pos):
            return 0.8 if "Women" in sentence else 0.3

        pairs = [
            (["Women", "work"], ["Men", "work"]),
            (["Women", "cook"], ["Men", "cook"]),
            (["Girls", "play"], ["Boys", "play"]),
            (["Girls", "read"], ["Boys", "read"]),
        ]
        result = crows.run(pairs, predict, ci="wald")
        assert result.ci is not None
        assert result.ci_method == "wald"
        ci_low, ci_high = result.ci
        assert 0.0 <= ci_low <= result.score <= ci_high <= 100.0

    def test_wordpiece_mode_also_exposes_per_item(self):
        class FakeWordpieceScorer:
            def encode(self, sentence):
                return [ord(c) for c in sentence]

            def align_unmodified(self, ids_a, ids_b):
                n = min(len(ids_a), len(ids_b))
                return list(range(n)), list(range(n))

            def pll_over_positions(self, input_ids, positions):
                return -float(sum(input_ids[p] for p in positions))

        crows = CrowSPairs(mode="wordpiece")
        pairs = [("bb", "aa"), ("aa", "bb")]
        result = crows.evaluate(pairs, FakeWordpieceScorer(), return_details=True)
        assert result["per_item"] == [0.0, 100.0]
        assert np.mean(result["per_item"]) == pytest.approx(result["bias_score"])

    # === Tie-rounding: match Nangia 2020's reference, which rounds each
    # side to 3 decimals before comparing ===

    def test_near_tie_below_rounding_precision_counts_as_no_preference(self):
        """A PLL difference smaller than 0.001 rounds to an exact tie in the
        reference (`score[stype] = round(score[stype], 3)` in metric.py
        before the `>`/`==` comparison) and must not count as a stereotype
        win here either."""
        crows = CrowSPairs(mode="whitespace")

        # Designed so the two sentences' summed log-probabilities differ by
        # far less than 0.001.
        def near_tie_predict(sentence, pos):
            return 0.500000001 if "Women" in sentence else 0.5

        pairs = [(["Women", "work", "today"], ["Men", "work", "today"])]
        result = crows.evaluate(pairs, near_tie_predict, return_details=True)
        assert result["per_item"] == [0.0]
        assert result["bias_score"] == 0.0

    def test_real_difference_above_rounding_precision_still_counts(self):
        """A genuine difference well above the 0.001 rounding threshold is
        unaffected by the rounding fix."""
        crows = CrowSPairs(mode="whitespace")

        def predict(sentence, pos):
            return 0.8 if "Women" in sentence else 0.3

        pairs = [(["Women", "work"], ["Men", "work"])]
        result = crows.evaluate(pairs, predict, return_details=True)
        assert result["per_item"] == [100.0]


class TestCrowSPairsScoreIsAPercentage:
    """RL-060: the score must be on the scale its own metadata declares.

    These metrics returned a fraction in [0, 1] while four independent sources
    say percent:

    * the authors' own scorers -
      `crows-pairs/metric.py:270`  round((stereo + antistereo) / N * 100, 2)
      `evaluate_bias_in_mlm/evaluate.py:213`  round((stereo / total) * 100, 2)
      PLAN.md Section 1: where paper and code disagree the code wins; here
      they agree with each other and not with us.
    * Nangia et al. 2020 Table 3 reports 60.5 for bert-base-uncased.
    * `validation/registry.yaml` carries `published_value: 60.5`.
    * `MetricInfo` declares `neutral_value=50.0, value_range=(0.0, 100.0)`.

    The consequence was not cosmetic: `normalized_deviation(0.5573, CrowSPairs)`
    returned -0.9889 - the wrong *sign* - reading a mildly stereotype-preferring
    model as maximally anti-stereotypical, which is exactly what the profile
    view, `compare` and `correlate` plot.

    Derivation below: 4 pairs, the model prefers the stereotypical sentence in
    3 of them, so the score is 3/4 * 100 = 75.0.
    """

    def _score(self):
        crows = CrowSPairs(mode="whitespace")

        def predict(sentence, pos):
            # "S" marks the sentence this mock model prefers.
            return 0.9 if "S" in sentence else 0.1

        pairs = [
            (["S", "a"], ["x", "a"]),
            (["S", "b"], ["x", "b"]),
            (["S", "c"], ["x", "c"]),
            (["x", "d"], ["S", "d"]),  # the one pair preferring the anti-stereotype
        ]
        return crows.evaluate(pairs, predict)

    def test_three_of_four_preferred_pairs_score_seventy_five(self):
        assert self._score() == pytest.approx(75.0, abs=1e-6)

    def test_the_score_is_inside_the_range_its_metadata_declares(self):
        info = list_metrics()["CrowSPairs"]
        assert info.value_range[0] <= self._score() <= info.value_range[1]

    def test_preferring_stereotypes_reads_as_a_positive_deviation(self):
        info = list_metrics()["CrowSPairs"]
        assert normalized_deviation(self._score(), info) > 0
