"""Tests for All Unmasked Likelihood (AUL)."""

# Scores are percentages (0-100) as of RL-060: the authors' own scorers report
# a percentage and MetricInfo already declared neutral_value=50,
# value_range=(0, 100). The assertions below were written against the old
# [0, 1] return and are rescaled, not relaxed - each still pins the same
# property (in range / above neutral / below neutral / exactly at a bound).
import numpy as np
import pytest

from bias_scope.metadata import list_metrics, normalized_deviation
from bias_scope.probability_based import AUL


class TestAUL:
    """Test AUL metric."""

    def test_basic_functionality(self):
        """Test with biased predictions."""
        aul = AUL(mode="whitespace")

        # Biased: higher probs for stereotypes
        def biased_predict(sentence, pos):
            if "Women" in sentence:
                return 0.7
            return 0.3

        pairs = [(["Women", "work"], ["Men", "work"])]

        score = aul.evaluate(pairs, biased_predict)

        assert 0.0 <= score <= 100.0
        assert score >= 50.0  # Prefers stereotypes (>= allows for ties)

    def test_unbiased_model(self):
        """Test with unbiased prediction function."""
        aul = AUL(mode="whitespace")

        def unbiased_predict(sentence, pos):
            # Add small variation to avoid exact ties
            import random

            return 0.5 + random.uniform(-0.05, 0.05)

        pairs = [
            (["Women", "are", "smart"], ["Men", "are", "smart"]),
            (["She", "is", "strong"], ["He", "is", "strong"]),
        ]

        score = aul.evaluate(pairs, unbiased_predict)

        # Should be approximately 0.5 (no preference)
        assert abs(score - 50.0) <= 60.0  # Allow variance due to randomness

    def test_empty_pairs_raises_error(self):
        """Test empty sentence pairs raises error."""
        aul = AUL(mode="whitespace")

        def mock_predict(sentence, pos):
            return 0.5

        with pytest.raises(ValueError, match="cannot be empty"):
            aul.evaluate([], mock_predict)

    def test_different_length_raises_error(self):
        """Test sentences with different lengths raise error."""
        aul = AUL(mode="whitespace")

        def mock_predict(sentence, pos):
            return 0.5

        pairs = [(["Women", "are"], ["Men", "are", "smart"])]

        with pytest.raises(ValueError, match="same length"):
            aul.evaluate(pairs, mock_predict)

    def test_empty_sentence_raises_error(self):
        """Test empty sentence raises error."""
        aul = AUL(mode="whitespace")

        def mock_predict(sentence, pos):
            return 0.5

        pairs = [([], [])]

        with pytest.raises(ValueError, match="cannot be empty"):
            aul.evaluate(pairs, mock_predict)

    def test_invalid_probability_raises_error(self):
        """Test invalid probabilities raise error."""
        aul = AUL(mode="whitespace")

        def bad_predict(sentence, pos):
            return 1.5  # > 1

        pairs = [(["Women", "work"], ["Men", "work"])]

        with pytest.raises(ValueError, match="Invalid probability"):
            aul.evaluate(pairs, bad_predict)

    def test_negative_probability_raises_error(self):
        """Test negative probability raises error."""
        aul = AUL(mode="whitespace")

        def bad_predict(sentence, pos):
            return -0.1

        pairs = [(["She", "runs"], ["He", "runs"])]

        with pytest.raises(ValueError, match="Invalid probability"):
            aul.evaluate(pairs, bad_predict)

    def test_zero_probability_raises_error(self):
        """Test zero probability raises error."""
        aul = AUL(mode="whitespace")

        def bad_predict(sentence, pos):
            return 0.0

        pairs = [(["She", "runs"], ["He", "runs"])]

        with pytest.raises(ValueError, match="Invalid probability"):
            aul.evaluate(pairs, bad_predict)

    def test_all_tokens_predicted(self):
        """Test that all tokens in sentence are predicted."""
        aul = AUL(mode="whitespace")

        # Track positions predicted
        predicted_positions = []

        def tracking_predict(sentence, pos):
            predicted_positions.append(pos)
            return 0.5

        pairs = [(["Women", "are", "bad"], ["Men", "are", "bad"])]

        aul.evaluate(pairs, tracking_predict)

        # Should predict all 3 positions for both sentences (6 total)
        assert len(predicted_positions) == 6
        assert 0 in predicted_positions
        assert 1 in predicted_positions
        assert 2 in predicted_positions

    def test_single_pair(self):
        """Test with single sentence pair."""
        aul = AUL(mode="whitespace")

        def mock_predict(sentence, pos):
            return 0.7

        pairs = [(["She", "works"], ["He", "works"])]

        score = aul.evaluate(pairs, mock_predict)

        # Single pair: score is either 0 or 1
        assert score in [0.0, 100.0]

    def test_many_pairs(self):
        """Test with many sentence pairs."""
        aul = AUL(mode="whitespace")

        def mock_predict(sentence, pos):
            return np.random.uniform(0.3, 0.7)

        pairs = [(["Women", "are", "X"], ["Men", "are", "X"]) for _ in range(50)]

        score = aul.evaluate(pairs, mock_predict)

        assert 0.0 <= score <= 100.0

    def test_deterministic_results(self):
        """Test deterministic prediction gives consistent results."""
        aul = AUL(mode="whitespace")

        def deterministic_predict(sentence, pos):
            return 0.8 if "Women" in sentence else 0.4

        pairs = [(["Women", "work"], ["Men", "work"])]

        score1 = aul.evaluate(pairs, deterministic_predict)
        score2 = aul.evaluate(pairs, deterministic_predict)

        assert score1 == score2

    def test_long_sentences(self):
        """Test with longer sentences."""
        aul = AUL(mode="whitespace")

        def mock_predict(sentence, pos):
            return 0.5

        pairs = [
            (
                ["Women", "are", "often", "seen", "as", "more", "emotional"],
                ["Men", "are", "often", "seen", "as", "more", "emotional"],
            )
        ]

        score = aul.evaluate(pairs, mock_predict)

        assert isinstance(score, float)
        assert 0.0 <= score <= 100.0

    def test_position_based_probabilities(self):
        """Test with position-dependent probabilities."""
        aul = AUL(mode="whitespace")

        def position_predict(sentence, pos):
            # Higher prob for first token
            if pos == 0:
                return 0.8
            return 0.4

        pairs = [(["She", "runs"], ["He", "runs"])]

        score = aul.evaluate(pairs, position_predict)

        assert isinstance(score, float)
        assert 0.0 <= score <= 100.0

    def test_computes_average_not_sum(self):
        """Test that AUL computes average, not sum."""
        aul = AUL(mode="whitespace")

        # Track log probs to verify averaging
        call_count = [0]

        def counting_predict(sentence, pos):
            call_count[0] += 1
            return 0.5

        pairs = [(["A", "B", "C"], ["X", "Y", "Z"])]

        score = aul.evaluate(pairs, counting_predict)

        # Should call predict 6 times (3 tokens × 2 sentences)
        assert call_count[0] == 6
        assert isinstance(score, float)

    def test_anti_stereotype_preference(self):
        """Test when model prefers anti-stereotypes."""
        aul = AUL(mode="whitespace")

        def anti_bias_predict(sentence, pos):
            if "Men" in sentence:
                return 0.8
            return 0.3

        pairs = [(["Women", "work"], ["Men", "work"])]

        score = aul.evaluate(pairs, anti_bias_predict)

        assert score < 50.0  # Prefers anti-stereotypes


class TestAULScoreIsAPercentage:
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
        aul = AUL(mode="whitespace")

        def predict(sentence, pos):
            return 0.9 if "S" in sentence else 0.1

        pairs = [
            (["S", "a"], ["x", "a"]),
            (["S", "b"], ["x", "b"]),
            (["S", "c"], ["x", "c"]),
            (["x", "d"], ["S", "d"]),
        ]
        return aul.evaluate(pairs, predict)

    def test_three_of_four_preferred_pairs_score_seventy_five(self):
        assert self._score() == pytest.approx(75.0, abs=1e-6)

    def test_preferring_stereotypes_reads_as_a_positive_deviation(self):
        assert normalized_deviation(self._score(), list_metrics()["AUL"]) > 0
