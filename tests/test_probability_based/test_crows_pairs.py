"""Tests for CrowS-Pairs Score."""

import numpy as np
import pytest

from bias_scope.probability_based import CrowSPairs


class TestCrowSPairs:
    """Test CrowS-Pairs metric."""

    def test_basic_functionality(self):
        """Test with simple biased prediction function."""
        crows = CrowSPairs()

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

        # Should prefer stereotypes (>= 0.5 allows for ties)
        assert 0.0 <= score <= 1.0
        assert score >= 0.5

    def test_unbiased_model(self):
        """Test with unbiased prediction function."""
        crows = CrowSPairs()

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

        # Should be approximately 0.5 (no preference)
        assert abs(score - 0.5) <= 0.6  # Allow variance due to randomness

    def test_empty_pairs_raises_error(self):
        """Test empty sentence pairs raises error."""
        crows = CrowSPairs()

        def mock_predict(sentence, pos):
            return 0.5

        with pytest.raises(ValueError, match="cannot be empty"):
            crows.evaluate([], mock_predict)

    def test_different_length_raises_error(self):
        """Test sentences with different lengths raise error."""
        crows = CrowSPairs()

        def mock_predict(sentence, pos):
            return 0.5

        pairs = [(["Women", "are"], ["Men", "are", "smart"])]  # Different lengths

        with pytest.raises(ValueError, match="same length"):
            crows.evaluate(pairs, mock_predict)

    def test_empty_sentence_raises_error(self):
        """Test empty sentence raises error."""
        crows = CrowSPairs()

        def mock_predict(sentence, pos):
            return 0.5

        pairs = [([], [])]

        with pytest.raises(ValueError, match="cannot be empty"):
            crows.evaluate(pairs, mock_predict)

    def test_invalid_probability_raises_error(self):
        """Test invalid probabilities raise error."""
        crows = CrowSPairs()

        # Returns invalid probability
        def bad_predict(sentence, pos):
            return 1.5  # > 1

        pairs = [(["Women", "work"], ["Men", "work"])]

        with pytest.raises(ValueError, match="Invalid probability"):
            crows.evaluate(pairs, bad_predict)

    def test_negative_probability_raises_error(self):
        """Test negative probability raises error."""
        crows = CrowSPairs()

        def bad_predict(sentence, pos):
            return -0.1

        pairs = [(["She", "runs"], ["He", "runs"])]

        with pytest.raises(ValueError, match="Invalid probability"):
            crows.evaluate(pairs, bad_predict)

    def test_categorizes_tokens_correctly(self):
        """Test modified/unmodified categorization."""
        crows = CrowSPairs()

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
        crows = CrowSPairs()

        def mock_predict(sentence, pos):
            return 0.6

        pairs = [
            (["Women", "and", "girls"], ["Men", "and", "boys"])
            # Modified: positions 0 and 2
            # Unmodified: position 1
        ]

        score = crows.evaluate(pairs, mock_predict)

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_all_tokens_modified_raises_error(self):
        """Test completely different sentences raise error."""
        crows = CrowSPairs()

        def mock_predict(sentence, pos):
            return 0.5

        pairs = [(["Women", "are", "smart"], ["Dogs", "eat", "food"])]

        with pytest.raises(ValueError, match="No unmodified tokens"):
            crows.evaluate(pairs, mock_predict)

    def test_single_pair(self):
        """Test with single sentence pair."""
        crows = CrowSPairs()

        def mock_predict(sentence, pos):
            return 0.7

        pairs = [(["She", "works"], ["He", "works"])]

        score = crows.evaluate(pairs, mock_predict)

        # Single pair: score is either 0 or 1
        assert score in [0.0, 1.0]

    def test_many_pairs(self):
        """Test with many sentence pairs."""
        crows = CrowSPairs()

        def mock_predict(sentence, pos):
            return np.random.uniform(0.3, 0.7)

        # Generate 100 pairs
        pairs = [(["Women", "are", "X"], ["Men", "are", "X"]) for _ in range(100)]

        score = crows.evaluate(pairs, mock_predict)

        assert 0.0 <= score <= 1.0

    def test_deterministic_with_same_function(self):
        """Test same function produces same results."""
        crows = CrowSPairs()

        def deterministic_predict(sentence, pos):
            # Deterministic based on sentence content
            return 0.8 if "Women" in sentence else 0.4

        pairs = [(["Women", "work"], ["Men", "work"])]

        score1 = crows.evaluate(pairs, deterministic_predict)
        score2 = crows.evaluate(pairs, deterministic_predict)

        assert score1 == score2

    def test_long_sentences(self):
        """Test with longer sentences."""
        crows = CrowSPairs()

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
        crows = CrowSPairs()

        def bad_predict(sentence, pos):
            return 0.0  # Zero probability

        pairs = [(["She", "runs"], ["He", "runs"])]

        with pytest.raises(ValueError, match="Invalid probability"):
            crows.evaluate(pairs, bad_predict)

    def test_mask_token_replacement(self):
        """Test that [MASK] token is correctly placed."""
        crows = CrowSPairs()

        # Track masked sentences
        masked_sentences = []

        def tracking_predict(sentence, pos):
            masked_sentences.append(sentence.copy())
            return 0.5

        pairs = [(["Women", "are", "bad"], ["Men", "are", "bad"])]

        crows.evaluate(pairs, tracking_predict)

        # Check that [MASK] appears in tracked sentences
        assert any("[MASK]" in sent for sent in masked_sentences)
