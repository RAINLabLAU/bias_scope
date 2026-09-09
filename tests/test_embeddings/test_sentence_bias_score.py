"""Tests for SentenceBiasScore metric."""

import numpy as np
import pytest
import torch

from bias_scope.embeddings_based import SentenceBiasScore


def _no_gender_terms(n_tokens: int) -> np.ndarray:
    return np.zeros(n_tokens, dtype=bool)


class TestSentenceBiasScore:
    """Test Sentence Bias Score metric."""

    def test_basic_functionality(self):
        """Test basic bias score computation."""
        word_embeddings = np.array(
            [
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )
        gender_direction = np.array([1.0, 0.0, 0.0])
        importance = np.array([0.3, 0.3, 0.4])

        # Test OO API
        sbs = SentenceBiasScore()
        female_bias, male_bias = sbs.evaluate(
            word_embeddings, gender_direction, importance, _no_gender_terms(3)
        )

        assert isinstance(female_bias, float)
        assert isinstance(male_bias, float)
        assert female_bias > 0
        assert male_bias < 0

    def test_gender_word_exclusion(self):
        """Test that masked words are excluded."""
        sbs = SentenceBiasScore()

        word_embeddings = np.array(
            [
                [1.0, 0.0],
                [0.8, 0.2],
            ]
        )
        gender_direction = np.array([1.0, 0.0])
        importance = np.array([0.5, 0.5])
        mask = np.array([True, False])

        bias_with_mask, _ = sbs.evaluate(
            word_embeddings, gender_direction, importance, mask
        )
        bias_without_mask, _ = sbs.evaluate(
            word_embeddings, gender_direction, importance, _no_gender_terms(2)
        )

        assert bias_with_mask < bias_without_mask

    def test_importance_weighting(self):
        """Test that importance weights are applied correctly."""
        sbs = SentenceBiasScore()

        word_embeddings = np.array(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ]
        )
        gender_direction = np.array([1.0, 0.0, 0.0])

        importance_low = np.array([0.1, 0.1])
        importance_high = np.array([0.9, 0.9])

        mask = _no_gender_terms(2)
        bias_low, _ = sbs.evaluate(
            word_embeddings, gender_direction, importance_low, mask
        )
        bias_high, _ = sbs.evaluate(
            word_embeddings, gender_direction, importance_high, mask
        )

        assert bias_high > bias_low

    def test_with_torch_tensors(self):
        """Test handles PyTorch tensors."""
        sbs = SentenceBiasScore()

        word_embeddings = torch.randn(5, 100)
        gender_direction = torch.randn(100)
        importance = torch.rand(5)

        female_bias, male_bias = sbs.evaluate(
            word_embeddings, gender_direction, importance, _no_gender_terms(5)
        )

        assert isinstance(female_bias, float)
        assert isinstance(male_bias, float)

    def test_zero_importance(self):
        """Test that zero importance means no contribution."""
        sbs = SentenceBiasScore()

        word_embeddings = np.array([[1.0, 0.0], [1.0, 0.0]])
        gender_direction = np.array([1.0, 0.0])
        importance = np.array([1.0, 0.0])

        female_bias, _ = sbs.evaluate(
            word_embeddings, gender_direction, importance, _no_gender_terms(2)
        )

        assert abs(female_bias - 1.0) < 0.01

    def test_validates_dimension_mismatch(self):
        """Test raises error on dimension mismatch."""
        sbs = SentenceBiasScore()

        word_embeddings = np.random.randn(5, 100)
        gender_direction = np.random.randn(50)  # Wrong dimension
        importance = np.array([0.2] * 5)

        with pytest.raises(ValueError, match="does not match embedding dimension"):
            sbs.evaluate(
                word_embeddings, gender_direction, importance, _no_gender_terms(5)
            )

    def test_validates_importance_length(self):
        """Test raises error on wrong importance length."""
        sbs = SentenceBiasScore()

        word_embeddings = np.random.randn(5, 100)
        gender_direction = np.random.randn(100)
        importance = np.array([0.2] * 3)  # Wrong length

        with pytest.raises(ValueError, match="does not match number of words"):
            sbs.evaluate(
                word_embeddings, gender_direction, importance, _no_gender_terms(5)
            )

    def test_validates_mask_length(self):
        """Test raises error on wrong mask length."""
        sbs = SentenceBiasScore()

        word_embeddings = np.random.randn(5, 100)
        gender_direction = np.random.randn(100)
        importance = np.array([0.2] * 5)
        mask = np.array([True, False, True])  # Wrong length

        with pytest.raises(ValueError, match="does not match number of words"):
            sbs.evaluate(word_embeddings, gender_direction, importance, mask)

    def test_sentence_bias_all_words_masked(self):
        """Test behavior when all words are masked."""
        sbs = SentenceBiasScore()

        num_words = 5
        word_embeddings = np.random.randn(num_words, 100)
        gender_direction = np.random.randn(100)
        importance = np.array([0.2] * num_words)
        mask = np.ones(num_words, dtype=bool)  # All True

        female_bias, male_bias = sbs.evaluate(
            word_embeddings, gender_direction, importance, mask
        )
        assert female_bias == 0.0
        assert male_bias == 0.0

    def test_sentence_bias_missing_mask_rejected(self):
        """Test gender-word exclusion mask is required."""
        sbs = SentenceBiasScore()

        word_embeddings = np.random.randn(5, 100)
        gender_direction = np.random.randn(100)
        importance = np.array([0.2] * 5)

        with pytest.raises(TypeError):
            sbs.evaluate(word_embeddings, gender_direction, importance)

    def test_sentence_bias_no_words_masked(self):
        """Test behavior when mask is all False."""
        sbs = SentenceBiasScore()

        num_words = 5
        word_embeddings = np.random.randn(num_words, 100)
        gender_direction = np.random.randn(100)
        importance = np.array([0.2] * num_words)
        mask = np.zeros(num_words, dtype=bool)  # All False

        fb, mb = sbs.evaluate(word_embeddings, gender_direction, importance, mask)

        assert isinstance(fb, float)
        assert isinstance(mb, float)

    def test_sentence_bias_single_word(self):
        """Test with single word sentence."""
        sbs = SentenceBiasScore()

        word_embeddings = np.random.randn(1, 100)
        gender_direction = np.random.randn(100)
        importance = np.array([1.0])

        fb, mb = sbs.evaluate(
            word_embeddings, gender_direction, importance, _no_gender_terms(1)
        )
        assert isinstance(fb, float)
        assert isinstance(mb, float)

    def test_sentence_bias_very_long_sentence(self):
        """Test with 50+ word sentence."""
        sbs = SentenceBiasScore()

        num_words = 60
        word_embeddings = np.random.randn(num_words, 100)
        gender_direction = np.random.randn(100)
        importance = np.random.rand(num_words)
        importance /= importance.sum()  # Normalize

        fb, mb = sbs.evaluate(
            word_embeddings, gender_direction, importance, _no_gender_terms(num_words)
        )
        assert isinstance(fb, float)

    def test_sentence_bias_negative_importance(self):
        """Test raises error on negative importance values."""
        sbs = SentenceBiasScore()

        word_embeddings = np.random.randn(3, 100)
        gender_direction = np.random.randn(100)
        importance = np.array([0.5, 0.6, -0.1])

        with pytest.raises(ValueError, match="must be non-negative"):
            sbs.evaluate(
                word_embeddings, gender_direction, importance, _no_gender_terms(3)
            )

    def test_sentence_bias_nan_in_embeddings(self):
        """Test handles NaN in word embeddings."""
        sbs = SentenceBiasScore()

        word_embeddings = np.random.randn(3, 100)
        word_embeddings[0, 0] = np.nan
        gender_direction = np.random.randn(100)
        importance = np.array([0.3, 0.3, 0.4])

        with pytest.raises(ValueError, match="contains NaN values"):
            sbs.evaluate(
                word_embeddings, gender_direction, importance, _no_gender_terms(3)
            )

    def test_sentence_bias_inf_error(self):
        """Test handles Inf values in inputs."""
        sbs = SentenceBiasScore()

        gender_direction = np.random.randn(100)
        importance = np.array([0.3, 0.3, 0.4])

        # Inf in embeddings
        word_embeddings = np.random.randn(3, 100)
        word_embeddings[0, 0] = np.inf
        with pytest.raises(ValueError, match="contains Inf values"):
            sbs.evaluate(
                word_embeddings, gender_direction, importance, _no_gender_terms(3)
            )

        # Inf in importance
        word_embeddings = np.random.randn(3, 100)
        importance_inf = importance.copy()
        importance_inf[0] = np.inf
        with pytest.raises(ValueError, match="contains Inf values"):
            sbs.evaluate(
                word_embeddings, gender_direction, importance_inf, _no_gender_terms(3)
            )

    def test_sentence_bias_importance_normalization(self):
        """Test works with importance not summing to 1.0."""
        sbs = SentenceBiasScore()

        word_embeddings = np.random.randn(3, 100)
        gender_direction = np.random.randn(100)
        # Importance sums to 10.0, should still work
        importance = np.array([3.0, 3.0, 4.0])

        fb, mb = sbs.evaluate(
            word_embeddings, gender_direction, importance, _no_gender_terms(3)
        )
        assert isinstance(fb, float)

    def test_sentence_bias_zero_gender_direction(self):
        """Test handles zero vector as gender direction."""
        sbs = SentenceBiasScore()

        word_embeddings = np.random.randn(3, 100)
        gender_direction = np.zeros(100)
        importance = np.array([0.3, 0.3, 0.4])

        with pytest.raises(ValueError, match="zero or near-zero magnitude"):
            sbs.evaluate(
                word_embeddings, gender_direction, importance, _no_gender_terms(3)
            )

    def test_sentence_bias_non_boolean_mask(self):
        """Test sentence_bias with non-boolean mask values."""
        sbs = SentenceBiasScore()

        word_embeddings = np.random.randn(3, 100)
        gender_direction = np.random.randn(100)
        importance = np.array([0.3, 0.3, 0.4])
        mask = np.array([1, 0, 1])  # Integers

        with pytest.raises(TypeError, match="must be boolean array"):
            sbs.evaluate(word_embeddings, gender_direction, importance, mask)

    def test_sentence_bias_rejects_column_importance(self):
        """Test (n, 1) importance weights cannot silently broadcast."""
        sbs = SentenceBiasScore()

        word_embeddings = np.array([[1.0, 0.0], [-1.0, 0.0]])
        gender_direction = np.array([1.0, 0.0])
        importance = np.array([[0.5], [0.5]])

        with pytest.raises(ValueError, match="word_importance must be a rank-1"):
            sbs.evaluate(
                word_embeddings, gender_direction, importance, _no_gender_terms(2)
            )

    def test_sentence_bias_rejects_rank_invalid_gender_direction(self):
        """Test gender direction must be one-dimensional."""
        sbs = SentenceBiasScore()

        word_embeddings = np.random.randn(3, 2)
        gender_direction = np.array([[1.0, 0.0]])
        importance = np.array([0.3, 0.3, 0.4])

        with pytest.raises(ValueError, match="gender_direction must be a rank-1"):
            sbs.evaluate(
                word_embeddings, gender_direction, importance, _no_gender_terms(3)
            )

    def test_sentence_bias_rejects_rank_invalid_mask(self):
        """Test gender-word mask must be one-dimensional."""
        sbs = SentenceBiasScore()

        word_embeddings = np.random.randn(3, 2)
        gender_direction = np.array([1.0, 0.0])
        importance = np.array([0.3, 0.3, 0.4])
        mask = np.array([[False], [True], [False]])

        with pytest.raises(ValueError, match="gender_words_mask must be a rank-1"):
            sbs.evaluate(word_embeddings, gender_direction, importance, mask)

    def test_sentence_bias_rejects_raw_string_inputs(self):
        """Test raw strings are rejected from the canonical metric path."""
        sbs = SentenceBiasScore()

        with pytest.raises(TypeError, match="Raw text inputs are noncanonical"):
            sbs.evaluate(
                ["nurse", "engineer"],
                np.array([1.0, 0.0]),
                np.array([0.5, 0.5]),
                _no_gender_terms(2),
            )

    def test_sentence_bias_excluded_terms_contribute_zero(self):
        """Test masked explicit gender terms do not affect the score."""
        sbs = SentenceBiasScore()

        word_embeddings = np.array([[1.0, 0.0], [-1.0, 0.0], [0.6, 0.8]])
        gender_direction = np.array([1.0, 0.0])
        importance = np.array([10.0, 10.0, 0.5])
        mask = np.array([True, True, False])

        result = sbs.evaluate(
            word_embeddings, gender_direction, importance, mask, return_details=True
        )

        assert result["female_bias"] == pytest.approx(0.3)
        assert result["male_bias"] == 0.0
        assert result["absolute_bias"] == pytest.approx(0.3)

    def test_sentence_bias_numeric_oracle_unchanged(self):
        """Test canonical inputs preserve the published scoring equations."""
        sbs = SentenceBiasScore()

        word_embeddings = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]])
        gender_direction = np.array([2.0, 0.0])
        importance = np.array([0.3, 0.2, 0.5])
        mask = np.array([False, False, False])

        result = sbs.evaluate(
            word_embeddings, gender_direction, importance, mask, return_details=True
        )

        assert result["female_bias"] == pytest.approx(0.3)
        assert result["male_bias"] == pytest.approx(-0.2)
        assert result["absolute_bias"] == pytest.approx(0.5)


class TestIntegration:
    """Integration tests for bias metrics."""

    def test_metrics_consistency(self):
        """Test that metrics produce consistent results across runs."""
        sbs = SentenceBiasScore()

        # Fix seed
        np.random.seed(42)

        word_embeddings = np.random.randn(5, 300)
        gender_direction = np.random.randn(300)
        importance = np.array([0.2] * 5)

        mask = _no_gender_terms(5)
        fb1, mb1 = sbs.evaluate(word_embeddings, gender_direction, importance, mask)
        fb2, mb2 = sbs.evaluate(word_embeddings, gender_direction, importance, mask)

        assert fb1 == fb2
        assert mb1 == mb2
