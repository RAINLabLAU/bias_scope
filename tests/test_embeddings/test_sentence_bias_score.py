"""Tests for SentenceBiasScore metric."""

import numpy as np
import pytest
import torch

from bias_scope.embeddings_based import SentenceBiasScore
from bias_scope.embeddings_based.sentence_bias_score import (
    build_gender_words_mask,
    derive_gender_direction,
    derive_word_importance,
)


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


class TestSentenceBiasScoreRun:
    """PLAN.md 5.3: run() must work. Dolci et al. 2023 Eq. 3 (Abs-BiasScore) is
    the paper's own single-value summary ("useful ... when sorting multiple
    sentences"), so it is the natural BiasResult.score."""

    def test_run_reports_abs_bias_score_as_the_headline_score(self):
        we = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]])
        gd = np.array([2.0, 0.0])
        imp = np.array([0.3, 0.2, 0.5])
        mask = np.array([False, False, False])

        result = SentenceBiasScore().run(we, gd, imp, mask)

        assert result.score == pytest.approx(0.5)  # |0.3| + |-0.2|
        assert result.n == 3
        assert result.breakdown == pytest.approx(
            {"female_bias": 0.3, "male_bias": -0.2}
        )

    def test_run_score_matches_evaluate_absolute_bias(self):
        rng = np.random.default_rng(3)
        we = rng.normal(size=(6, 8))
        gd = rng.normal(size=8)
        imp = rng.random(6)
        mask = np.zeros(6, dtype=bool)

        details = SentenceBiasScore().evaluate(
            we, gd, imp, mask, return_details=True
        )
        result = SentenceBiasScore().run(we, gd, imp, mask)

        assert result.score == pytest.approx(details["absolute_bias"])
        assert result.p_value is None
        assert result.ci is None


class TestDeriveGenderDirection:
    """Dolci et al. 2023, Sec. 3.2: gender direction = top PCA component of
    gender word-pair difference vectors, oriented so female words are
    positive."""

    def test_recovers_a_clear_axis(self):
        rng = np.random.default_rng(0)
        n_pairs, dim = 10, 5
        noise = rng.normal(scale=0.01, size=(n_pairs, dim))
        male = rng.normal(size=(n_pairs, dim))
        offset = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        female = male + offset + noise

        direction = derive_gender_direction(female, male)

        assert direction.shape == (dim,)
        assert np.linalg.norm(direction) == pytest.approx(1.0)
        assert abs(direction[0]) > 0.99
        assert direction[0] > 0  # oriented toward the female argument

    def test_swapping_arguments_flips_the_sign(self):
        rng = np.random.default_rng(1)
        male = rng.normal(size=(10, 4))
        female = male + np.array([1.0, 0.0, 0.0, 0.0])

        towards_female = derive_gender_direction(female, male)
        towards_male = derive_gender_direction(male, female)

        assert np.allclose(towards_female, -towards_male, atol=1e-8)

    def test_rejects_mismatched_shapes(self):
        with pytest.raises(ValueError, match="n_pairs, embedding_dim"):
            derive_gender_direction(np.zeros((3, 4)), np.zeros((2, 4)))

    def test_rejects_too_few_pairs(self):
        with pytest.raises(ValueError, match="at least 2"):
            derive_gender_direction(np.zeros((1, 4)), np.zeros((1, 4)))


class TestDeriveWordImportance:
    """Dolci et al. 2023, Sec. 3.4: importance = the fraction of max-pooled
    output dimensions for which each token's hidden state was the maximum."""

    def test_known_answer_three_tokens_four_dims(self):
        hidden_states = np.array(
            [
                [5.0, 5.0, 0.0, 0.0],  # wins dims 0, 1
                [0.0, 0.0, 5.0, 0.0],  # wins dim 2
                [0.0, 0.0, 0.0, 5.0],  # wins dim 3
            ]
        )

        importance = derive_word_importance(hidden_states)

        assert importance == pytest.approx([0.5, 0.25, 0.25])
        assert importance.sum() == pytest.approx(1.0)

    def test_reproduces_the_papers_saxophone_ratio(self):
        """Fig. 3 of Dolci et al.: 'saxophone' wins 1106 of 4096 max-pool
        dimensions (~27%) in 'A man is playing the saxophone'."""
        dim = 4096
        hidden_states = np.zeros((6, dim))
        hidden_states[4, :1106] = 1.0
        hidden_states[0, 1106:] = 1.0

        importance = derive_word_importance(hidden_states)

        assert importance[4] == pytest.approx(1106 / 4096, abs=1e-9)

    def test_rejects_non_2d_input(self):
        with pytest.raises(ValueError, match="rank-2"):
            derive_word_importance(np.zeros(5))

    def test_rejects_nan(self):
        hidden_states = np.zeros((2, 3))
        hidden_states[0, 0] = np.nan
        with pytest.raises(ValueError, match="NaN or Inf"):
            derive_word_importance(hidden_states)


class TestBuildGenderWordsMask:
    """Dolci et al. 2023, Sec. 3.3: L is matched case-insensitively (their own
    list stores both lower-cased and capitalised forms explicitly)."""

    def test_matches_case_insensitively(self):
        tokens = ["She", "likes", "the", "new", "pink", "dress"]
        lexicon = ["she", "he", "mother", "father"]

        mask = build_gender_words_mask(tokens, lexicon)

        assert mask.tolist() == [True, False, False, False, False, False]
        assert mask.dtype == bool
