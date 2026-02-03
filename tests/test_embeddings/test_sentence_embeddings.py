"""Tests for sentence embedding metrics (SEAT, SentenceBiasScore)."""

import pytest
import numpy as np
import torch
from bias_scope.embeddings import SEAT, SentenceBiasScore
from bias_scope.embeddings.sentence_embeddings import seat, sentence_bias
from bias_scope.embeddings.word_embeddings import weat


class TestSEAT:
    """Test Sentence Encoder Association Test."""
    
    def test_basic_functionality(self):
        """Test SEAT with simple inputs."""
        target1 = np.random.randn(4, 768)
        target2 = np.random.randn(4, 768)
        attr1 = np.random.randn(3, 768)
        attr2 = np.random.randn(3, 768)
        
        # Test OO API
        seat_instance = SEAT()
        score = seat_instance.compute((target1, target2), (attr1, attr2))
        
        assert isinstance(score, float)
        assert not np.isnan(score)
    
    def test_functional_api(self):
        """Test backward-compatible functional API."""
        target1 = np.random.randn(4, 768)
        target2 = np.random.randn(4, 768)
        attr1 = np.random.randn(3, 768)
        attr2 = np.random.randn(3, 768)
        
        score = seat((target1, target2), (attr1, attr2))
        assert isinstance(score, float)
    
    def test_metadata_properties(self):
        """Test metric metadata is accessible."""
        seat_instance = SEAT()
        
        assert seat_instance.name == "SEAT"
        assert seat_instance.category == "embedding"
        assert seat_instance.complexity == "easy"
        assert "May" in seat_instance.reference
    
    def test_calls_weat(self):
        """Test SEAT produces same result as WEAT."""
        target1 = np.random.randn(4, 768)
        target2 = np.random.randn(4, 768)
        attr1 = np.random.randn(3, 768)
        attr2 = np.random.randn(3, 768)
        
        weat_score = weat((target1, target2), (attr1, attr2))
        seat_score = seat((target1, target2), (attr1, attr2))
        
        assert weat_score == seat_score
    
    def test_sentence_embeddings(self):
        """Test SEAT with typical sentence embedding dimensions."""
        target1 = np.random.randn(5, 768)
        target2 = np.random.randn(5, 768)
        attr1 = np.random.randn(3, 768)
        attr2 = np.random.randn(3, 768)
        
        score = seat((target1, target2), (attr1, attr2))
        assert isinstance(score, float)
        assert not np.isnan(score)

    def test_seat_very_long_sentences(self):
        """Test SEAT with long sentence embeddings."""
        target1 = np.random.randn(5, 768)
        target2 = np.random.randn(5, 768)
        attr1 = np.random.randn(3, 768)
        attr2 = np.random.randn(3, 768)
        
        score = seat((target1, target2), (attr1, attr2))
        assert isinstance(score, float)

    def test_seat_different_group_sizes(self):
        """Test SEAT with unbalanced groups."""
        target1 = np.random.randn(3, 768)
        target2 = np.random.randn(10, 768)
        attr1 = np.random.randn(5, 768)
        attr2 = np.random.randn(2, 768)
        
        score = seat((target1, target2), (attr1, attr2))
        assert isinstance(score, float)


class TestSentenceBiasScore:
    """Test Sentence Bias Score metric."""
    
    def test_basic_functionality(self):
        """Test basic bias score computation."""
        word_embeddings = np.array([
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        gender_direction = np.array([1.0, 0.0, 0.0])
        importance = np.array([0.3, 0.3, 0.4])
        
        # Test OO API
        sbs = SentenceBiasScore()
        female_bias, male_bias = sbs.compute(
            word_embeddings, gender_direction, importance
        )
        
        assert isinstance(female_bias, float)
        assert isinstance(male_bias, float)
        assert female_bias > 0
        assert male_bias < 0
    
    def test_functional_api(self):
        """Test backward-compatible functional API."""
        word_embeddings = np.array([
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
        ])
        gender_direction = np.array([1.0, 0.0, 0.0])
        importance = np.array([0.5, 0.5])
        
        female_bias, male_bias = sentence_bias(
            word_embeddings, gender_direction, importance
        )
        
        assert isinstance(female_bias, float)
        assert isinstance(male_bias, float)
    
    def test_metadata_properties(self):
        """Test metric metadata is accessible."""
        sbs = SentenceBiasScore()
        
        assert sbs.name == "Sentence Bias Score"
        assert sbs.category == "embedding"
        assert sbs.complexity == "easy"
        assert "Dolci" in sbs.reference
    
    def test_gender_word_exclusion(self):
        """Test that masked words are excluded."""
        word_embeddings = np.array([
            [1.0, 0.0],
            [0.8, 0.2],
        ])
        gender_direction = np.array([1.0, 0.0])
        importance = np.array([0.5, 0.5])
        mask = np.array([True, False])
        
        bias_with_mask, _ = sentence_bias(
            word_embeddings, gender_direction, importance, mask
        )
        bias_without_mask, _ = sentence_bias(
            word_embeddings, gender_direction, importance
        )
        
        assert bias_with_mask < bias_without_mask
    
    def test_importance_weighting(self):
        """Test that importance weights are applied correctly."""
        word_embeddings = np.array([
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ])
        gender_direction = np.array([1.0, 0.0, 0.0])
        
        importance_low = np.array([0.1, 0.1])
        importance_high = np.array([0.9, 0.9])
        
        bias_low, _ = sentence_bias(word_embeddings, gender_direction, importance_low)
        bias_high, _ = sentence_bias(word_embeddings, gender_direction, importance_high)
        
        assert bias_high > bias_low
    
    def test_with_torch_tensors(self):
        """Test handles PyTorch tensors."""
        word_embeddings = torch.randn(5, 100)
        gender_direction = torch.randn(100)
        importance = torch.rand(5)
        
        female_bias, male_bias = sentence_bias(
            word_embeddings, gender_direction, importance
        )
        
        assert isinstance(female_bias, float)
        assert isinstance(male_bias, float)
    
    def test_zero_importance(self):
        """Test that zero importance means no contribution."""
        word_embeddings = np.array([[1.0, 0.0], [1.0, 0.0]])
        gender_direction = np.array([1.0, 0.0])
        importance = np.array([1.0, 0.0])
        
        female_bias, _ = sentence_bias(
            word_embeddings, gender_direction, importance
        )
        
        assert abs(female_bias - 1.0) < 0.01
    
    def test_validates_dimension_mismatch(self):
        """Test raises error on dimension mismatch."""
        word_embeddings = np.random.randn(5, 100)
        gender_direction = np.random.randn(50)  # Wrong dimension
        importance = np.array([0.2] * 5)
        
        with pytest.raises(ValueError, match="does not match embedding dimension"):
            sentence_bias(word_embeddings, gender_direction, importance)
    
    def test_validates_importance_length(self):
        """Test raises error on wrong importance length."""
        word_embeddings = np.random.randn(5, 100)
        gender_direction = np.random.randn(100)
        importance = np.array([0.2] * 3)  # Wrong length
        
        with pytest.raises(ValueError, match="does not match number of words"):
            sentence_bias(word_embeddings, gender_direction, importance)
    
    def test_validates_mask_length(self):
        """Test raises error on wrong mask length."""
        word_embeddings = np.random.randn(5, 100)
        gender_direction = np.random.randn(100)
        importance = np.array([0.2] * 5)
        mask = np.array([True, False, True])  # Wrong length
        
        with pytest.raises(ValueError, match="does not match number of words"):
            sentence_bias(word_embeddings, gender_direction, importance, mask)

    def test_sentence_bias_all_words_masked(self):
        """Test behavior when all words are masked."""
        num_words = 5
        word_embeddings = np.random.randn(num_words, 100)
        gender_direction = np.random.randn(100)
        importance = np.array([0.2] * num_words)
        mask = np.ones(num_words, dtype=bool)  # All True
        
        female_bias, male_bias = sentence_bias(
            word_embeddings, gender_direction, importance, mask
        )
        assert female_bias == 0.0
        assert male_bias == 0.0

    def test_sentence_bias_no_words_masked(self):
        """Test behavior when mask is all False."""
        num_words = 5
        word_embeddings = np.random.randn(num_words, 100)
        gender_direction = np.random.randn(100)
        importance = np.array([0.2] * num_words)
        mask = np.zeros(num_words, dtype=bool)  # All False
        
        # With mask
        fb1, mb1 = sentence_bias(word_embeddings, gender_direction, importance, mask)
        # Without mask argument
        fb2, mb2 = sentence_bias(word_embeddings, gender_direction, importance)
        
        assert fb1 == fb2
        assert mb1 == mb2

    def test_sentence_bias_single_word(self):
        """Test with single word sentence."""
        word_embeddings = np.random.randn(1, 100)
        gender_direction = np.random.randn(100)
        importance = np.array([1.0])
        
        fb, mb = sentence_bias(word_embeddings, gender_direction, importance)
        assert isinstance(fb, float)
        assert isinstance(mb, float)

    def test_sentence_bias_very_long_sentence(self):
        """Test with 50+ word sentence."""
        num_words = 60
        word_embeddings = np.random.randn(num_words, 100)
        gender_direction = np.random.randn(100)
        importance = np.random.rand(num_words)
        importance /= importance.sum()  # Normalize
        
        fb, mb = sentence_bias(word_embeddings, gender_direction, importance)
        assert isinstance(fb, float)

    def test_sentence_bias_negative_importance(self):
        """Test raises error on negative importance values."""
        word_embeddings = np.random.randn(3, 100)
        gender_direction = np.random.randn(100)
        importance = np.array([0.5, 0.6, -0.1])
        
        with pytest.raises(ValueError, match="must be non-negative"):
            sentence_bias(word_embeddings, gender_direction, importance)

    def test_sentence_bias_nan_in_embeddings(self):
        """Test handles NaN in word embeddings."""
        word_embeddings = np.random.randn(3, 100)
        word_embeddings[0, 0] = np.nan
        gender_direction = np.random.randn(100)
        importance = np.array([0.3, 0.3, 0.4])
        
        with pytest.raises(ValueError, match="contains NaN values"):
            sentence_bias(word_embeddings, gender_direction, importance)

    def test_sentence_bias_inf_error(self):
        """Test handles Inf values in inputs."""
        gender_direction = np.random.randn(100)
        importance = np.array([0.3, 0.3, 0.4])
        
        # Inf in embeddings
        word_embeddings = np.random.randn(3, 100)
        word_embeddings[0, 0] = np.inf
        with pytest.raises(ValueError, match="contains Inf values"):
            sentence_bias(word_embeddings, gender_direction, importance)
            
        # Inf in importance
        word_embeddings = np.random.randn(3, 100)
        importance_inf = importance.copy()
        importance_inf[0] = np.inf
        with pytest.raises(ValueError, match="contains Inf values"):
            sentence_bias(word_embeddings, gender_direction, importance_inf)

    def test_sentence_bias_importance_normalization(self):
        """Test works with importance not summing to 1.0."""
        word_embeddings = np.random.randn(3, 100)
        gender_direction = np.random.randn(100)
        # Importance sums to 10.0, should still work
        importance = np.array([3.0, 3.0, 4.0]) 
        
        fb, mb = sentence_bias(word_embeddings, gender_direction, importance)
        assert isinstance(fb, float)

    def test_sentence_bias_zero_gender_direction(self):
        """Test handles zero vector as gender direction."""
        word_embeddings = np.random.randn(3, 100)
        gender_direction = np.zeros(100)
        importance = np.array([0.3, 0.3, 0.4])
        
        with pytest.raises(ValueError, match="zero or near-zero magnitude"):
            sentence_bias(word_embeddings, gender_direction, importance)

    def test_sentence_bias_non_boolean_mask(self):
        """Test sentence_bias with non-boolean mask values."""
        word_embeddings = np.random.randn(3, 100)
        gender_direction = np.random.randn(100)
        importance = np.array([0.3, 0.3, 0.4])
        mask = np.array([1, 0, 1])  # Integers
        
        with pytest.raises(TypeError, match="must be boolean array"):
            sentence_bias(word_embeddings, gender_direction, importance, mask)


class TestIntegration:
    """Integration tests for bias metrics."""

    def test_metrics_consistency(self):
        """Test that metrics produce consistent results across runs."""
        # Fix seed
        np.random.seed(42)
        
        word_embeddings = np.random.randn(5, 300)
        gender_direction = np.random.randn(300)
        importance = np.array([0.2] * 5)
        
        fb1, mb1 = sentence_bias(word_embeddings, gender_direction, importance)
        fb2, mb2 = sentence_bias(word_embeddings, gender_direction, importance)
        
        assert fb1 == fb2
        assert mb1 == mb2
