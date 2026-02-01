"""Tests for embedding-based bias detection metrics."""

import pytest
import numpy as np
import torch
from bias_scope.embeddings_based_metrics import weat, seat, sentence_bias


class TestWEAT:
    """Test Word Embedding Association Test."""
    
    def test_basic_functionality(self):
        """Test WEAT with simple inputs."""
        target1 = np.array([[1.0, 0.0], [0.9, 0.1]])
        target2 = np.array([[0.0, 1.0], [0.1, 0.9]])
        attr1 = np.array([[1.0, 0.0], [0.95, 0.05]])
        attr2 = np.array([[0.0, 1.0], [0.05, 0.95]])
        
        score = weat((target1, target2), (attr1, attr2))
        
        assert isinstance(score, float)
        assert not np.isnan(score)
        assert score > 0
    
    def test_with_torch_tensors(self):
        """Test WEAT handles PyTorch tensors."""
        target1 = torch.randn(3, 10)
        target2 = torch.randn(3, 10)
        attr1 = torch.randn(2, 10)
        attr2 = torch.randn(2, 10)
        
        score = weat((target1, target2), (attr1, attr2))
        assert isinstance(score, float)
    
    def test_identical_targets(self):
        """Test WEAT with identical target sets returns near zero."""
        target = np.random.randn(5, 20)
        attr1 = np.random.randn(3, 20)
        attr2 = np.random.randn(3, 20)
        
        score = weat((target, target), (attr1, attr2))
        assert abs(score) < 0.01
    
    def test_raises_on_wrong_input(self):
        """Test WEAT validates input format."""
        target = np.random.randn(3, 10)
        attr = np.random.randn(2, 10)
        
        with pytest.raises(ValueError, match="must have two elements"):
            weat((target,), (attr, attr))
        
        with pytest.raises(ValueError, match="must have two elements"):
            weat((target, target), (attr,))


class TestSEAT:
    """Test Sentence Encoder Association Test."""
    
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


class TestSentenceBias:
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
        
        female_bias, male_bias = sentence_bias(
            word_embeddings, gender_direction, importance
        )
        
        assert isinstance(female_bias, float)
        assert isinstance(male_bias, float)
        assert female_bias > 0
        assert male_bias < 0
    
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
