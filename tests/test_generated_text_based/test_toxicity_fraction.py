"""Tests for Toxicity Fraction."""

import pytest
import numpy as np
from bias_scope.generated_text_based import ToxicityFraction


class TestToxicityFraction:
    """Test suite for Toxicity Fraction metric."""
    
    @pytest.fixture
    def mock_tf(self):
        """Create TF instance with mocked API."""
        tf = ToxicityFraction(api_key="mock_key")
        
        # Mock the API client
        class MockPerspective:
            def score_text(self, text):
                # Deterministic scoring: "bad" = toxic
                return 0.9 if "bad" in text.lower() else 0.1
            
            def score_batch(self, texts):
                return [self.score_text(t) for t in texts]
        
        tf.perspective = MockPerspective()
        return tf
    
    def test_basic_functionality(self, mock_tf):
        """Test basic TF computation."""
        texts = [
            ["Good", "Bad", "Nice"],    # 1/3 toxic
            ["Hello", "Thanks", "Hi"]   # 0/3 toxic
        ]
        
        score = mock_tf.evaluate(texts, threshold=0.5)
        
        expected = (1/3 + 0/3) / 2  # 0.167
        assert abs(score - expected) < 0.01
    
    def test_all_toxic(self, mock_tf):
        """Test when all texts are toxic."""
        texts = [
            ["Bad", "Bad", "Bad"],
            ["Bad", "Bad", "Bad"]
        ]
        
        score = mock_tf.evaluate(texts, threshold=0.5)
        assert score == 1.0
    
    def test_all_clean(self, mock_tf):
        """Test when all texts are clean."""
        texts = [
            ["Good", "Nice", "Great"],
            ["Hello", "Thanks", "Hi"]
        ]
        
        score = mock_tf.evaluate(texts, threshold=0.5)
        assert score == 0.0
    
    def test_empty_texts_raises_error(self, mock_tf):
        """Test that empty input raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            mock_tf.evaluate([], threshold=0.5)
    
    def test_empty_inner_list_raises_error(self, mock_tf):
        """Test that empty inner list raises ValueError."""
        texts = [["Good"], []]  # Second list is empty
        
        with pytest.raises(ValueError, match="cannot be empty"):
            mock_tf.evaluate(texts, threshold=0.5)
    
    def test_invalid_threshold_too_high(self, mock_tf):
        """Test that threshold > 1 raises ValueError."""
        texts = [["test"]]
        
        with pytest.raises(ValueError, match="threshold"):
            mock_tf.evaluate(texts, threshold=1.5)
    
    def test_invalid_threshold_too_low(self, mock_tf):
        """Test that threshold < 0 raises ValueError."""
        texts = [["test"]]
        
        with pytest.raises(ValueError, match="threshold"):
            mock_tf.evaluate(texts, threshold=-0.1)
    
    def test_different_thresholds(self, mock_tf):
        """Test that different thresholds give different results."""
        texts = [["Bad"]]
        
        score_low = mock_tf.evaluate(texts, threshold=0.3)
        score_high = mock_tf.evaluate(texts, threshold=0.95)
        
        # "Bad" gets 0.9, so only threshold 0.3 catches it
        assert score_low == 1.0
        assert score_high == 0.0
    
    def test_multiple_prompts_averaging(self, mock_tf):
        """Test correct averaging across prompts."""
        texts = [
            ["Bad", "Good", "Bad"],     # 2/3 = 0.667
            ["Good", "Good", "Good"],   # 0/3 = 0.0
            ["Bad", "Bad", "Bad"]       # 3/3 = 1.0
        ]
        
        score = mock_tf.evaluate(texts, threshold=0.5)
        
        expected = (2/3 + 0/3 + 3/3) / 3  # 0.556
        assert abs(score - expected) < 0.01
    
    def test_metadata_name(self, mock_tf):
        """Test metric name property."""
        assert mock_tf.name == "Toxicity Fraction"
    
    def test_metadata_category(self, mock_tf):
        """Test metric category property."""
        assert mock_tf.category == "generated_text"
    
    def test_return_type_is_float(self, mock_tf):
        """Test that evaluate returns float."""
        texts = [["test"]]
        score = mock_tf.evaluate(texts, threshold=0.5)
        
        assert isinstance(score, float)
    
    def test_return_value_in_valid_range(self, mock_tf):
        """Test that result is in [0, 1]."""
        texts = [["Good", "Bad", "Nice"]]
        score = mock_tf.evaluate(texts, threshold=0.5)
        
        assert 0.0 <= score <= 1.0
    
    def test_single_prompt_single_text(self, mock_tf):
        """Test with minimal input."""
        texts = [["Good"]]
        score = mock_tf.evaluate(texts, threshold=0.5)
        
        assert score == 0.0
    
    def test_threshold_boundary_values(self, mock_tf):
        """Test threshold at boundary values."""
        texts = [["Bad"]]
        
        # Threshold 0 should catch everything
        score_zero = mock_tf.evaluate(texts, threshold=0.0)
        assert score_zero == 1.0
        
        # Threshold 1 should catch nothing
        score_one = mock_tf.evaluate(texts, threshold=1.0)
        assert score_one == 0.0
    
    def test_mixed_case_sensitivity(self, mock_tf):
        """Test that scoring works with different cases."""
        texts = [["BAD", "bad", "Bad", "good"]]
        score = mock_tf.evaluate(texts, threshold=0.5)
        
        # All "bad" variants should be toxic (3/4 = 0.75)
        assert abs(score - 0.75) < 0.01
