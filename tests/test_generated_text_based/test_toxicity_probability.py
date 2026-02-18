"""Tests for Toxicity Probability."""

import pytest
import numpy as np
from bias_scope.generated_text_based import ToxicityProbability


class TestToxicityProbability:
    """Test suite for Toxicity Probability metric."""
    
    @pytest.fixture
    def mock_tp(self):
        """Create TP instance with mocked API."""
        tp = ToxicityProbability(api_key="mock_key")
        
        # Mock the API client
        class MockPerspective:
            def score_text(self, text):
                # Deterministic scoring: "bad" = toxic
                return 0.9 if "bad" in text.lower() else 0.1
            
            def score_batch(self, texts):
                return [self.score_text(t) for t in texts]
        
        tp.perspective = MockPerspective()
        return tp
    
    def test_basic_functionality(self, mock_tp):
        """Test basic TP computation."""
        texts = [
            ["Good", "Bad", "Nice"],    # Has toxic
            ["Hello", "Thanks", "Hi"]   # No toxic
        ]
        
        score = mock_tp.evaluate(texts, threshold=0.5)
        
        # 1 out of 2 prompts has toxicity = 0.5
        assert score == 0.5
    
    def test_all_prompts_have_toxic(self, mock_tp):
        """Test when all prompts have at least one toxic text."""
        texts = [
            ["Good", "Bad", "Nice"],
            ["Hello", "Bad", "Hi"]
        ]
        
        score = mock_tp.evaluate(texts, threshold=0.5)
        assert score == 1.0
    
    def test_no_prompts_have_toxic(self, mock_tp):
        """Test when no prompts have toxic texts."""
        texts = [
            ["Good", "Nice", "Great"],
            ["Hello", "Thanks", "Hi"]
        ]
        
        score = mock_tp.evaluate(texts, threshold=0.5)
        assert score == 0.0
    
    def test_empty_texts_raises_error(self, mock_tp):
        """Test that empty input raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            mock_tp.evaluate([], threshold=0.5)
    
    def test_empty_inner_list_raises_error(self, mock_tp):
        """Test that empty inner list raises ValueError."""
        texts = [["Good"], []]
        
        with pytest.raises(ValueError, match="cannot be empty"):
            mock_tp.evaluate(texts, threshold=0.5)
    
    def test_invalid_threshold_too_high(self, mock_tp):
        """Test that threshold > 1 raises ValueError."""
        texts = [["test"]]
        
        with pytest.raises(ValueError, match="threshold"):
            mock_tp.evaluate(texts, threshold=1.5)
    
    def test_invalid_threshold_too_low(self, mock_tp):
        """Test that threshold < 0 raises ValueError."""
        texts = [["test"]]
        
        with pytest.raises(ValueError, match="threshold"):
            mock_tp.evaluate(texts, threshold=-0.1)
    
    def test_different_thresholds(self, mock_tp):
        """Test that different thresholds give different results."""
        texts = [["Bad"]]
        
        score_low = mock_tp.evaluate(texts, threshold=0.3)
        score_high = mock_tp.evaluate(texts, threshold=0.95)
        
        # "Bad" gets 0.9, so only threshold 0.3 catches it
        assert score_low == 1.0
        assert score_high == 0.0
    
    def test_multiple_prompts(self, mock_tp):
        """Test with multiple prompts."""
        texts = [
            ["Bad", "Good", "Nice"],    # Has toxic
            ["Good", "Great", "Nice"],  # No toxic
            ["Bad", "Bad", "Bad"],      # Has toxic
            ["Hello", "Hi", "Thanks"]   # No toxic
        ]
        
        score = mock_tp.evaluate(texts, threshold=0.5)
        
        # 2 out of 4 prompts have toxicity = 0.5
        assert score == 0.5
    
    def test_metadata_category(self, mock_tp):
        """Test metric category property."""
        assert mock_tp.category == "generated_text"
    
    def test_return_type_is_float(self, mock_tp):
        """Test that evaluate returns float."""
        texts = [["test"]]
        score = mock_tp.evaluate(texts, threshold=0.5)
        
        assert isinstance(score, float)
    
    def test_return_value_in_valid_range(self, mock_tp):
        """Test that result is in [0, 1]."""
        texts = [["Good", "Bad", "Nice"]]
        score = mock_tp.evaluate(texts, threshold=0.5)
        
        assert 0.0 <= score <= 1.0
    
    def test_single_prompt_single_text(self, mock_tp):
        """Test with minimal input."""
        texts = [["Good"]]
        score = mock_tp.evaluate(texts, threshold=0.5)
        
        assert score == 0.0
    
    def test_threshold_boundary_values(self, mock_tp):
        """Test threshold at boundary values."""
        texts = [["Bad"]]
        
        # Threshold 0 should catch everything
        score_zero = mock_tp.evaluate(texts, threshold=0.0)
        assert score_zero == 1.0
        
        # Threshold 1 should catch nothing
        score_one = mock_tp.evaluate(texts, threshold=1.0)
        assert score_one == 0.0
    
    def test_binary_nature(self, mock_tp):
        """Test that TP is binary (presence vs absence)."""
        # One toxic text vs many toxic texts should give same result
        texts_one = [["Bad", "Good", "Good"]]
        texts_many = [["Bad", "Bad", "Bad"]]
        
        score_one = mock_tp.evaluate(texts_one, threshold=0.5)
        score_many = mock_tp.evaluate(texts_many, threshold=0.5)
        
        # Both should be 1.0 (prompt has toxicity)
        assert score_one == score_many == 1.0
    
    def test_difference_from_fraction(self, mock_tp):
        """Test that TP differs from TF for same input."""
        from bias_scope.generated_text_based import ToxicityFraction
        
        tf = ToxicityFraction(api_key="mock_key")
        tf.perspective = mock_tp.perspective  # Use same mock
        
        texts = [
            ["Bad", "Good", "Good"],  # TF: 1/3, TP: 1
            ["Good", "Good", "Good"]  # TF: 0/3, TP: 0
        ]
        
        tp_score = mock_tp.evaluate(texts, threshold=0.5)
        tf_score = tf.evaluate(texts, threshold=0.5)
        
        # TP: (1 + 0) / 2 = 0.5
        # TF: (1/3 + 0/3) / 2 = 0.167
        assert tp_score == 0.5
        assert abs(tf_score - 0.167) < 0.01
        assert tp_score != tf_score
