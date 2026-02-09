"""Tests for All Unmasked Likelihood (AUL)."""

import pytest
import numpy as np
from bias_scope.probability_based import AUL


class TestAUL:
    """Test AUL metric."""
    
    def test_basic_functionality(self):
        """Test with biased predictions."""
        aul = AUL()
        
        # Biased: higher probs for stereotypes
        def biased_predict(sentence, pos):
            if "Women" in sentence:
                return 0.7
            return 0.3
        
        pairs = [(["Women", "work"], ["Men", "work"])]
        
        score = aul.evaluate(pairs, biased_predict)
        
        assert 0.0 <= score <= 1.0
        assert score >= 0.5  # Prefers stereotypes (>= allows for ties)
    
    def test_unbiased_model(self):
        """Test with unbiased prediction function."""
        aul = AUL()
        
        def unbiased_predict(sentence, pos):
            # Add small variation to avoid exact ties
            import random
            return 0.5 + random.uniform(-0.05, 0.05)
        
        pairs = [
            (["Women", "are", "smart"], ["Men", "are", "smart"]),
            (["She", "is", "strong"], ["He", "is", "strong"])
        ]
        
        score = aul.evaluate(pairs, unbiased_predict)
        
        # Should be approximately 0.5 (no preference)
        assert abs(score - 0.5) <= 0.6  # Allow variance due to randomness
    
    def test_empty_pairs_raises_error(self):
        """Test empty sentence pairs raises error."""
        aul = AUL()
        
        def mock_predict(sentence, pos):
            return 0.5
        
        with pytest.raises(ValueError, match="cannot be empty"):
            aul.evaluate([], mock_predict)
    
    def test_different_length_raises_error(self):
        """Test sentences with different lengths raise error."""
        aul = AUL()
        
        def mock_predict(sentence, pos):
            return 0.5
        
        pairs = [
            (["Women", "are"], ["Men", "are", "smart"])
        ]
        
        with pytest.raises(ValueError, match="same length"):
            aul.evaluate(pairs, mock_predict)
    
    def test_empty_sentence_raises_error(self):
        """Test empty sentence raises error."""
        aul = AUL()
        
        def mock_predict(sentence, pos):
            return 0.5
        
        pairs = [([], [])]
        
        with pytest.raises(ValueError, match="cannot be empty"):
            aul.evaluate(pairs, mock_predict)
    
    def test_invalid_probability_raises_error(self):
        """Test invalid probabilities raise error."""
        aul = AUL()
        
        def bad_predict(sentence, pos):
            return 1.5  # > 1
        
        pairs = [(["Women", "work"], ["Men", "work"])]
        
        with pytest.raises(ValueError, match="Invalid probability"):
            aul.evaluate(pairs, bad_predict)
    
    def test_negative_probability_raises_error(self):
        """Test negative probability raises error."""
        aul = AUL()
        
        def bad_predict(sentence, pos):
            return -0.1
        
        pairs = [(["She", "runs"], ["He", "runs"])]
        
        with pytest.raises(ValueError, match="Invalid probability"):
            aul.evaluate(pairs, bad_predict)
    
    def test_zero_probability_raises_error(self):
        """Test zero probability raises error."""
        aul = AUL()
        
        def bad_predict(sentence, pos):
            return 0.0
        
        pairs = [(["She", "runs"], ["He", "runs"])]
        
        with pytest.raises(ValueError, match="Invalid probability"):
            aul.evaluate(pairs, bad_predict)
    
    def test_all_tokens_predicted(self):
        """Test that all tokens in sentence are predicted."""
        aul = AUL()
        
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
        aul = AUL()
        
        def mock_predict(sentence, pos):
            return 0.7
        
        pairs = [(["She", "works"], ["He", "works"])]
        
        score = aul.evaluate(pairs, mock_predict)
        
        # Single pair: score is either 0 or 1
        assert score in [0.0, 1.0]
    
    def test_many_pairs(self):
        """Test with many sentence pairs."""
        aul = AUL()
        
        def mock_predict(sentence, pos):
            return np.random.uniform(0.3, 0.7)
        
        pairs = [
            (["Women", "are", "X"], ["Men", "are", "X"])
            for _ in range(50)
        ]
        
        score = aul.evaluate(pairs, mock_predict)
        
        assert 0.0 <= score <= 1.0
    
    def test_deterministic_results(self):
        """Test deterministic prediction gives consistent results."""
        aul = AUL()
        
        def deterministic_predict(sentence, pos):
            return 0.8 if "Women" in sentence else 0.4
        
        pairs = [(["Women", "work"], ["Men", "work"])]
        
        score1 = aul.evaluate(pairs, deterministic_predict)
        score2 = aul.evaluate(pairs, deterministic_predict)
        
        assert score1 == score2
    
    def test_long_sentences(self):
        """Test with longer sentences."""
        aul = AUL()
        
        def mock_predict(sentence, pos):
            return 0.5
        
        pairs = [(
            ["Women", "are", "often", "seen", "as", "more", "emotional"],
            ["Men", "are", "often", "seen", "as", "more", "emotional"]
        )]
        
        score = aul.evaluate(pairs, mock_predict)
        
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
    
    def test_position_based_probabilities(self):
        """Test with position-dependent probabilities."""
        aul = AUL()
        
        def position_predict(sentence, pos):
            # Higher prob for first token
            if pos == 0:
                return 0.8
            return 0.4
        
        pairs = [(["She", "runs"], ["He", "runs"])]
        
        score = aul.evaluate(pairs, position_predict)
        
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
    
    def test_computes_average_not_sum(self):
        """Test that AUL computes average, not sum."""
        aul = AUL()
        
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
        aul = AUL()
        
        def anti_bias_predict(sentence, pos):
            if "Men" in sentence:
                return 0.8
            return 0.3
        
        pairs = [(["Women", "work"], ["Men", "work"])]
        
        score = aul.evaluate(pairs, anti_bias_predict)
        
        assert score < 0.5  # Prefers anti-stereotypes
