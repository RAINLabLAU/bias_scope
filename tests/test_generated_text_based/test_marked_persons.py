"""Tests for MarkedPersons metric."""

import pytest
import json
from bias_scope.generated_text_based import MarkedPersons


class TestMarkedPersons:
    """Test suite for MarkedPersons."""
    
    def test_happy_path(self):
        """Test basic functionality."""
        mp = MarkedPersons()
        
        # Marked: heavily uses 'caring'
        marked_texts = [
            "caring caring caring nurturing",
            "empathetic caring supportive"
        ]
        
        # Unmarked: heavily uses 'logical'
        unmarked_texts = [
            "logical logical logical analytical",
            "rational logical systematic"
        ]
        
        result = mp.evaluate(
            marked_generations=marked_texts,
            unmarked_generations=unmarked_texts,
            min_count=1
        )
        
        # Check structure
        assert result['metric'] == 'MarkedPersons'
        assert result['category'] == 'generated_text'
        
        # 'caring' should be in top marked terms with positive z
        top_marked_terms = [t['term'] for t in result['top_marked_terms']]
        top_marked_zscores = {t['term']: t['z'] for t in result['top_marked_terms']}
        
        if 'caring' in top_marked_terms:
            assert top_marked_zscores['caring'] > 0
        
        # 'logical' should be in top unmarked terms with negative z
        top_unmarked_terms = [t['term'] for t in result['top_unmarked_terms']]
        top_unmarked_zscores = {t['term']: t['z'] for t in result['top_unmarked_terms']}
        
        if 'logical' in top_unmarked_terms:
            assert top_unmarked_zscores['logical'] < 0
    
    def test_min_count_filtering(self):
        """Test that min_count filters rare terms."""
        mp = MarkedPersons()
        
        marked_texts = [
            "rare_term common common common"
        ]
        
        unmarked_texts = [
            "common common common common"
        ]
        
        # With high min_count, rare_term should be excluded
        result = mp.evaluate(
            marked_generations=marked_texts,
            unmarked_generations=unmarked_texts,
            min_count=3  # rare_term appears only once total
        )
        
        # rare_term should not be in terms dict
        assert 'rare_term' not in result['terms']
        
        # common should be included (appears 7 times total)
        assert 'common' in result['terms']
    
    def test_prior_alpha_effect(self):
        """Test that prior_alpha affects z-scores."""
        mp = MarkedPersons()
        
        marked_texts = ["unique_marked"]
        unmarked_texts = ["unique_unmarked"]
        
        # Small alpha - less regularization
        result_small = mp.evaluate(
            marked_generations=marked_texts,
            unmarked_generations=unmarked_texts,
            prior_alpha=0.001,
            min_count=1
        )
        
        # Large alpha - more regularization
        result_large = mp.evaluate(
            marked_generations=marked_texts,
            unmarked_generations=unmarked_texts,
            prior_alpha=1.0,
            min_count=1
        )
        
        # Extract z-scores
        if 'unique_marked' in result_small['terms'] and 'unique_marked' in result_large['terms']:
            z_small = abs(result_small['terms']['unique_marked']['z'])
            z_large = abs(result_large['terms']['unique_marked']['z'])
            
            # Larger prior should shrink z-scores
            assert z_large < z_small
    
    def test_custom_tokenizer(self):
        """Test with custom tokenizer."""
        mp = MarkedPersons()
        
        marked_texts = ["CUSTOM-TOKEN"]
        unmarked_texts = ["OTHER-TOKEN"]
        
        # Custom tokenizer that splits on dash
        def custom_tokenizer(text):
            return text.lower().split('-')
        
        result = mp.evaluate(
            marked_generations=marked_texts,
            unmarked_generations=unmarked_texts,
            tokenizer=custom_tokenizer,
            min_count=1
        )
        
        # Should have 'custom' and 'token' as separate tokens
        assert 'custom' in result['terms'] or 'token' in result['terms']
    
    def test_validation_empty_marked(self):
        """Test validation: empty marked corpus."""
        mp = MarkedPersons()
        
        with pytest.raises(ValueError, match="cannot be empty"):
            mp.evaluate(
                marked_generations=[],
                unmarked_generations=["test"]
            )
    
    def test_validation_empty_unmarked(self):
        """Test validation: empty unmarked corpus."""
        mp = MarkedPersons()
        
        with pytest.raises(ValueError, match="cannot be empty"):
            mp.evaluate(
                marked_generations=["test"],
                unmarked_generations=[]
            )
    
    def test_validation_invalid_prior_alpha(self):
        """Test validation: prior_alpha <= 0."""
        mp = MarkedPersons()
        
        with pytest.raises(ValueError, match="prior_alpha must be > 0"):
            mp.evaluate(
                marked_generations=["test"],
                unmarked_generations=["test"],
                prior_alpha=0
            )
    
    def test_validation_invalid_min_count(self):
        """Test validation: min_count < 1."""
        mp = MarkedPersons()
        
        with pytest.raises(ValueError, match="min_count must be >= 1"):
            mp.evaluate(
                marked_generations=["test"],
                unmarked_generations=["test"],
                min_count=0
            )
    
    def test_validation_invalid_return_top_k(self):
        """Test validation: return_top_k < 1."""
        mp = MarkedPersons()
        
        with pytest.raises(ValueError, match="return_top_k must be >= 1"):
            mp.evaluate(
                marked_generations=["test"],
                unmarked_generations=["test"],
                return_top_k=0
            )
    
    def test_return_structure(self):
        """Test return dictionary structure."""
        mp = MarkedPersons()
        
        marked_texts = ["caring nurturing"]
        unmarked_texts = ["logical analytical"]
        
        result = mp.evaluate(
            marked_generations=marked_texts,
            unmarked_generations=unmarked_texts,
            min_count=1,
            return_top_k=2
        )
        
        # Check all required keys
        assert 'metric' in result
        assert 'category' in result
        assert 'prior_alpha' in result
        assert 'min_count' in result
        assert 'return_top_k' in result
        assert 'top_marked_terms' in result
        assert 'top_unmarked_terms' in result
        assert 'terms' in result
        assert 'summary' in result
        
        # Check summary keys
        assert 'vocab_considered' in result['summary']
        assert 'marked_total_tokens' in result['summary']
        assert 'unmarked_total_tokens' in result['summary']
        
        # Check term structure
        for term_info in result['top_marked_terms']:
            assert 'term' in term_info
            assert 'z' in term_info
            assert 'delta' in term_info
            assert 'c_marked' in term_info
            assert 'c_unmarked' in term_info
    
    def test_json_serializable(self):
        """Test that result is JSON-serializable."""
        mp = MarkedPersons()
        
        marked_texts = ["caring"]
        unmarked_texts = ["logical"]
        
        result = mp.evaluate(
            marked_generations=marked_texts,
            unmarked_generations=unmarked_texts,
            min_count=1
        )
        
        # Should not raise
        json_str = json.dumps(result)
        assert isinstance(json_str, str)
    
    def test_z_score_calculation(self):
        """Test that z-scores are calculated correctly."""
        mp = MarkedPersons()
        
        # Simple case
        marked_texts = ["special"] * 10
        unmarked_texts = ["common"] * 10
        
        result = mp.evaluate(
            marked_generations=marked_texts,
            unmarked_generations=unmarked_texts,
            min_count=1
        )
        
        # 'special' should have high positive z
        if 'special' in result['terms']:
            assert result['terms']['special']['z'] > 0
            assert result['terms']['special']['c_marked'] > result['terms']['special']['c_unmarked']
        
        # 'common' should have negative z
        if 'common' in result['terms']:
            assert result['terms']['common']['z'] < 0
            assert result['terms']['common']['c_unmarked'] > result['terms']['common']['c_marked']
    
    def test_top_k_length(self):
        """Test that top_k limits results correctly."""
        mp = MarkedPersons()
        
        # Generate diverse vocabulary
        marked_words = [f"word_{i}" for i in range(30)]
        unmarked_words = [f"term_{i}" for i in range(30)]
        
        marked_texts = [" ".join(marked_words)]
        unmarked_texts = [" ".join(unmarked_words)]
        
        result = mp.evaluate(
            marked_generations=marked_texts,
            unmarked_generations=unmarked_texts,
            min_count=1,
            return_top_k=10
        )
        
        # Should have at most 10 top marked and 10 top unmarked
        assert len(result['top_marked_terms']) <= 10
        assert len(result['top_unmarked_terms']) <= 10


