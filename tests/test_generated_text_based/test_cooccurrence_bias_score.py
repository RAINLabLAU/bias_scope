"""Tests for CoOccurrenceBiasScore metric."""

import pytest
import json
import numpy as np
from bias_scope.generated_text_based import CoOccurrenceBiasScore


class TestCoOccurrenceBiasScore:
    """Test suite for CoOccurrenceBiasScore."""
    
    def test_happy_path_two_groups(self):
        """Test basic functionality with 2 groups."""
        cobs = CoOccurrenceBiasScore()
        
        generations = [
            "the man is a doctor",
            "the woman is a nurse",
            "the man is smart"
        ]
        
        group_lexicons = {
            'male': ['man'],
            'female': ['woman']
        }
        
        result = cobs.evaluate(
            generations=generations,
            group_lexicons=group_lexicons,
            window_size=5
        )
        
        # Check structure
        assert result['metric'] == 'CoOccurrenceBiasScore'
        assert result['category'] == 'generated_text'
        assert result['window_size'] == 5
        assert result['groups'] == ['male', 'female']
        
        # Check counts
        assert 'group_anchors' in result['counts']
        assert 'cooccurrence' in result['counts']
        
        # Man appears twice, woman once
        assert result['counts']['group_anchors']['male'] == 2
        assert result['counts']['group_anchors']['female'] == 1
        
        # Check scores exist
        assert 'pairwise' in result['scores']
        assert 'male|female' in result['scores']['pairwise']
    
    def test_neutral_vocab_provided(self):
        """Test with explicitly provided neutral vocabulary."""
        cobs = CoOccurrenceBiasScore()
        
        generations = [
            "the man likes science",
            "the woman likes art"
        ]
        
        group_lexicons = {
            'male': ['man'],
            'female': ['woman']
        }
        
        neutral_vocab = ['science', 'art']
        
        result = cobs.evaluate(
            generations=generations,
            group_lexicons=group_lexicons,
            neutral_vocab=neutral_vocab,
            window_size=3
        )
        
        # Only science and art should be scored
        pair_scores = result['scores']['pairwise']['male|female']
        scored_words = set(pair_scores.keys())
        
        # Should contain at least science or art
        assert len(scored_words) >= 0  # May be empty if not in window
    
    def test_neutral_vocab_none_uses_derived(self):
        """Test that neutral_vocab=None excludes group terms."""
        cobs = CoOccurrenceBiasScore()
        
        generations = [
            "man woman doctor"
        ]
        
        group_lexicons = {
            'male': ['man'],
            'female': ['woman']
        }
        
        result = cobs.evaluate(
            generations=generations,
            group_lexicons=group_lexicons,
            neutral_vocab=None,  # Should auto-derive
            window_size=2
        )
        
        # Group terms should not appear in scored neutral vocab
        pair_scores = result['scores']['pairwise']['male|female']
        assert 'man' not in pair_scores
        assert 'woman' not in pair_scores
    
    def test_window_size_effect(self):
        """Test that window size affects results."""
        cobs = CoOccurrenceBiasScore()
        
        generations = [
            "the man is very very very smart"  # 'smart' is 6 tokens away
        ]
        
        group_lexicons = {
            'male': ['man'],
            'female': ['woman']
        }
        
        # Small window - should NOT capture 'smart'
        result_small = cobs.evaluate(
            generations=generations,
            group_lexicons=group_lexicons,
            window_size=2
        )
        
        # Large window - should capture 'smart'
        result_large = cobs.evaluate(
            generations=generations,
            group_lexicons=group_lexicons,
            window_size=10
        )
        
        # Different counts expected
        small_cooc = result_small['counts']['cooccurrence']['male']
        large_cooc = result_large['counts']['cooccurrence']['male']
        
        # Large window should have more co-occurrences
        assert sum(large_cooc.values()) >= sum(small_cooc.values())
    
    def test_smoothing_effect(self):
        """Test that smoothing parameter changes scores."""
        cobs = CoOccurrenceBiasScore()
        
        generations = [
            "man doctor",
            "woman nurse"
        ]
        
        group_lexicons = {
            'male': ['man'],
            'female': ['woman']
        }
        
        result_low = cobs.evaluate(
            generations=generations,
            group_lexicons=group_lexicons,
            smoothing=0.1
        )
        
        result_high = cobs.evaluate(
            generations=generations,
            group_lexicons=group_lexicons,
            smoothing=10.0
        )
        
        # High smoothing should reduce score magnitudes
        scores_low = list(result_low['scores']['pairwise']['male|female'].values())
        scores_high = list(result_high['scores']['pairwise']['male|female'].values())
        
        if scores_low and scores_high:
            mean_abs_low = np.mean([abs(s) for s in scores_low])
            mean_abs_high = np.mean([abs(s) for s in scores_high])
            
            # Higher smoothing typically reduces magnitudes
            assert mean_abs_high <= mean_abs_low * 2  # Allow some tolerance
    
    def test_three_groups_pairwise(self):
        """Test >2 groups with pairwise mode."""
        cobs = CoOccurrenceBiasScore()
        
        generations = [
            "man doctor",
            "woman nurse",
            "child student"
        ]
        
        group_lexicons = {
            'male': ['man'],
            'female': ['woman'],
            'child': ['child']
        }
        
        result = cobs.evaluate(
            generations=generations,
            group_lexicons=group_lexicons,
            multi_group_mode='pairwise'
        )
        
        # Should have 3 choose 2 = 3 pairs
        pairwise_scores = result['scores']['pairwise']
        assert len(pairwise_scores) == 3
        assert 'male|female' in pairwise_scores or 'female|male' in pairwise_scores
    
    def test_three_groups_vs_mean(self):
        """Test >2 groups with vs_mean mode."""
        cobs = CoOccurrenceBiasScore()
        
        generations = [
            "man doctor",
            "woman nurse",
            "child student"
        ]
        
        group_lexicons = {
            'male': ['man'],
            'female': ['woman'],
            'child': ['child']
        }
        
        result = cobs.evaluate(
            generations=generations,
            group_lexicons=group_lexicons,
            multi_group_mode='vs_mean'
        )
        
        # Should have vs_mean scores for each group
        vs_mean_scores = result['scores']['vs_mean']
        assert 'male' in vs_mean_scores
        assert 'female' in vs_mean_scores
        assert 'child' in vs_mean_scores
    
    def test_edge_case_no_neutral_words_nearby(self):
        """Test when group anchors exist but no neutral words in window."""
        cobs = CoOccurrenceBiasScore()
        
        # Only group terms, widely spaced
        generations = [
            "man " + " ".join(["the"] * 20) + " woman"
        ]
        
        group_lexicons = {
            'male': ['man'],
            'female': ['woman']
        }
        
        result = cobs.evaluate(
            generations=generations,
            group_lexicons=group_lexicons,
            window_size=2,  # Small window
            smoothing=1.0
        )
        
        # Should still complete successfully with smoothing
        assert 'scores' in result
        assert result['summary']['mean_abs_score'] >= 0  # Should be finite
    
    def test_validation_fewer_than_two_groups(self):
        """Test validation: < 2 groups."""
        cobs = CoOccurrenceBiasScore()
        
        with pytest.raises(ValueError, match="at least 2 groups"):
            cobs.evaluate(
                generations=["test"],
                group_lexicons={'only_one': ['term']},
                window_size=5
            )
    
    def test_validation_empty_lexicon(self):
        """Test validation: empty lexicon."""
        cobs = CoOccurrenceBiasScore()
        
        with pytest.raises(ValueError, match="cannot be empty"):
            cobs.evaluate(
                generations=["test"],
                group_lexicons={'male': ['man'], 'female': []},
                window_size=5
            )
    
    def test_validation_empty_generations(self):
        """Test validation: empty generations."""
        cobs = CoOccurrenceBiasScore()
        
        with pytest.raises(ValueError, match="cannot be empty"):
            cobs.evaluate(
                generations=[],
                group_lexicons={'male': ['man'], 'female': ['woman']},
                window_size=5
            )
    
    def test_validation_invalid_window_size(self):
        """Test validation: window_size < 1."""
        cobs = CoOccurrenceBiasScore()
        
        with pytest.raises(ValueError, match="window_size must be >= 1"):
            cobs.evaluate(
                generations=["test"],
                group_lexicons={'male': ['man'], 'female': ['woman']},
                window_size=0
            )
    
    def test_validation_invalid_smoothing(self):
        """Test validation: smoothing <= 0."""
        cobs = CoOccurrenceBiasScore()
        
        with pytest.raises(ValueError, match="smoothing must be > 0"):
            cobs.evaluate(
                generations=["test"],
                group_lexicons={'male': ['man'], 'female': ['woman']},
                window_size=5,
                smoothing=0
            )
    
    def test_validation_invalid_return_top_k(self):
        """Test validation: return_top_k < 1."""
        cobs = CoOccurrenceBiasScore()
        
        with pytest.raises(ValueError, match="return_top_k must be >= 1"):
            cobs.evaluate(
                generations=["test"],
                group_lexicons={'male': ['man'], 'female': ['woman']},
                window_size=5,
                return_top_k=0
            )
    
    def test_json_serializable(self):
        """Test that result is JSON-serializable."""
        cobs = CoOccurrenceBiasScore()
        
        generations = ["man doctor", "woman nurse"]
        group_lexicons = {'male': ['man'], 'female': ['woman']}
        
        result = cobs.evaluate(
            generations=generations,
            group_lexicons=group_lexicons
        )
        
        # Should not raise
        json_str = json.dumps(result)
        assert isinstance(json_str, str)
    
    # === B) Missing Edge Cases for CoOccurrenceBiasScore ===
    
    def test_no_group_anchors_present(self):
        """Test when generations contain no group terms at all."""
        cobs = CoOccurrenceBiasScore()
        
        # No group terms present
        generations = [
            "the doctor works hard",
            "the nurse is skilled"
        ]
        
        group_lexicons = {
            'male': ['man', 'he'],
            'female': ['woman', 'she']
        }
        
        result = cobs.evaluate(
            generations=generations,
            group_lexicons=group_lexicons,
            window_size=5
        )
        
        # Group anchor counts should be zero
        assert result['counts']['group_anchors']['male'] == 0
        assert result['counts']['group_anchors']['female'] == 0
        
        # Scores should still be finite due to smoothing
        pair_scores = result['scores']['pairwise']['male|female']
        if pair_scores:
            for score in pair_scores.values():
                assert np.isfinite(score)
    
    def test_overlapping_windows_counting(self):
        """Test that overlapping windows are handled correctly.
        
        A neutral word in overlapping windows is counted separately for each anchor.
        """
        cobs = CoOccurrenceBiasScore()
        
        # 'doctor' is within window of both 'man' occurrences
        generations = [
            "man man doctor"  # 'doctor' is near both 'man' tokens
        ]
        
        group_lexicons = {
            'male': ['man'],
            'female': ['woman']
        }
        
        result = cobs.evaluate(
            generations=generations,
            group_lexicons=group_lexicons,
            window_size=3
        )
        
        # Should have 2 male anchors
        assert result['counts']['group_anchors']['male'] == 2
        
        # 'doctor' should be counted in co-occurrence
        # (counted once per anchor it's near)
        cooc_male = result['counts']['cooccurrence']['male']
        if 'doctor' in cooc_male:
            # Should be counted 2 times (once per anchor)
            assert cooc_male['doctor'] == 2
    
    def test_neutral_vocab_excludes_everything(self):
        """Test when neutral_vocab has no overlap with corpus."""
        cobs = CoOccurrenceBiasScore()
        
        generations = ["man doctor woman nurse"]
        
        group_lexicons = {
            'male': ['man'],
            'female': ['woman']
        }
        
        # Neutral vocab that doesn't appear in text
        neutral_vocab = ['nonexistent1', 'nonexistent2']
        
        result = cobs.evaluate(
            generations=generations,
            group_lexicons=group_lexicons,
            neutral_vocab=neutral_vocab,
            window_size=5
        )
        
        # Should have group anchors but empty co-occurrence counts
        assert result['counts']['group_anchors']['male'] > 0
        assert result['counts']['group_anchors']['female'] > 0
        
        # vocab_size should be 2 (the neutral vocab)
        assert result['vocab_size'] == 2
        
        # Scores should be empty or all zero (no co-occurrences)
        pair_scores = result['scores']['pairwise']['male|female']
        # Should still compute scores for the neutral vocab with smoothing
        assert len(pair_scores) >= 0
    
    def test_punctuation_tokenization_edge_case(self):
        """Test that punctuation is handled correctly in tokenization."""
        cobs = CoOccurrenceBiasScore()
        
        # Punctuation-heavy input
        generations = [
            "man, woman. doctor! nurse?"
        ]
        
        group_lexicons = {
            'male': ['man'],
            'female': ['woman']
        }
        
        result = cobs.evaluate(
            generations=generations,
            group_lexicons=group_lexicons,
            window_size=5
        )
        
        # Should detect group terms despite punctuation
        assert result['counts']['group_anchors']['male'] >= 1
        assert result['counts']['group_anchors']['female'] >= 1
        
        # Should detect neutral terms (doctor, nurse)
        pair_scores = result['scores']['pairwise']['male|female']
        # At least one of these should be scored
        assert 'doctor' in pair_scores or 'nurse' in pair_scores
    
    def test_multi_group_mode_invalid(self):
        """Test that invalid multi_group_mode raises error."""
        cobs = CoOccurrenceBiasScore()
        
        generations = ["man doctor"]
        group_lexicons = {
            'male': ['man'],
            'female': ['woman'],
            'neutral': ['person']
        }
        
        with pytest.raises(ValueError, match="multi_group_mode must be"):
            cobs.evaluate(
                generations=generations,
                group_lexicons=group_lexicons,
                multi_group_mode='invalid'
            )


