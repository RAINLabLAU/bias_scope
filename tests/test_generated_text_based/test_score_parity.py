"""Tests for Score Parity."""

import pytest
import numpy as np
from bias_scope.generated_text_based import ScoreParity


class TestScoreParity:
    """Test suite for Score Parity metric."""
    
    @pytest.fixture
    def simple_classifier(self):
        """Create simple mock classifier."""
        def classifier(texts):
            # Simple: "high" = 0.9, "low" = 0.1, else 0.5
            scores = []
            for text in texts:
                if "high" in text.lower():
                    scores.append(0.9)
                elif "low" in text.lower():
                    scores.append(0.1)
                else:
                    scores.append(0.5)
            return scores
        return classifier
    
    def test_basic_functionality(self, simple_classifier):
        """Test basic score parity computation."""
        parity = ScoreParity(classifier=simple_classifier)
        
        group_a = [["high score", "high value"]]
        group_b = [["low score", "low value"]]
        
        scores = parity.evaluate(group_a, group_b)
        
        # Group A: 0.9, Group B: 0.1, Diff: 0.8
        assert abs(scores['difference'] - 0.8) < 0.01
        assert abs(scores['group_a_mean'] - 0.9) < 0.01
        assert abs(scores['group_b_mean'] - 0.1) < 0.01
    
    def test_equal_groups(self, simple_classifier):
        """Test when both groups have same scores."""
        parity = ScoreParity(classifier=simple_classifier)
        
        group_a = [["high", "high"]]
        group_b = [["high", "high"]]
        
        scores = parity.evaluate(group_a, group_b)
        
        assert abs(scores['difference']) < 0.01
        assert abs(scores['group_a_mean'] - scores['group_b_mean']) < 0.01
    
    def test_non_callable_classifier_raises_error(self):
        """Test that non-callable classifier raises ValueError."""
        with pytest.raises(ValueError, match="callable"):
            ScoreParity(classifier="not_callable")
    
    def test_empty_group_raises_error(self, simple_classifier):
        """Test that empty group raises ValueError."""
        parity = ScoreParity(classifier=simple_classifier)
        
        group_a = [["test"]]
        group_b = []
        
        with pytest.raises(ValueError, match="cannot be empty"):
            parity.evaluate(group_a, group_b)
    
    def test_empty_inner_list_raises_error(self, simple_classifier):
        """Test that empty inner list raises ValueError."""
        parity = ScoreParity(classifier=simple_classifier)
        
        group_a = [["test"]]
        group_b = [[]]
        
        with pytest.raises(ValueError, match="cannot be empty"):
            parity.evaluate(group_a, group_b)
    
    def test_return_keys(self, simple_classifier):
        """Test that all expected keys are in result."""
        parity = ScoreParity(classifier=simple_classifier)
        
        group_a = [["high"]]
        group_b = [["low"]]
        
        scores = parity.evaluate(group_a, group_b)
        
        expected_keys = [
            'difference', 'group_a_mean', 'group_b_mean',
            'group_a_std', 'group_b_std', 'effect_size'
        ]
        
        for key in expected_keys:
            assert key in scores
    
    def test_metadata_category(self, simple_classifier):
        """Test metric category property."""
        parity = ScoreParity(classifier=simple_classifier)
        assert parity.category == "generated_text"
    
    def test_effect_size_calculation(self, simple_classifier):
        """Test Cohen's d effect size calculation."""
        parity = ScoreParity(classifier=simple_classifier)

        # Need variation within groups for non-zero pooled std
        group_a = [["high", "high", "medium"]]
        group_b = [["low", "low", "medium"]]

        scores = parity.evaluate(group_a, group_b)

        # Effect size should be large (> 0.8)
        assert abs(scores['effect_size']) > 0.8
    
    def test_standard_deviation(self, simple_classifier):
        """Test standard deviation calculation."""
        parity = ScoreParity(classifier=simple_classifier)
        
        # Group with variation
        group_a = [["high", "low", "medium"]]
        group_b = [["high", "high", "high"]]
        
        scores = parity.evaluate(group_a, group_b)
        
        # Group A should have higher std
        assert scores['group_a_std'] > scores['group_b_std']
    
    def test_multiple_prompts(self, simple_classifier):
        """Test with multiple prompts per group."""
        parity = ScoreParity(classifier=simple_classifier)
        
        group_a = [
            ["high", "high"],
            ["high", "high"]
        ]
        group_b = [
            ["low", "low"],
            ["low", "low"]
        ]
        
        scores = parity.evaluate(group_a, group_b)
        
        # All A: 0.9, all B: 0.1
        assert abs(scores['group_a_mean'] - 0.9) < 0.01
        assert abs(scores['group_b_mean'] - 0.1) < 0.01
    
    def test_return_type_is_dict(self, simple_classifier):
        """Test that evaluate returns dictionary."""
        parity = ScoreParity(classifier=simple_classifier)
        
        group_a = [["test"]]
        group_b = [["test"]]
        
        scores = parity.evaluate(group_a, group_b)
        assert isinstance(scores, dict)
    
    def test_all_values_are_floats(self, simple_classifier):
        """Test that all returned values are floats."""
        parity = ScoreParity(classifier=simple_classifier)
        
        group_a = [["high"]]
        group_b = [["low"]]
        
        scores = parity.evaluate(group_a, group_b)
        
        for value in scores.values():
            assert isinstance(value, float)
    
    def test_difference_consistency(self, simple_classifier):
        """Test that difference matches mean difference."""
        parity = ScoreParity(classifier=simple_classifier)
        
        group_a = [["high"]]
        group_b = [["low"]]
        
        scores = parity.evaluate(group_a, group_b)
        
        expected_diff = scores['group_a_mean'] - scores['group_b_mean']
        assert abs(scores['difference'] - expected_diff) < 0.01
    
    def test_single_text_per_group(self, simple_classifier):
        """Test with minimal input."""
        parity = ScoreParity(classifier=simple_classifier)
        
        group_a = [["high"]]
        group_b = [["low"]]
        
        scores = parity.evaluate(group_a, group_b)
        
        assert scores['group_a_mean'] == 0.9
        assert scores['group_b_mean'] == 0.1
        assert abs(scores['difference'] - 0.8) < 0.01
    
    def test_custom_classifier(self):
        """Test with custom classifier function."""
        def word_count_classifier(texts):
            # Score based on word count (normalized by fixed max)
            # Use fixed max=10 to ensure consistent scoring across groups
            return [min(len(text.split()) / 10.0, 1.0) for text in texts]
        
        parity = ScoreParity(classifier=word_count_classifier)
        
        group_a = [["one two three four five"]]
        group_b = [["one"]]
        
        scores = parity.evaluate(group_a, group_b)
        
        # Group A: 0.5, Group B: 0.1
        assert scores['group_a_mean'] > scores['group_b_mean']
        assert scores['difference'] > 0
    
    def test_invalid_scores_raise_error(self):
        """Test that invalid classifier scores raise error."""
        def bad_classifier(texts):
            return [1.5] * len(texts)  # Out of range
        
        parity = ScoreParity(classifier=bad_classifier)
        
        group_a = [["test"]]
        group_b = [["test"]]
        
        with pytest.raises(ValueError, match="must be in"):
            parity.evaluate(group_a, group_b)
