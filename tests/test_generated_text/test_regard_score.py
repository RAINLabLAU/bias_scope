"""Tests for Regard Score."""

import pytest
from bias_scope.generated_text import RegardScore


class TestRegardScore:
    """Test suite for Regard Score metric."""
    
    @pytest.fixture
    def mock_regard(self):
        """Create RegardScore instance with mocked classifier."""
        regard = RegardScore()
        
        # Mock the sentiment classifier
        class MockClassifier:
            def __call__(self, texts):
                results = []
                for text in texts:
                    # Simple mock: "good" = positive, "bad" = negative, else neutral
                    if "good" in text.lower() or "great" in text.lower():
                        results.append([
                            {'label': 'POSITIVE', 'score': 0.9},
                            {'label': 'NEGATIVE', 'score': 0.05},
                            {'label': 'NEUTRAL', 'score': 0.05}
                        ])
                    elif "bad" in text.lower() or "terrible" in text.lower():
                        results.append([
                            {'label': 'POSITIVE', 'score': 0.05},
                            {'label': 'NEGATIVE', 'score': 0.9},
                            {'label': 'NEUTRAL', 'score': 0.05}
                        ])
                    else:
                        results.append([
                            {'label': 'POSITIVE', 'score': 0.05},
                            {'label': 'NEGATIVE', 'score': 0.05},
                            {'label': 'NEUTRAL', 'score': 0.9}
                        ])
                return results
        
        regard.sentiment_classifier = MockClassifier()
        return regard
    
    def test_basic_functionality(self, mock_regard):
        """Test basic regard score computation."""
        group_a = [["This is good", "Great work"]]
        group_b = [["This is bad", "Terrible"]]
        
        scores = mock_regard.evaluate(group_a, group_b)
        
        # Group A: 2 positive, Group B: 2 negative
        assert scores['positive_diff'] > 0  # A more positive
        assert scores['negative_diff'] < 0  # B more negative
    
    def test_equal_groups(self, mock_regard):
        """Test when both groups have same sentiment."""
        group_a = [["This is good", "Great"]]
        group_b = [["This is good", "Great"]]
        
        scores = mock_regard.evaluate(group_a, group_b)
        
        # Differences should be near zero
        assert abs(scores['positive_diff']) < 0.01
        assert abs(scores['negative_diff']) < 0.01
        assert abs(scores['neutral_diff']) < 0.01
    
    def test_empty_group_raises_error(self, mock_regard):
        """Test that empty group raises ValueError."""
        group_a = [["test"]]
        group_b = []
        
        with pytest.raises(ValueError, match="cannot be empty"):
            mock_regard.evaluate(group_a, group_b)
    
    def test_empty_inner_list_raises_error(self, mock_regard):
        """Test that empty inner list raises ValueError."""
        group_a = [["test"]]
        group_b = [[]]
        
        with pytest.raises(ValueError, match="cannot be empty"):
            mock_regard.evaluate(group_a, group_b)
    
    def test_return_keys(self, mock_regard):
        """Test that all expected keys are in result."""
        group_a = [["good"]]
        group_b = [["bad"]]
        
        scores = mock_regard.evaluate(group_a, group_b)
        
        expected_keys = [
            'positive_diff', 'negative_diff', 'neutral_diff',
            'group_a_positive', 'group_a_negative', 'group_a_neutral',
            'group_b_positive', 'group_b_negative', 'group_b_neutral'
        ]
        
        for key in expected_keys:
            assert key in scores
    
    def test_distributions_sum_to_one(self, mock_regard):
        """Test that sentiment distributions sum to 1."""
        group_a = [["good", "bad", "hello"]]
        group_b = [["great", "terrible", "hi"]]
        
        scores = mock_regard.evaluate(group_a, group_b)
        
        # Group A distribution should sum to 1
        sum_a = (scores['group_a_positive'] + 
                 scores['group_a_negative'] + 
                 scores['group_a_neutral'])
        assert abs(sum_a - 1.0) < 0.01
        
        # Group B distribution should sum to 1
        sum_b = (scores['group_b_positive'] + 
                 scores['group_b_negative'] + 
                 scores['group_b_neutral'])
        assert abs(sum_b - 1.0) < 0.01
    
    def test_metadata_name(self, mock_regard):
        """Test metric name property."""
        assert mock_regard.name == "Regard Score"
    
    def test_metadata_category(self, mock_regard):
        """Test metric category property."""
        assert mock_regard.category == "generated_text"
    
    def test_metadata_complexity(self, mock_regard):
        """Test complexity rating property."""
        assert mock_regard.complexity == "medium"
    
    def test_metadata_reference(self, mock_regard):
        """Test reference contains key info."""
        ref = mock_regard.reference
        assert "Sheng" in ref
        assert "2019" in ref
        assert "Babysitter" in ref
    
    def test_multiple_prompts(self, mock_regard):
        """Test with multiple prompts per group."""
        group_a = [
            ["good", "great"],
            ["excellent", "wonderful"]
        ]
        group_b = [
            ["bad", "terrible"],
            ["awful", "horrible"]
        ]
        
        scores = mock_regard.evaluate(group_a, group_b)
        
        # All A texts positive, all B texts negative
        assert scores['group_a_positive'] > 0.9
        assert scores['group_b_negative'] > 0.9
    
    def test_mixed_sentiments(self, mock_regard):
        """Test with mixed sentiments in same group."""
        group_a = [["good", "bad", "hello"]]
        group_b = [["great", "terrible", "hi"]]
        
        scores = mock_regard.evaluate(group_a, group_b)
        
        # Each group has 1 positive, 1 negative, 1 neutral
        assert abs(scores['group_a_positive'] - 1/3) < 0.01
        assert abs(scores['group_a_negative'] - 1/3) < 0.01
        assert abs(scores['group_a_neutral'] - 1/3) < 0.01
    
    def test_return_type_is_dict(self, mock_regard):
        """Test that evaluate returns dictionary."""
        group_a = [["test"]]
        group_b = [["test"]]
        
        scores = mock_regard.evaluate(group_a, group_b)
        assert isinstance(scores, dict)
    
    def test_all_values_are_floats(self, mock_regard):
        """Test that all returned values are floats."""
        group_a = [["good"]]
        group_b = [["bad"]]
        
        scores = mock_regard.evaluate(group_a, group_b)
        
        for value in scores.values():
            assert isinstance(value, float)
    
    def test_differences_are_consistent(self, mock_regard):
        """Test that differences match individual distributions."""
        group_a = [["good"]]
        group_b = [["bad"]]
        
        scores = mock_regard.evaluate(group_a, group_b)
        
        # Check positive difference
        expected_pos_diff = scores['group_a_positive'] - scores['group_b_positive']
        assert abs(scores['positive_diff'] - expected_pos_diff) < 0.01
        
        # Check negative difference
        expected_neg_diff = scores['group_a_negative'] - scores['group_b_negative']
        assert abs(scores['negative_diff'] - expected_neg_diff) < 0.01
    
    def test_single_text_per_group(self, mock_regard):
        """Test with minimal input."""
        group_a = [["good"]]
        group_b = [["bad"]]
        
        scores = mock_regard.evaluate(group_a, group_b)
        
        # Group A should be 100% positive
        assert scores['group_a_positive'] == 1.0
        # Group B should be 100% negative
        assert scores['group_b_negative'] == 1.0
