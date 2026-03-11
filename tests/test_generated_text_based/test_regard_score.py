"""Tests for Regard Score."""

import pytest
from unittest.mock import patch


# --- Mock helpers ---

_POSITIVE_WORDS = {"good", "great", "wonderful", "excellent"}
_NEGATIVE_WORDS = {"bad", "terrible", "awful", "horrible"}


def _classify_text(text):
    """Return a HuggingFace-style label based on keywords in text."""
    lower = text.lower()
    for word in _POSITIVE_WORDS:
        if word in lower:
            return "POSITIVE"
    for word in _NEGATIVE_WORDS:
        if word in lower:
            return "NEGATIVE"
    return "NEUTRAL"


def _make_mock_classifier():
    """Return a callable that mimics a HuggingFace sentiment pipeline.

    Returns results in the ``return_all_scores=True`` format:
    a list (one entry per text) of lists of label-score dicts.
    """
    def _classifier(texts):
        return [[{"label": _classify_text(t), "score": 0.9}] for t in texts]

    return _classifier


@patch("bias_scope.generated_text_based.regard_score.pipeline")
class TestRegardScore:
    """Test suite for Regard Score metric."""

    def _make_metric(self, mock_pipeline):
        """Create RegardScore instance with mocked HuggingFace pipeline."""
        mock_pipeline.return_value = _make_mock_classifier()
        from bias_scope.generated_text_based import RegardScore
        return RegardScore()

    def test_basic_functionality(self, mock_pipeline):
        """Test basic regard score computation."""
        mock_regard = self._make_metric(mock_pipeline)
        group_a = [["This is good", "Great work"]]
        group_b = [["This is bad", "Terrible"]]

        scores = mock_regard.evaluate(group_a, group_b)

        # Group A: positive, Group B: negative
        assert scores['positive_difference'] > 0  # A more positive
        assert scores['negative_difference'] < 0  # B more negative

    def test_equal_groups(self, mock_pipeline):
        """Test when both groups have same sentiment."""
        mock_regard = self._make_metric(mock_pipeline)
        group_a = [["This is good", "Great"]]
        group_b = [["This is good", "Great"]]

        scores = mock_regard.evaluate(group_a, group_b)

        # Differences should be near zero
        assert abs(scores['positive_difference']) < 0.01
        assert abs(scores['negative_difference']) < 0.01

    def test_empty_group_raises_error(self, mock_pipeline):
        """Test that empty group raises ValueError."""
        mock_regard = self._make_metric(mock_pipeline)
        group_a = [["test"]]
        group_b = []

        with pytest.raises(ValueError, match="cannot be empty"):
            mock_regard.evaluate(group_a, group_b)

    def test_empty_inner_list_raises_error(self, mock_pipeline):
        """Test that empty inner list raises ValueError."""
        mock_regard = self._make_metric(mock_pipeline)
        group_a = [["test"]]
        group_b = [[]]

        with pytest.raises(ValueError, match="cannot be empty"):
            mock_regard.evaluate(group_a, group_b)

    def test_return_keys(self, mock_pipeline):
        """Test that difference keys are in result."""
        mock_regard = self._make_metric(mock_pipeline)
        group_a = [["good"]]
        group_b = [["bad"]]

        scores = mock_regard.evaluate(group_a, group_b)

        # Difference keys are always present
        assert 'positive_difference' in scores
        assert 'negative_difference' in scores

        # Group-level keys present for labels that appear
        assert 'group_a_positive' in scores or 'group_a_negative' in scores
        assert 'group_b_positive' in scores or 'group_b_negative' in scores

    def test_distributions_sum_to_one(self, mock_pipeline):
        """Test that sentiment distributions sum to 1."""
        mock_regard = self._make_metric(mock_pipeline)
        group_a = [["good", "bad", "hello"]]
        group_b = [["great", "terrible", "hi"]]

        scores = mock_regard.evaluate(group_a, group_b)

        # Group A distribution should sum to 1
        sum_a = sum(v for k, v in scores.items() if k.startswith('group_a_'))
        assert abs(sum_a - 1.0) < 0.01

        # Group B distribution should sum to 1
        sum_b = sum(v for k, v in scores.items() if k.startswith('group_b_'))
        assert abs(sum_b - 1.0) < 0.01

    def test_metadata_category(self, mock_pipeline):
        """Test metric category property."""
        mock_regard = self._make_metric(mock_pipeline)
        assert mock_regard.category == "generated_text"

    def test_multiple_prompts(self, mock_pipeline):
        """Test with multiple prompts per group."""
        mock_regard = self._make_metric(mock_pipeline)
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

    def test_mixed_sentiments(self, mock_pipeline):
        """Test with mixed sentiments in same group."""
        mock_regard = self._make_metric(mock_pipeline)
        group_a = [["This is wonderful", "This is terrible"]]
        group_b = [["This is great", "This is awful"]]

        scores = mock_regard.evaluate(group_a, group_b)

        # Each group has 1 positive and 1 negative
        assert abs(scores.get('group_a_positive', 0.0) - 0.5) < 0.1
        assert abs(scores.get('group_a_negative', 0.0) - 0.5) < 0.1

    def test_return_type_is_dict(self, mock_pipeline):
        """Test that evaluate returns dictionary."""
        mock_regard = self._make_metric(mock_pipeline)
        group_a = [["test"]]
        group_b = [["test"]]

        scores = mock_regard.evaluate(group_a, group_b)
        assert isinstance(scores, dict)

    def test_all_values_are_floats(self, mock_pipeline):
        """Test that all returned values are floats."""
        mock_regard = self._make_metric(mock_pipeline)
        group_a = [["good"]]
        group_b = [["bad"]]

        scores = mock_regard.evaluate(group_a, group_b)

        for value in scores.values():
            assert isinstance(value, float)

    def test_differences_are_consistent(self, mock_pipeline):
        """Test that differences match individual distributions."""
        mock_regard = self._make_metric(mock_pipeline)
        group_a = [["good"]]
        group_b = [["bad"]]

        scores = mock_regard.evaluate(group_a, group_b)

        # Check positive difference
        expected_pos_diff = scores.get('group_a_positive', 0.0) - scores.get('group_b_positive', 0.0)
        assert abs(scores['positive_difference'] - expected_pos_diff) < 0.01

        # Check negative difference
        expected_neg_diff = scores.get('group_a_negative', 0.0) - scores.get('group_b_negative', 0.0)
        assert abs(scores['negative_difference'] - expected_neg_diff) < 0.01

    def test_single_text_per_group(self, mock_pipeline):
        """Test with minimal input."""
        mock_regard = self._make_metric(mock_pipeline)
        group_a = [["good"]]
        group_b = [["bad"]]

        scores = mock_regard.evaluate(group_a, group_b)

        # Group A should be 100% positive
        assert scores['group_a_positive'] == 1.0
        # Group B should be 100% negative
        assert scores['group_b_negative'] == 1.0
