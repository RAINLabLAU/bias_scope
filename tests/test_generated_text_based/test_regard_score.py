"""Tests for Regard Score."""

from unittest.mock import patch

import pytest

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
        """Test that all returned values are floats, except 'n' (an int,
        per the "n": int(len(...)) convention run() relies on for its
        n > 0 guard - see base.py::_count_items)."""
        mock_regard = self._make_metric(mock_pipeline)
        group_a = [["good"]]
        group_b = [["bad"]]

        scores = mock_regard.evaluate(group_a, group_b)

        for key, value in scores.items():
            if key == "n":
                assert isinstance(value, int)
            else:
                assert isinstance(value, float)

    def test_differences_are_consistent(self, mock_pipeline):
        """Test that differences match individual distributions."""
        mock_regard = self._make_metric(mock_pipeline)
        group_a = [["good"]]
        group_b = [["bad"]]

        scores = mock_regard.evaluate(group_a, group_b)

        # Check positive difference
        expected_pos_diff = (
            scores.get('group_a_positive', 0.0) - scores.get('group_b_positive', 0.0)
        )
        assert abs(scores['positive_difference'] - expected_pos_diff) < 0.01

        # Check negative difference
        expected_neg_diff = (
            scores.get('group_a_negative', 0.0) - scores.get('group_b_negative', 0.0)
        )
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

    def test_headline_is_the_negative_regard_gap(self, mock_pipeline):
        """run() reports `negative_difference` = P(neg | A) - P(neg | B), the
        gap Sheng et al. actually report (REVIEW_LATER RL-062). The composite
        proposed in RL-090 was not adopted. All-positive-A vs all-negative-B
        hits the declared value_range's lower boundary exactly."""
        mock_regard = self._make_metric(mock_pipeline)
        group_a = [["good", "great"]]
        group_b = [["bad", "terrible"]]

        scores = mock_regard.evaluate(group_a, group_b)
        assert scores["negative_difference"] == pytest.approx(-1.0)
        assert "bias_score" not in scores

    def test_negative_difference_swap_antisymmetry(self, mock_pipeline):
        mock_regard = self._make_metric(mock_pipeline)
        group_a = [["good", "hello"]]
        group_b = [["bad", "terrible"]]

        forward = mock_regard.evaluate(group_a, group_b)
        backward = mock_regard.evaluate(group_b, group_a)

        assert forward["negative_difference"] == pytest.approx(-backward["negative_difference"])

    def test_negative_difference_is_zero_for_identical_groups(self, mock_pipeline):
        mock_regard = self._make_metric(mock_pipeline)
        group_a = [["good", "bad", "hello"]]
        group_b = [["good", "bad", "hello"]]

        scores = mock_regard.evaluate(group_a, group_b)
        assert scores["negative_difference"] == pytest.approx(0.0)

    def test_n_counts_all_flattened_texts(self, mock_pipeline):
        mock_regard = self._make_metric(mock_pipeline)
        group_a = [["good", "great"], ["hello"]]
        group_b = [["bad"]]

        scores = mock_regard.evaluate(group_a, group_b)
        assert scores["n"] == 4

    def test_run_no_longer_crashes(self, mock_pipeline):
        """Before the fix, run() always raised BiasScopeError: evaluate()'s
        dict had no 'bias_score'/'score'/'value'/'effect_size' key, and no
        'n'-like key either."""
        mock_regard = self._make_metric(mock_pipeline)
        group_a = [["good", "great"]]
        group_b = [["bad", "terrible"]]

        result = mock_regard.run(group_a, group_b, ci="none")
        expected = mock_regard.evaluate(group_a, group_b)
        assert result.score == pytest.approx(expected["negative_difference"])
        assert result.n == 4

    def test_run_has_no_bootstrap_ci_without_per_item(self, mock_pipeline):
        """RegardScore has no natural per-item score (it's a two-group
        distributional comparison, not a per-prompt statistic), so it
        degrades to ci='none' the same way WEAT/SEAT/CEAT/CBS do."""
        mock_regard = self._make_metric(mock_pipeline)
        group_a = [["good", "great"]]
        group_b = [["bad", "terrible"]]

        result = mock_regard.run(group_a, group_b)  # default bootstrap
        assert result.ci is None
        assert result.ci_method == "none"


class TestRegardScoreReportsHowManyTextsItScored:
    """RL-062: RegardScore returned no item count, so `run()` rejected it.

    `_count_items` found no recognised key among its sixteen numbers and
    raised "n must be positive, got 0" - the metric was unreachable through
    `run()`, `BiasSuite` and the agent, like CAT/ICAT before RL-061. `n` is
    unambiguous here: the number of generated texts actually classified,
    across both groups.
    """

    def test_details_carry_the_number_of_texts_classified(self):
        from bias_scope.generated_text_based import RegardScore

        metric = RegardScore()
        metric._score_sentiments = lambda texts: ["negative"] * len(texts)  # type: ignore
        result = metric.evaluate(
            group_a_texts=[["a one"], ["a two"]],
            group_b_texts=[["b one"], ["b two"], ["b three"]],
            return_details=True,
        )
        assert result["n"] == 5
