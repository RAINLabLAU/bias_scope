"""Tests for Analogical Reasoning Bias metric."""

import pytest
from unittest.mock import patch, MagicMock


def make_completion_response(content):
    """Create a mock LiteLLM completion response."""
    resp = MagicMock()
    resp.choices = [MagicMock(message=MagicMock(content=content))]
    return resp


@patch("bias_scope.prompts_based.analogical_reasoning_bias.completion")
class TestAnalogicalReasoningBias:
    """Test suite for AnalogicalReasoningBias."""

    def _make_metric(self):
        """Helper to create metric instance."""
        from bias_scope.prompts_based import AnalogicalReasoningBias
        return AnalogicalReasoningBias(model_name="openai/gpt-4o", api_key="test-key")

    def test_basic_functionality(self, mock_completion):
        """Test basic metric returns correct keys and types."""
        mock_completion.return_value = make_completion_response("programmer")
        metric = self._make_metric()

        result = metric.evaluate(num_analogies=1)

        assert "stereotype_rate" in result
        assert "symmetry_rate" in result
        assert "per_analogy" in result
        assert isinstance(result["stereotype_rate"], float)
        assert isinstance(result["symmetry_rate"], float)
        assert isinstance(result["per_analogy"], dict)

    def test_invalid_num_analogies_zero(self, mock_completion):
        """Test num_analogies=0 raises ValueError."""
        metric = self._make_metric()

        with pytest.raises(ValueError, match="num_analogies"):
            metric.evaluate(num_analogies=0)

    def test_invalid_num_analogies_negative(self, mock_completion):
        """Test negative num_analogies raises ValueError."""
        metric = self._make_metric()

        with pytest.raises(ValueError, match="num_analogies"):
            metric.evaluate(num_analogies=-1)

    def test_invalid_num_analogies_float(self, mock_completion):
        """Test non-integer num_analogies raises ValueError."""
        metric = self._make_metric()

        with pytest.raises(ValueError, match="num_analogies"):
            metric.evaluate(num_analogies=2.5)

    def test_stereotype_detected(self, mock_completion):
        """Test when completion contains stereotype word, stereotyped=True."""
        mock_completion.side_effect = [
            make_completion_response("homemaker"),      # original prompt
            make_completion_response("programmer"),     # symmetric prompt
        ]
        metric = self._make_metric()

        result = metric.evaluate(num_analogies=1)
        prompt_key = metric.ANALOGY_TEMPLATES[0]["prompt"]

        assert result["per_analogy"][prompt_key]["stereotyped"] is True

    def test_stereotype_not_detected(self, mock_completion):
        """Test when completion contains neutral word, stereotyped=False."""
        mock_completion.side_effect = [
            make_completion_response("programmer"),     # original prompt
            make_completion_response("programmer"),     # symmetric prompt
        ]
        metric = self._make_metric()

        result = metric.evaluate(num_analogies=1)
        prompt_key = metric.ANALOGY_TEMPLATES[0]["prompt"]

        assert result["per_analogy"][prompt_key]["stereotyped"] is False

    def test_stereotype_rate_all_stereotyped(self, mock_completion):
        """Test when all completions are stereotyped, stereotype_rate=1.0."""
        # 2 analogies: each needs original + symmetric = 4 calls
        mock_completion.side_effect = [
            make_completion_response("homemaker"),      # analogy 1 original
            make_completion_response("something"),      # analogy 1 symmetric
            make_completion_response("nurse"),           # analogy 2 original
            make_completion_response("something"),      # analogy 2 symmetric
        ]
        metric = self._make_metric()

        result = metric.evaluate(num_analogies=2)

        assert result["stereotype_rate"] == 1.0

    def test_stereotype_rate_none_stereotyped(self, mock_completion):
        """Test when no completions are stereotyped, stereotype_rate=0.0."""
        mock_completion.side_effect = [
            make_completion_response("programmer"),     # analogy 1 original
            make_completion_response("programmer"),     # analogy 1 symmetric
            make_completion_response("doctor"),          # analogy 2 original
            make_completion_response("doctor"),          # analogy 2 symmetric
        ]
        metric = self._make_metric()

        result = metric.evaluate(num_analogies=2)

        assert result["stereotype_rate"] == 0.0

    def test_symmetry_detected(self, mock_completion):
        """Test when symmetric prompt returns neutral word, symmetric=True."""
        mock_completion.side_effect = [
            make_completion_response("homemaker"),      # original
            make_completion_response("programmer"),     # symmetric (neutral word)
        ]
        metric = self._make_metric()

        result = metric.evaluate(num_analogies=1)
        prompt_key = metric.ANALOGY_TEMPLATES[0]["prompt"]

        assert result["per_analogy"][prompt_key]["symmetric"] is True

    def test_symmetry_not_detected(self, mock_completion):
        """Test when symmetric prompt returns stereotype, symmetric=False."""
        mock_completion.side_effect = [
            make_completion_response("homemaker"),      # original
            make_completion_response("homemaker"),      # symmetric (not neutral)
        ]
        metric = self._make_metric()

        result = metric.evaluate(num_analogies=1)
        prompt_key = metric.ANALOGY_TEMPLATES[0]["prompt"]

        assert result["per_analogy"][prompt_key]["symmetric"] is False

    def test_symmetry_rate_all_symmetric(self, mock_completion):
        """Test symmetry_rate=1.0 when all symmetric prompts return neutral."""
        mock_completion.side_effect = [
            make_completion_response("homemaker"),       # analogy 1 original
            make_completion_response("programmer"),      # analogy 1 symmetric (neutral)
            make_completion_response("nurse"),            # analogy 2 original
            make_completion_response("doctor"),           # analogy 2 symmetric (neutral)
        ]
        metric = self._make_metric()

        result = metric.evaluate(num_analogies=2)

        assert result["symmetry_rate"] == 1.0

    def test_per_analogy_keys(self, mock_completion):
        """Test per_analogy is keyed by prompt string."""
        mock_completion.return_value = make_completion_response("test")
        metric = self._make_metric()

        result = metric.evaluate(num_analogies=2)

        assert metric.ANALOGY_TEMPLATES[0]["prompt"] in result["per_analogy"]
        assert metric.ANALOGY_TEMPLATES[1]["prompt"] in result["per_analogy"]

    def test_per_analogy_structure(self, mock_completion):
        """Test each per_analogy entry has expected keys."""
        mock_completion.return_value = make_completion_response("test")
        metric = self._make_metric()

        result = metric.evaluate(num_analogies=1)
        prompt_key = metric.ANALOGY_TEMPLATES[0]["prompt"]
        entry = result["per_analogy"][prompt_key]

        assert "completion" in entry
        assert "stereotyped" in entry
        assert "symmetric_completion" in entry
        assert "symmetric" in entry

    def test_litellm_call_count(self, mock_completion):
        """Test 2 calls per analogy (original + symmetric)."""
        mock_completion.return_value = make_completion_response("test")
        metric = self._make_metric()

        metric.evaluate(num_analogies=3)

        assert mock_completion.call_count == 6  # 3 * 2

    def test_is_stereotyped_true(self, mock_completion):
        """Test _is_stereotyped returns True when stereotype in completion."""
        metric = self._make_metric()

        assert metric._is_stereotyped("homemaker is typical", "homemaker") is True

    def test_is_stereotyped_false(self, mock_completion):
        """Test _is_stereotyped returns False when stereotype not present."""
        metric = self._make_metric()

        assert metric._is_stereotyped("programmer", "homemaker") is False

    def test_is_stereotyped_case_insensitive(self, mock_completion):
        """Test _is_stereotyped is case insensitive."""
        metric = self._make_metric()

        assert metric._is_stereotyped("HOMEMAKER", "homemaker") is True
        assert metric._is_stereotyped("homemaker", "HOMEMAKER") is True

    def test_is_symmetric_true(self, mock_completion):
        """Test _is_symmetric returns True when completion_b contains neutral."""
        metric = self._make_metric()
        template = metric.ANALOGY_TEMPLATES[0]

        assert metric._is_symmetric("homemaker", "programmer", template) is True

    def test_is_symmetric_false(self, mock_completion):
        """Test _is_symmetric returns False when completion_b has stereotype."""
        metric = self._make_metric()
        template = metric.ANALOGY_TEMPLATES[0]

        assert metric._is_symmetric("homemaker", "homemaker", template) is False

    def test_num_analogies_larger_than_list(self, mock_completion):
        """Test num_analogies larger than template list is handled gracefully."""
        mock_completion.return_value = make_completion_response("test")
        metric = self._make_metric()

        result = metric.evaluate(num_analogies=100)

        assert len(result["per_analogy"]) == len(metric.ANALOGY_TEMPLATES)

    def test_category_property(self, mock_completion):
        """Test category returns 'prompt_based'."""
        metric = self._make_metric()

        assert metric.category == "prompt_based"
