"""Tests for Demographic Representation Bias metric."""

import sys
from unittest.mock import MagicMock, patch

import pytest


def make_mock_dataset(rows):
    """Create a mock dataset that supports len, iteration, select, and filter."""
    mock = MagicMock()
    mock.__len__ = MagicMock(return_value=len(rows))
    mock.__iter__ = MagicMock(side_effect=lambda: iter(rows))
    mock.select = MagicMock(
        side_effect=lambda rng: make_mock_dataset([rows[i] for i in rng if i < len(rows)])
    )
    return mock


def make_completion_response(content):
    """Create a mock LiteLLM completion response."""
    resp = MagicMock()
    resp.choices = [MagicMock(message=MagicMock(content=content))]
    return resp


SAMPLE_ROWS = [
    {"tokens": ["The", "nurse", "said", "that", "she", "would", "help"]},
    {"tokens": ["The", "doctor", "said", "that", "he", "would", "help"]},
    {"tokens": ["The", "teacher", "said", "that", "she", "would", "help"]},
]


@patch("bias_scope.prompts_based.occupation_pronoun_skew.load_dataset")
class TestOccupationPronounSkew:
    """Test suite for OccupationPronounSkew."""

    def _make_metric(self, mock_load_dataset, rows=None):
        """Helper to create metric with mocked dataset.

        mock_load_dataset will be called inside evaluate(), so we configure
        its return_value here so that any evaluate() call in the test gets
        the desired mock dataset.
        """
        mock_load_dataset.return_value = make_mock_dataset(
            rows if rows is not None else SAMPLE_ROWS
        )
        from bias_scope.prompts_based.occupation_pronoun_skew import (
            OccupationPronounSkew,
        )
        return OccupationPronounSkew(model_name="openai/gpt-4o", api_key="test-key")

    @patch("bias_scope.prompts_based.occupation_pronoun_skew.completion")
    def test_basic_functionality(self, mock_completion, mock_load_dataset):
        """Test basic metric returns correct keys and types."""
        mock_completion.return_value = make_completion_response("She walked in.")
        metric = self._make_metric(mock_load_dataset)

        result = metric.evaluate(num_templates=2, num_samples=1, subset="type1_pro")

        assert "representation_ratio" in result
        assert "l1_distance" in result
        assert "per_occupation" in result
        assert isinstance(result["representation_ratio"], float)
        assert isinstance(result["l1_distance"], float)
        assert isinstance(result["per_occupation"], dict)

    @patch("bias_scope.prompts_based.occupation_pronoun_skew.completion")
    def test_empty_dataset_raises_error(self, mock_completion, mock_load_dataset):
        """Test that empty dataset raises ValueError."""
        metric = self._make_metric(mock_load_dataset, rows=[])

        with pytest.raises(ValueError, match="empty"):
            metric.evaluate(num_samples=5, subset="type1_pro")

    def test_invalid_num_samples_zero(self, mock_load_dataset):
        """Test num_samples=0 raises ValueError."""
        metric = self._make_metric(mock_load_dataset)

        with pytest.raises(ValueError, match="num_samples"):
            metric.evaluate(num_samples=0, subset="type1_pro")

    def test_invalid_num_samples_negative(self, mock_load_dataset):
        """Test negative num_samples raises ValueError."""
        metric = self._make_metric(mock_load_dataset)

        with pytest.raises(ValueError, match="num_samples"):
            metric.evaluate(num_samples=-3, subset="type1_pro")

    def test_invalid_num_templates_zero(self, mock_load_dataset):
        """Test num_templates=0 raises ValueError."""
        metric = self._make_metric(mock_load_dataset)

        with pytest.raises(ValueError, match="num_templates"):
            metric.evaluate(num_templates=0, num_samples=1, subset="type1_pro")

    def test_invalid_num_templates_negative(self, mock_load_dataset):
        """Test negative num_templates raises ValueError."""
        metric = self._make_metric(mock_load_dataset)

        with pytest.raises(ValueError, match="num_templates"):
            metric.evaluate(num_templates=-1, num_samples=1, subset="type1_pro")

    @patch("bias_scope.prompts_based.occupation_pronoun_skew.completion")
    def test_pronoun_extraction_he(self, mock_completion, mock_load_dataset):
        """Test pronoun extraction correctly identifies 'he'."""
        metric = self._make_metric(mock_load_dataset)
        assert metric._extract_pronoun("He went to the store.") == "he"

    @patch("bias_scope.prompts_based.occupation_pronoun_skew.completion")
    def test_pronoun_extraction_she(self, mock_completion, mock_load_dataset):
        """Test pronoun extraction correctly identifies 'she'."""
        metric = self._make_metric(mock_load_dataset)
        assert metric._extract_pronoun("She walked in.") == "she"

    @patch("bias_scope.prompts_based.occupation_pronoun_skew.completion")
    def test_pronoun_extraction_they(self, mock_completion, mock_load_dataset):
        """Test pronoun extraction correctly identifies 'they'."""
        metric = self._make_metric(mock_load_dataset)
        assert metric._extract_pronoun("They arrived early.") == "they"

    @patch("bias_scope.prompts_based.occupation_pronoun_skew.completion")
    def test_pronoun_extraction_none(self, mock_completion, mock_load_dataset):
        """Test pronoun extraction returns None when no pronoun found."""
        metric = self._make_metric(mock_load_dataset)
        assert metric._extract_pronoun("The building is tall.") is None

    @patch("bias_scope.prompts_based.occupation_pronoun_skew.completion")
    def test_pronoun_extraction_empty(self, mock_completion, mock_load_dataset):
        """Test pronoun extraction with empty string."""
        metric = self._make_metric(mock_load_dataset)
        assert metric._extract_pronoun("") is None

    @patch("bias_scope.prompts_based.occupation_pronoun_skew.completion")
    def test_all_male_responses(self, mock_completion, mock_load_dataset):
        """Test representation ratio when all responses are male."""
        mock_completion.return_value = make_completion_response("He did the work.")
        metric = self._make_metric(mock_load_dataset)

        result = metric.evaluate(num_templates=1, num_samples=3, subset="type1_pro")

        assert result["representation_ratio"] == sys.float_info.max

    @patch("bias_scope.prompts_based.occupation_pronoun_skew.completion")
    def test_all_female_responses(self, mock_completion, mock_load_dataset):
        """Test representation ratio when all responses are female."""
        mock_completion.return_value = make_completion_response("She did the work.")
        metric = self._make_metric(mock_load_dataset)

        result = metric.evaluate(num_templates=1, num_samples=3, subset="type1_pro")

        assert result["representation_ratio"] == 0.0

    @patch("bias_scope.prompts_based.occupation_pronoun_skew.completion")
    def test_balanced_responses(self, mock_completion, mock_load_dataset):
        """Test representation ratio with balanced he/she responses."""
        responses = [
            make_completion_response("He did it."),
            make_completion_response("She did it."),
        ]
        # 6 calls for 2 templates * 3 samples; multiply to have enough responses.
        mock_completion.side_effect = responses * 3
        # Use 1 template, 2 samples to get 1 he + 1 she
        mock_completion.side_effect = [
            make_completion_response("He did it."),
            make_completion_response("She did it."),
        ]
        metric = self._make_metric(mock_load_dataset)

        result = metric.evaluate(num_templates=1, num_samples=2, subset="type1_pro")

        assert result["representation_ratio"] == pytest.approx(1.0)

    @patch("bias_scope.prompts_based.occupation_pronoun_skew.completion")
    def test_per_occupation_keys(self, mock_completion, mock_load_dataset):
        """Test per_occupation has correct occupation keys."""
        mock_completion.return_value = make_completion_response("She helped.")
        metric = self._make_metric(mock_load_dataset)

        result = metric.evaluate(num_templates=2, num_samples=1, subset="type1_pro")

        assert "nurse" in result["per_occupation"]
        assert "doctor" in result["per_occupation"]

    @patch("bias_scope.prompts_based.occupation_pronoun_skew.completion")
    def test_per_occupation_proportions_sum_to_one(self, mock_completion, mock_load_dataset):
        """Test per_occupation proportions sum to 1.0."""
        mock_completion.return_value = make_completion_response("She helped.")
        metric = self._make_metric(mock_load_dataset)

        result = metric.evaluate(num_templates=1, num_samples=3, subset="type1_pro")

        for occ, props in result["per_occupation"].items():
            total = sum(props.values())
            assert total == pytest.approx(1.0), f"Proportions for {occ} sum to {total}"

    @patch("bias_scope.prompts_based.occupation_pronoun_skew.completion")
    def test_l1_distance_uniform(self, mock_completion, mock_load_dataset):
        """Test L1 distance is 0 for perfectly uniform distribution."""
        metric = self._make_metric(mock_load_dataset)
        counts = {"he": 10, "she": 10, "they": 10}

        l1 = metric._compute_l1_distance(counts)

        assert l1 == pytest.approx(0.0, abs=1e-9)

    @patch("bias_scope.prompts_based.occupation_pronoun_skew.completion")
    def test_l1_distance_skewed(self, mock_completion, mock_load_dataset):
        """Test L1 distance is positive for skewed distribution."""
        metric = self._make_metric(mock_load_dataset)
        counts = {"he": 100, "she": 0, "they": 0}

        l1 = metric._compute_l1_distance(counts)

        assert l1 > 0

    def test_category_property(self, mock_load_dataset):
        """Test category returns 'prompt_based'."""
        metric = self._make_metric(mock_load_dataset)

        assert metric.category == "prompt_based"

    @patch("bias_scope.prompts_based.occupation_pronoun_skew.completion")
    def test_litellm_call_count(self, mock_completion, mock_load_dataset):
        """Test LiteLLM is called num_templates * num_samples times."""
        mock_completion.return_value = make_completion_response("She said hello.")
        metric = self._make_metric(mock_load_dataset)

        num_templates = 2
        num_samples = 3
        metric.evaluate(num_templates=num_templates, num_samples=num_samples, subset="type1_pro")

        assert mock_completion.call_count == num_templates * num_samples

    @patch("bias_scope.prompts_based.occupation_pronoun_skew.completion")
    def test_num_templates_larger_than_dataset(self, mock_completion, mock_load_dataset):
        """Test num_templates larger than dataset is handled gracefully."""
        mock_completion.return_value = make_completion_response("She helped.")
        metric = self._make_metric(mock_load_dataset, rows=SAMPLE_ROWS[:1])

        result = metric.evaluate(num_templates=100, num_samples=1, subset="type1_pro")

        assert "representation_ratio" in result
        # Only 1 template available, so only 1 call
        assert mock_completion.call_count == 1

    @patch("bias_scope.prompts_based.occupation_pronoun_skew.completion")
    def test_pronoun_extraction_first_match(self, mock_completion, mock_load_dataset):
        """Test pronoun extraction returns first pronoun in text."""
        metric = self._make_metric(mock_load_dataset)

        assert metric._extract_pronoun("He told her that she was right.") == "he"
        assert metric._extract_pronoun("After a while, she said he was wrong.") == "she"

    def test_num_samples_float_raises_error(self, mock_load_dataset):
        """Test non-integer num_samples raises ValueError."""
        metric = self._make_metric(mock_load_dataset)

        with pytest.raises(ValueError, match="num_samples"):
            metric.evaluate(num_samples=3.5, subset="type1_pro")

    def test_invalid_subset_raises_error(self, mock_load_dataset):
        """Test that an invalid subset string raises ValueError."""
        metric = self._make_metric(mock_load_dataset)

        with pytest.raises(ValueError, match="subset"):
            metric.evaluate(num_samples=1, subset="invalid_subset")

    @patch("bias_scope.prompts_based.occupation_pronoun_skew.completion")
    def test_valid_subsets(self, mock_completion, mock_load_dataset):
        """Test that all four valid subset strings are accepted without error."""
        mock_completion.return_value = make_completion_response("She walked in.")
        valid_subsets = ["type1_pro", "type1_anti", "type2_pro", "type2_anti"]
        for subset in valid_subsets:
            metric = self._make_metric(mock_load_dataset)
            # Should not raise; use minimal call to keep tests fast
            result = metric.evaluate(num_templates=1, num_samples=1, subset=subset)
            assert "representation_ratio" in result, f"Missing key for subset={subset!r}"


@patch("bias_scope.prompts_based.occupation_pronoun_skew.load_dataset")
class TestRunReadsTheRepresentationRatio:
    """OccupationPronounSkew returned `representation_ratio` and `l1_distance`
    and no headline or count, so run() could not read it and the agent's
    `prompt_benchmarks` provider skipped it (API-target run, 2026-09-21). Its
    MetricInfo - neutral 1.0, signed, range (0, inf) - describes the
    representation ratio (he / she); n is the completions actually generated
    (templates x samples)."""

    @patch("bias_scope.prompts_based.occupation_pronoun_skew.completion")
    def test_run_headline_is_the_ratio_and_n_counts_completions(self, mock_completion,
                                                                mock_load_dataset):
        mock_load_dataset.return_value = make_mock_dataset(SAMPLE_ROWS)
        from bias_scope.prompts_based.occupation_pronoun_skew import OccupationPronounSkew

        # 2 templates x 3 samples: four "he", two "she" -> ratio 2.0
        replies = ["he did", "he did", "she did", "he did", "he did", "she did"]
        mock_completion.side_effect = [make_completion_response(r) for r in replies]
        metric = OccupationPronounSkew(model_name="openai/gpt-4o", api_key="test-key")
        result = metric.run(num_templates=2, num_samples=3, ci="none")
        assert result.score == pytest.approx(2.0)
        assert result.n == 6

    def test_evaluate_reports_the_completion_count(self, mock_load_dataset):
        mock_load_dataset.return_value = make_mock_dataset(SAMPLE_ROWS)
        from bias_scope.prompts_based.occupation_pronoun_skew import OccupationPronounSkew

        with patch("bias_scope.prompts_based.occupation_pronoun_skew.completion",
                   side_effect=[make_completion_response("she") for _ in range(4)]):
            details = OccupationPronounSkew(model_name="m", api_key="k").evaluate(
                num_templates=2, num_samples=2, return_details=True)
        assert details["num_completions"] == 4
