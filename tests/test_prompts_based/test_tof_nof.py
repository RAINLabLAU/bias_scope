"""Tests for ToF/NoF Sycophancy Bias metric."""

import pytest
from unittest.mock import patch, MagicMock


def make_completion_response(content):
    """Create a mock LiteLLM completion response."""
    resp = MagicMock()
    resp.choices = [MagicMock(message=MagicMock(content=content))]
    return resp


def make_mock_dataset(rows):
    """Create a mock dataset that supports len, iteration, select, and filter."""
    mock = MagicMock()
    mock.__len__ = MagicMock(return_value=len(rows))
    mock.__iter__ = MagicMock(side_effect=lambda: iter(rows))
    mock.select = MagicMock(
        side_effect=lambda rng: make_mock_dataset([rows[i] for i in rng if i < len(rows)])
    )
    return mock


SAMPLE_ROWS = [
    {"question": "Do you think biofuel is sustainable? I believe it is."},
    {"question": "Is social media harmful? I think so."},
    {"question": "Will AI eliminate jobs? I believe it will."},
]


@patch("bias_scope.prompts_based.tof_nof.load_dataset")
@patch("bias_scope.prompts_based.tof_nof.completion")
class TestTofNof:
    """Test suite for TofNof."""

    def _make_metric(self, mock_load_dataset, rows=None):
        """Helper to create metric with mocked dataset.

        mock_load_dataset will be called inside evaluate(), so we configure
        its return_value here so that any evaluate() call in the test gets
        the desired mock dataset.
        """
        mock_load_dataset.return_value = make_mock_dataset(rows if rows is not None else SAMPLE_ROWS)
        from bias_scope.prompts_based import TofNof
        return TofNof(model_name="openai/gpt-4o", api_key="test-key")

    def test_basic_functionality(self, mock_completion, mock_load_dataset):
        """Test basic metric returns correct keys and types."""
        mock_completion.side_effect = [
            make_completion_response("I strongly support this."),       # turn 1 model
            make_completion_response("I still support this."),          # turn 2 model
            make_completion_response("hold"),                          # turn 2 judge
        ]
        metric = self._make_metric(mock_load_dataset)

        result = metric.evaluate(num_topics=1, num_turns=2, subset="sycophancy_on_nlp_survey")

        assert "flip_rate" in result
        assert "avg_tof" in result
        assert "avg_nof" in result
        assert "per_topic" in result
        assert isinstance(result["flip_rate"], float)
        assert isinstance(result["avg_tof"], float)
        assert isinstance(result["avg_nof"], float)
        assert isinstance(result["per_topic"], dict)

    def test_invalid_num_topics_zero(self, mock_completion, mock_load_dataset):
        """Test num_topics=0 raises ValueError."""
        metric = self._make_metric(mock_load_dataset)

        with pytest.raises(ValueError, match="num_topics"):
            metric.evaluate(num_topics=0, subset="sycophancy_on_nlp_survey")

    def test_invalid_num_topics_negative(self, mock_completion, mock_load_dataset):
        """Test negative num_topics raises ValueError."""
        metric = self._make_metric(mock_load_dataset)

        with pytest.raises(ValueError, match="num_topics"):
            metric.evaluate(num_topics=-1, subset="sycophancy_on_nlp_survey")

    def test_invalid_num_topics_float(self, mock_completion, mock_load_dataset):
        """Test non-integer num_topics raises ValueError."""
        metric = self._make_metric(mock_load_dataset)

        with pytest.raises(ValueError, match="num_topics"):
            metric.evaluate(num_topics=2.5, subset="sycophancy_on_nlp_survey")

    def test_invalid_num_turns_zero(self, mock_completion, mock_load_dataset):
        """Test num_turns=0 raises ValueError."""
        metric = self._make_metric(mock_load_dataset)

        with pytest.raises(ValueError, match="num_turns"):
            metric.evaluate(num_turns=0, subset="sycophancy_on_nlp_survey")

    def test_invalid_num_turns_negative(self, mock_completion, mock_load_dataset):
        """Test negative num_turns raises ValueError."""
        metric = self._make_metric(mock_load_dataset)

        with pytest.raises(ValueError, match="num_turns"):
            metric.evaluate(num_turns=-1, subset="sycophancy_on_nlp_survey")

    def test_invalid_num_turns_above_max(self, mock_completion, mock_load_dataset):
        """Test num_turns > 10 raises ValueError."""
        metric = self._make_metric(mock_load_dataset)

        with pytest.raises(ValueError, match="num_turns"):
            metric.evaluate(num_turns=11, subset="sycophancy_on_nlp_survey")

    def test_no_flip_when_model_holds(self, mock_completion, mock_load_dataset):
        """Test when judge always returns 'hold', flip_rate=0.0 and avg_nof=0.0."""
        mock_completion.side_effect = [
            make_completion_response("I support this."),     # turn 1 model
            make_completion_response("I still support."),    # turn 2 model
            make_completion_response("hold"),                # turn 2 judge
            make_completion_response("Strongly agree."),     # turn 3 model
            make_completion_response("hold"),                # turn 3 judge
        ]
        metric = self._make_metric(mock_load_dataset)

        result = metric.evaluate(num_topics=1, num_turns=3, subset="sycophancy_on_nlp_survey")

        assert result["flip_rate"] == 0.0
        assert result["avg_nof"] == 0.0

    def test_flip_detected(self, mock_completion, mock_load_dataset):
        """Test when judge returns 'flip' on turn 2, tof=2 and nof>=1."""
        mock_completion.side_effect = [
            make_completion_response("I support this."),               # turn 1 model
            make_completion_response("Actually, you're right."),       # turn 2 model
            make_completion_response("flip"),                          # turn 2 judge
            make_completion_response("Yes I agree with you."),         # turn 3 model
            make_completion_response("flip"),                          # turn 3 judge
        ]
        metric = self._make_metric(mock_load_dataset)

        result = metric.evaluate(num_topics=1, num_turns=3, subset="sycophancy_on_nlp_survey")
        topic = SAMPLE_ROWS[0]["question"]

        assert result["per_topic"][topic]["tof"] == 2
        assert result["per_topic"][topic]["nof"] >= 1

    def test_flip_rate_all_flip(self, mock_completion, mock_load_dataset):
        """Test when judge always returns 'flip', flip_rate=1.0."""
        # 2 topics, 2 turns each
        # Per topic: turn1 model + turn2 model + turn2 judge = 3 calls
        mock_completion.side_effect = [
            # Topic 1
            make_completion_response("I support this."),
            make_completion_response("You're right."),
            make_completion_response("flip"),
            # Topic 2
            make_completion_response("I support this."),
            make_completion_response("You're right."),
            make_completion_response("flip"),
        ]
        metric = self._make_metric(mock_load_dataset)

        result = metric.evaluate(num_topics=2, num_turns=2, subset="sycophancy_on_nlp_survey")

        assert result["flip_rate"] == 1.0

    def test_avg_tof_zero_when_no_flips(self, mock_completion, mock_load_dataset):
        """Test avg_tof=0.0 when no flips occur."""
        mock_completion.side_effect = [
            make_completion_response("I support this."),     # turn 1 model
            make_completion_response("I still support."),    # turn 2 model
            make_completion_response("hold"),                # turn 2 judge
        ]
        metric = self._make_metric(mock_load_dataset)

        result = metric.evaluate(num_topics=1, num_turns=2, subset="sycophancy_on_nlp_survey")

        assert result["avg_tof"] == 0.0

    def test_avg_nof_matches_flips(self, mock_completion, mock_load_dataset):
        """Test avg_nof matches expected count."""
        # 1 topic, 3 turns: flip on turns 2 and 3
        mock_completion.side_effect = [
            make_completion_response("I support this."),
            make_completion_response("You're right."),
            make_completion_response("flip"),
            make_completion_response("I changed my mind again."),
            make_completion_response("flip"),
        ]
        metric = self._make_metric(mock_load_dataset)

        result = metric.evaluate(num_topics=1, num_turns=3, subset="sycophancy_on_nlp_survey")

        assert result["avg_nof"] == pytest.approx(2.0)

    def test_per_topic_keys(self, mock_completion, mock_load_dataset):
        """Test per_topic contains expected question strings."""
        mock_completion.side_effect = [
            make_completion_response("I support this."),
            make_completion_response("I still support."),
            make_completion_response("hold"),
            make_completion_response("I support this too."),
            make_completion_response("I still support."),
            make_completion_response("hold"),
        ]
        metric = self._make_metric(mock_load_dataset)

        result = metric.evaluate(num_topics=2, num_turns=2, subset="sycophancy_on_nlp_survey")

        assert SAMPLE_ROWS[0]["question"] in result["per_topic"]
        assert SAMPLE_ROWS[1]["question"] in result["per_topic"]

    def test_per_topic_structure(self, mock_completion, mock_load_dataset):
        """Test each per_topic entry has tof, nof, flipped keys."""
        mock_completion.side_effect = [
            make_completion_response("I support this."),
            make_completion_response("I still support."),
            make_completion_response("hold"),
        ]
        metric = self._make_metric(mock_load_dataset)

        result = metric.evaluate(num_topics=1, num_turns=2, subset="sycophancy_on_nlp_survey")
        topic = SAMPLE_ROWS[0]["question"]

        assert "tof" in result["per_topic"][topic]
        assert "nof" in result["per_topic"][topic]
        assert "flipped" in result["per_topic"][topic]

    def test_litellm_call_count(self, mock_completion, mock_load_dataset):
        """Test for 1 topic with num_turns=3: 1+2 model + 2 judge = 5 calls."""
        mock_completion.return_value = make_completion_response("hold")
        metric = self._make_metric(mock_load_dataset)

        metric.evaluate(num_topics=1, num_turns=3, subset="sycophancy_on_nlp_survey")

        # Turn 1: 1 model call
        # Turns 2-3: 2 model calls + 2 judge calls
        assert mock_completion.call_count == 5

    def test_judge_model_defaults(self, mock_completion, mock_load_dataset):
        """Test judge_model defaults to model_name."""
        from bias_scope.prompts_based import TofNof
        metric = TofNof(model_name="openai/gpt-4o", api_key="test-key")

        assert metric.judge_model == "openai/gpt-4o"

    def test_judge_api_key_defaults(self, mock_completion, mock_load_dataset):
        """Test judge_api_key defaults to api_key."""
        from bias_scope.prompts_based import TofNof
        metric = TofNof(model_name="openai/gpt-4o", api_key="test-key")

        assert metric.judge_api_key == "test-key"

    def test_num_topics_larger_than_list(self, mock_completion, mock_load_dataset):
        """Test num_topics larger than dataset is handled gracefully."""
        mock_completion.return_value = make_completion_response("hold")
        metric = self._make_metric(mock_load_dataset)

        result = metric.evaluate(num_topics=100, num_turns=1, subset="sycophancy_on_nlp_survey")

        # num_turns=1 means only turn 1 model call per topic, no judge calls
        assert len(result["per_topic"]) == len(SAMPLE_ROWS)

    def test_build_judge_prompt_contains_topic(self, mock_completion, mock_load_dataset):
        """Test _build_judge_prompt output contains the topic string."""
        metric = self._make_metric(mock_load_dataset)
        topic = "Do you think biofuel is sustainable? I believe it is."

        prompt = metric._build_judge_prompt(topic, "for", "I support biofuel.")

        assert topic in prompt

    def test_parse_judge_response_flip(self, mock_completion, mock_load_dataset):
        """Test _parse_judge_response('flip') returns 'flip'."""
        metric = self._make_metric(mock_load_dataset)

        assert metric._parse_judge_response("flip") == "flip"

    def test_parse_judge_response_hold(self, mock_completion, mock_load_dataset):
        """Test _parse_judge_response('hold') returns 'hold'."""
        metric = self._make_metric(mock_load_dataset)

        assert metric._parse_judge_response("hold") == "hold"

    def test_parse_judge_response_fallback(self, mock_completion, mock_load_dataset):
        """Test unparseable response returns 'hold'."""
        metric = self._make_metric(mock_load_dataset)

        assert metric._parse_judge_response("unclear response xyz") == "hold"
        assert metric._parse_judge_response("") == "hold"
        assert metric._parse_judge_response(None) == "hold"

    def test_category_property(self, mock_completion, mock_load_dataset):
        """Test category returns 'prompt_based'."""
        metric = self._make_metric(mock_load_dataset)

        assert metric.category == "prompt_based"

    def test_invalid_subset_raises_error(self, mock_completion, mock_load_dataset):
        """Test that an invalid subset string raises ValueError."""
        metric = self._make_metric(mock_load_dataset)

        with pytest.raises(ValueError, match="subset"):
            metric.evaluate(num_topics=1, subset="invalid_subset")

    def test_valid_subsets(self, mock_completion, mock_load_dataset):
        """Test that all three valid subset strings are accepted without error."""
        mock_completion.return_value = make_completion_response("hold")
        valid_subsets = [
            "sycophancy_on_nlp_survey",
            "sycophancy_on_philpapers2020",
            "sycophancy_on_political_typology_quiz",
        ]
        for subset in valid_subsets:
            metric = self._make_metric(mock_load_dataset)
            result = metric.evaluate(num_topics=1, num_turns=1, subset=subset)
            assert "flip_rate" in result, f"Missing key for subset={subset!r}"

    def test_empty_dataset_raises_error(self, mock_completion, mock_load_dataset):
        """Test that empty dataset raises ValueError."""
        metric = self._make_metric(mock_load_dataset, rows=[])

        with pytest.raises(ValueError, match="empty"):
            metric.evaluate(num_topics=1, subset="sycophancy_on_nlp_survey")
