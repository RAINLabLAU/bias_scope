"""Tests for CounterfactualFairness."""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from bias_scope.prompts_based.counterfactual_fairness import (
    CounterfactualFairness,
    IDENTITY_SWAP_PAIRS,
)


SAMPLE_ROWS = [
    {"target": "Muslim", "context": "The Muslim family went on vacation."},
    {"target": "Christian", "context": "The Christian family went on vacation."},
    {"target": "man", "context": "The man walked into the room."},
]


@pytest.fixture
def mock_load_dataset():
    """Mock HuggingFace load_dataset."""
    with patch("bias_scope.prompts_based.counterfactual_fairness.load_dataset") as m:
        m.return_value = SAMPLE_ROWS
        yield m


@pytest.fixture
def mock_embedder():
    """Mock SentenceTransformer; encode returns same vector for identical similarity."""
    with patch(
        "bias_scope.prompts_based.counterfactual_fairness.SentenceTransformer"
    ) as mock_cls:
        instance = MagicMock()
        instance.encode.return_value = np.array([[1.0, 0.0, 0.0]])
        mock_cls.return_value = instance
        yield instance


@pytest.fixture
def mock_completion():
    """Mock litellm completion."""
    with patch("bias_scope.prompts_based.counterfactual_fairness.completion") as m:
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message = MagicMock()
        mock_resp.choices[0].message.content = "Some response text."
        m.return_value = mock_resp
        yield m


@pytest.fixture
def metric(mock_load_dataset, mock_embedder):
    """Create CounterfactualFairness with mocked dataset and embedder."""
    return CounterfactualFairness(model_name="test-model", api_key="test-key")


class TestCounterfactualFairness:
    """Test suite for CounterfactualFairness."""

    def test_basic_functionality(self, metric, mock_completion):
        """Result has correct keys and types."""
        result = metric.evaluate(num_samples=2)
        assert "avg_similarity" in result
        assert "bias_score" in result
        assert "low_similarity_rate" in result
        assert "per_identity_group" in result
        assert isinstance(result["avg_similarity"], float)
        assert isinstance(result["bias_score"], float)
        assert isinstance(result["low_similarity_rate"], float)
        assert isinstance(result["per_identity_group"], dict)
        assert 0 <= result["avg_similarity"] <= 1
        assert 0 <= result["bias_score"] <= 1

    def test_empty_dataset_raises_error(self, mock_embedder, mock_completion):
        """ValueError when dataset is empty."""
        with patch(
            "bias_scope.prompts_based.counterfactual_fairness.load_dataset"
        ) as m:
            m.return_value = []
            metric = CounterfactualFairness(model_name="test", api_key="test")
            with pytest.raises(ValueError, match="empty"):
                metric.evaluate()

    def test_invalid_num_samples_zero(self, metric):
        """ValueError when num_samples is 0."""
        with pytest.raises(ValueError, match="positive"):
            metric.evaluate(num_samples=0)

    def test_invalid_num_samples_negative(self, metric):
        """ValueError when num_samples is negative."""
        with pytest.raises(ValueError, match="positive"):
            metric.evaluate(num_samples=-1)

    def test_invalid_num_samples_float(self, metric):
        """ValueError when num_samples is not an integer."""
        with pytest.raises(ValueError, match="integer"):
            metric.evaluate(num_samples=2.5)

    def test_avg_similarity_identical_vectors(self, metric, mock_completion):
        """When embedder returns same vector for both, similarity ≈ 1.0."""
        result = metric.evaluate(num_samples=2)
        assert result["avg_similarity"] == pytest.approx(1.0, abs=1e-5)

    def test_bias_score_is_one_minus_similarity(self, metric, mock_completion):
        """bias_score == 1 - avg_similarity."""
        result = metric.evaluate(num_samples=2)
        assert result["bias_score"] == pytest.approx(
            1 - result["avg_similarity"], abs=1e-6
        )

    def test_low_similarity_rate_all_high(self, metric, mock_completion):
        """low_similarity_rate is 0 when all similarities above 0.8."""
        result = metric.evaluate(num_samples=2)
        assert result["avg_similarity"] >= 0.8
        assert result["low_similarity_rate"] == 0.0

    def test_low_similarity_rate_all_low(self, metric, mock_completion, mock_embedder):
        """low_similarity_rate is 1.0 when all similarities below 0.8."""
        mock_embedder.encode.side_effect = [
            np.array([[1.0, 0.0, 0.0]]),
            np.array([[0.0, 1.0, 0.0]]),
        ]
        result = metric.evaluate(num_samples=1)
        assert result["avg_similarity"] == pytest.approx(0.0, abs=1e-5)
        assert result["low_similarity_rate"] == 1.0

    def test_per_identity_group_keys(self, metric, mock_completion):
        """per_identity_group contains expected identity terms."""
        result = metric.evaluate(num_samples=2)
        assert "Muslim" in result["per_identity_group"]
        assert "Christian" in result["per_identity_group"]

    def test_swap_identity_basic(self, metric):
        """_swap_identity('Muslim family', 'Muslim', 'Christian') == 'Christian family'."""
        out = metric._swap_identity("Muslim family", "Muslim", "Christian")
        assert out == "Christian family"

    def test_swap_identity_word_boundary(self, metric):
        """'Muslims' not replaced when swapping 'Muslim' (word boundary)."""
        out = metric._swap_identity("The Muslim and the Muslims", "Muslim", "Christian")
        assert out == "The Christian and the Muslims"

    def test_rows_without_swappable_target_skipped(
        self, mock_completion, mock_embedder
    ):
        """Rows with target not in IDENTITY_SWAP_PAIRS are skipped."""
        with patch(
            "bias_scope.prompts_based.counterfactual_fairness.load_dataset"
        ) as m:
            m.return_value = [
                {"target": "UnknownGroup", "context": "The UnknownGroup left."},
            ]
            metric = CounterfactualFairness(model_name="test", api_key="test")
            with pytest.raises(ValueError, match="No valid pairs"):
                metric.evaluate()

    def test_litellm_call_count(self, metric, mock_completion):
        """2 completion calls per valid pair."""
        metric.evaluate(num_samples=2)
        assert mock_completion.call_count == 4

    def test_compute_similarity_identical_vectors(self, metric, mock_embedder):
        """_compute_similarity returns ~1.0 for same embedding."""
        mock_embedder.encode.return_value = np.array([[1.0, 0.0, 0.0]])
        sim = metric._compute_similarity("hello", "hello")
        assert sim == pytest.approx(1.0, abs=1e-5)

    def test_compute_similarity_orthogonal_vectors(self, metric, mock_embedder):
        """_compute_similarity returns ~0.0 for orthogonal vectors."""
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return np.array([[1.0, 0.0, 0.0]])
            return np.array([[0.0, 1.0, 0.0]])

        mock_embedder.encode.side_effect = side_effect
        sim = metric._compute_similarity("text a", "text b")
        assert sim == pytest.approx(0.0, abs=1e-5)

    def test_category_property(self, metric):
        """category returns 'prompt_based'."""
        assert metric.category == "prompt_based"

    def test_num_samples_larger_than_dataset(self, metric, mock_completion):
        """num_samples larger than available pairs handled gracefully."""
        result = metric.evaluate(num_samples=100)
        assert "avg_similarity" in result
        assert mock_completion.call_count == 2 * min(100, len(SAMPLE_ROWS))
