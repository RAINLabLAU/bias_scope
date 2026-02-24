"""Tests for Contextualized Embedding Association Test (CEAT)."""

import numpy as np
import pytest
import torch

from bias_scope.embeddings_based import CEAT


class TestCEAT:
    """Test Contextualized Embedding Association Test."""

    def test_basic_functionality(self):
        """Test CEAT with sufficient data and default parameters."""
        ceat = CEAT()

        # 50 embeddings per group, allowing for good sampling
        target1 = np.random.randn(50, 768)
        target2 = np.random.randn(50, 768)
        attr1 = np.random.randn(40, 768)
        attr2 = np.random.randn(40, 768)

        result = ceat.evaluate(
            (target1, target2), (attr1, attr2), n_samples=50, random_seed=42
        )

        # Verify return type and keys
        assert isinstance(result, dict)
        assert "ceat_score" in result
        assert "weat_mean" in result
        assert "weat_std" in result
        assert "weat_variance" in result
        assert "n_samples" in result

        # Verify all values are floats
        assert isinstance(result["ceat_score"], float)
        assert isinstance(result["weat_mean"], float)
        assert isinstance(result["weat_std"], float)
        assert isinstance(result["weat_variance"], float)

        # Verify n_samples matches input
        assert result["n_samples"] == 50

    def test_with_minimal_data(self):
        """Test CEAT with minimum viable data."""
        ceat = CEAT()

        # Exactly sample_size per group (10 each)
        target1 = np.random.randn(10, 300)
        target2 = np.random.randn(10, 300)
        attr1 = np.random.randn(10, 300)
        attr2 = np.random.randn(10, 300)

        result = ceat.evaluate(
            (target1, target2),
            (attr1, attr2),
            n_samples=20,
            sample_size=10,
            random_seed=42,
        )

        assert isinstance(result["ceat_score"], float)
        assert not np.isnan(result["ceat_score"])

    def test_with_large_data(self):
        """Test CEAT with large dataset."""
        ceat = CEAT()

        # 100+ embeddings per group
        target1 = np.random.randn(100, 768)
        target2 = np.random.randn(100, 768)
        attr1 = np.random.randn(80, 768)
        attr2 = np.random.randn(80, 768)

        result = ceat.evaluate(
            (target1, target2),
            (attr1, attr2),
            n_samples=100,
            sample_size=20,
            random_seed=42,
        )

        assert isinstance(result["ceat_score"], float)

    def test_reproducibility_with_seed(self):
        """Test same seed produces identical results."""
        ceat = CEAT()

        target1 = np.random.randn(50, 300)
        target2 = np.random.randn(50, 300)
        attr1 = np.random.randn(40, 300)
        attr2 = np.random.randn(40, 300)

        result1 = ceat.evaluate(
            (target1, target2), (attr1, attr2), n_samples=50, random_seed=42
        )

        result2 = ceat.evaluate(
            (target1, target2), (attr1, attr2), n_samples=50, random_seed=42
        )

        # All values should be exactly equal
        assert result1["ceat_score"] == result2["ceat_score"]
        assert result1["weat_mean"] == result2["weat_mean"]
        assert result1["weat_std"] == result2["weat_std"]
        assert result1["weat_variance"] == result2["weat_variance"]

    def test_different_seeds_different_results(self):
        """Test different seeds produce different results."""
        ceat = CEAT()

        target1 = np.random.randn(50, 300)
        target2 = np.random.randn(50, 300)
        attr1 = np.random.randn(40, 300)
        attr2 = np.random.randn(40, 300)

        result1 = ceat.evaluate(
            (target1, target2), (attr1, attr2), n_samples=50, random_seed=42
        )

        result2 = ceat.evaluate(
            (target1, target2), (attr1, attr2), n_samples=50, random_seed=123
        )

        # Results should differ (random sampling)
        assert result1["ceat_score"] != result2["ceat_score"]

    def test_ceat_score_within_reasonable_range(self):
        """Test CEAT score is within expected range."""
        ceat = CEAT()

        # Random embeddings should produce moderate scores
        target1 = np.random.randn(50, 300)
        target2 = np.random.randn(50, 300)
        attr1 = np.random.randn(40, 300)
        attr2 = np.random.randn(40, 300)

        result = ceat.evaluate(
            (target1, target2), (attr1, attr2), n_samples=50, random_seed=42
        )

        # For random data, effect sizes typically in [-3, 3]
        assert -3.0 < result["ceat_score"] < 3.0

    def test_weat_variance_non_negative(self):
        """Test variance is always non-negative."""
        ceat = CEAT()

        target1 = np.random.randn(50, 300)
        target2 = np.random.randn(50, 300)
        attr1 = np.random.randn(40, 300)
        attr2 = np.random.randn(40, 300)

        result = ceat.evaluate(
            (target1, target2), (attr1, attr2), n_samples=50, random_seed=42
        )

        # Statistical property: variance >= 0
        assert result["weat_variance"] >= 0

    def test_sample_size_parameter(self):
        """Test custom sample_size is respected."""
        ceat = CEAT()

        target1 = np.random.randn(50, 300)
        target2 = np.random.randn(50, 300)
        attr1 = np.random.randn(40, 300)
        attr2 = np.random.randn(40, 300)

        # Small sample size should work
        result = ceat.evaluate(
            (target1, target2),
            (attr1, attr2),
            n_samples=30,
            sample_size=5,
            random_seed=42,
        )

        assert isinstance(result["ceat_score"], float)

    def test_automatic_sample_size_selection(self):
        """Test sample_size=None uses min(10, smallest_group_size)."""
        ceat = CEAT()

        # Smallest group has 7 embeddings
        target1 = np.random.randn(7, 300)
        target2 = np.random.randn(8, 300)
        attr1 = np.random.randn(15, 300)
        attr2 = np.random.randn(20, 300)

        # Should auto-select sample_size = min(10, 7) = 7
        result = ceat.evaluate(
            (target1, target2), (attr1, attr2), n_samples=20, random_seed=42
        )

        assert isinstance(result["ceat_score"], float)

    def test_insufficient_data_raises_error(self):
        """Test error when not enough embeddings for sampling."""
        ceat = CEAT()

        # Only 5 embeddings but sample_size=10
        target1 = np.random.randn(5, 300)
        target2 = np.random.randn(5, 300)
        attr1 = np.random.randn(5, 300)
        attr2 = np.random.randn(5, 300)

        with pytest.raises(ValueError, match="sample_size"):
            ceat.evaluate((target1, target2), (attr1, attr2), sample_size=10)

    def test_validates_dimensions(self):
        """Test dimension mismatch raises error."""
        ceat = CEAT()

        target1 = np.random.randn(50, 100)
        target2 = np.random.randn(50, 200)  # Different dimension!
        attr1 = np.random.randn(40, 100)
        attr2 = np.random.randn(40, 100)

        with pytest.raises(ValueError, match="dimension"):
            ceat.evaluate((target1, target2), (attr1, attr2))

    def test_validates_n_samples_positive(self):
        """Test n_samples must be positive."""
        ceat = CEAT()

        target1 = np.random.randn(50, 300)
        target2 = np.random.randn(50, 300)
        attr1 = np.random.randn(40, 300)
        attr2 = np.random.randn(40, 300)

        # Zero samples
        with pytest.raises(ValueError, match="positive"):
            ceat.evaluate((target1, target2), (attr1, attr2), n_samples=0)

        # Negative samples
        with pytest.raises(ValueError, match="positive"):
            ceat.evaluate((target1, target2), (attr1, attr2), n_samples=-1)

    def test_nan_in_embeddings(self):
        """Test NaN detection."""
        ceat = CEAT()

        target1 = np.random.randn(50, 300)
        target1[0, 0] = np.nan
        target2 = np.random.randn(50, 300)
        attr1 = np.random.randn(40, 300)
        attr2 = np.random.randn(40, 300)

        with pytest.raises(ValueError, match="NaN"):
            ceat.evaluate((target1, target2), (attr1, attr2))

    def test_inf_in_embeddings(self):
        """Test Inf detection."""
        ceat = CEAT()

        target1 = np.random.randn(50, 300)
        target2 = np.random.randn(50, 300)
        target2[0, 0] = np.inf
        attr1 = np.random.randn(40, 300)
        attr2 = np.random.randn(40, 300)

        with pytest.raises(ValueError, match="Inf"):
            ceat.evaluate((target1, target2), (attr1, attr2))

    def test_high_dimensional_embeddings(self):
        """Test with realistic BERT dimensions (768-dim)."""
        ceat = CEAT()

        target1 = np.random.randn(50, 768)
        target2 = np.random.randn(50, 768)
        attr1 = np.random.randn(40, 768)
        attr2 = np.random.randn(40, 768)

        result = ceat.evaluate(
            (target1, target2), (attr1, attr2), n_samples=50, random_seed=42
        )

        assert isinstance(result["ceat_score"], float)
        assert not np.isnan(result["ceat_score"])

    def test_n_samples_equals_one(self):
        """Test edge case where n_samples=1."""
        ceat = CEAT()

        target1 = np.random.randn(50, 300)
        target2 = np.random.randn(50, 300)
        attr1 = np.random.randn(40, 300)
        attr2 = np.random.randn(40, 300)

        result = ceat.evaluate(
            (target1, target2), (attr1, attr2), n_samples=1, random_seed=42
        )

        # With n=1, CEAT score should equal the single WEAT score
        # Variance should be undefined but handled gracefully
        assert isinstance(result["ceat_score"], float)

    def test_with_torch_tensors(self):
        """Test CEAT handles PyTorch tensors."""
        ceat = CEAT()

        target1 = torch.randn(50, 300)
        target2 = torch.randn(50, 300)
        attr1 = torch.randn(40, 300)
        attr2 = torch.randn(40, 300)

        result = ceat.evaluate(
            (target1, target2), (attr1, attr2), n_samples=30, random_seed=42
        )

        assert isinstance(result["ceat_score"], float)

    def test_validates_tuple_length(self):
        """Test tuple validation."""
        ceat = CEAT()

        target = np.random.randn(50, 300)
        attr = np.random.randn(40, 300)

        # Wrong number of target groups
        with pytest.raises(ValueError, match="exactly 2 elements"):
            ceat.evaluate((target,), (attr, attr))

        # Wrong number of attribute groups
        with pytest.raises(ValueError, match="exactly 2 elements"):
            ceat.evaluate((target, target), (attr,))
