"""Tests for WEAT (Word Embedding Association Test)."""

import pytest
import numpy as np
import torch
from bias_scope.embeddings import WEAT


class TestWEAT:
    """Test Word Embedding Association Test."""
    
    def test_basic_functionality(self):
        """Test WEAT with simple inputs."""
        target1 = np.array([[1.0, 0.0], [0.9, 0.1]])
        target2 = np.array([[0.0, 1.0], [0.1, 0.9]])
        attr1 = np.array([[1.0, 0.0], [0.95, 0.05]])
        attr2 = np.array([[0.0, 1.0], [0.05, 0.95]])
        
        # Test OO API
        weat_instance = WEAT()
        score = weat_instance.compute((target1, target2), (attr1, attr2))
        
        assert isinstance(score, float)
        assert not np.isnan(score)
        assert score > 0
    
    def test_metadata_properties(self):
        """Test metric metadata is accessible."""
        weat_instance = WEAT()
        
        assert weat_instance.name == "WEAT"
        assert weat_instance.category == "embedding"
        assert weat_instance.complexity == "medium"
        assert "Caliskan" in weat_instance.reference
    
    def test_with_torch_tensors(self):
        """Test WEAT handles PyTorch tensors."""
        weat = WEAT()
        
        target1 = torch.randn(3, 10)
        target2 = torch.randn(3, 10)
        attr1 = torch.randn(2, 10)
        attr2 = torch.randn(2, 10)
        
        score = weat.compute((target1, target2), (attr1, attr2))
        assert isinstance(score, float)
    
    def test_identical_targets(self):
        """Test WEAT with identical target sets returns near zero."""
        weat = WEAT()
        
        target = np.random.randn(5, 20)
        attr1 = np.random.randn(3, 20)
        attr2 = np.random.randn(3, 20)
        
        score = weat.compute((target, target), (attr1, attr2))
        assert abs(score) < 0.01
    
    def test_raises_on_wrong_input(self):
        """Test WEAT validates input format."""
        weat = WEAT()
        
        target = np.random.randn(3, 10)
        attr = np.random.randn(2, 10)
        
        with pytest.raises(ValueError, match="must have exactly 2 elements"):
            weat.compute((target,), (attr, attr))
        
        with pytest.raises(ValueError, match="must have exactly 2 elements"):
            weat.compute((target, target), (attr,))

    def test_weat_single_embedding_per_group(self):
        """Test WEAT with single embedding in each group."""
        weat = WEAT()
        
        target1 = np.random.randn(1, 10)
        target2 = np.random.randn(1, 10)
        attr1 = np.random.randn(1, 10)
        attr2 = np.random.randn(1, 10)
        
        # Should return a valid score (float) as we have 2 embeddings total (1+1)
        score = weat.compute((target1, target2), (attr1, attr2))
        assert isinstance(score, float)

    def test_weat_empty_attribute_embeddings(self):
        """Test WEAT raises error with empty attribute arrays."""
        weat = WEAT()
        
        target1 = np.random.randn(5, 10)
        target2 = np.random.randn(5, 10)
        attr1 = np.array([]).reshape(0, 10)
        attr2 = np.random.randn(3, 10)
        
        with pytest.raises(ValueError, match="cannot be empty"):
            weat.compute((target1, target2), (attr1, attr2))

    def test_weat_zero_std(self):
        """Test WEAT raises error when standard deviation is zero."""
        weat = WEAT()
        
        # Create scenario where all target words have identical similarity to attributes
        target1 = np.ones((5, 10))
        target2 = np.ones((5, 10))
        attr1 = np.ones((1, 10))
        attr2 = -np.ones((1, 10))
        
        with pytest.raises(ValueError, match="Standard deviation .* is zero"):
            weat.compute((target1, target2), (attr1, attr2))

    def test_weat_empty_embeddings(self):
        """Test WEAT raises error with empty arrays."""
        weat = WEAT()
        
        target1 = np.array([]).reshape(0, 10)
        target2 = np.random.randn(5, 10)
        attr1 = np.random.randn(3, 10)
        attr2 = np.random.randn(3, 10)
        
        with pytest.raises(ValueError, match="cannot be empty"):
            weat.compute((target1, target2), (attr1, attr2))

    def test_weat_high_dimensional_embeddings(self):
        """Test WEAT with realistic embedding dimensions."""
        weat = WEAT()
        
        dims = [300, 768, 4096]
        for dim in dims:
            target1 = np.random.randn(5, dim)
            target2 = np.random.randn(5, dim)
            attr1 = np.random.randn(5, dim)
            attr2 = np.random.randn(5, dim)
            
            score = weat.compute((target1, target2), (attr1, attr2))
            assert isinstance(score, float)

    def test_weat_nan_in_embeddings(self):
        """Test WEAT handles NaN values."""
        weat = WEAT()
        
        target1 = np.random.randn(5, 10)
        target1[0, 0] = np.nan
        target2 = np.random.randn(5, 10)
        attr1 = np.random.randn(3, 10)
        attr2 = np.random.randn(3, 10)
        
        with pytest.raises(ValueError, match="contains NaN values"):
            weat.compute((target1, target2), (attr1, attr2))

    def test_weat_inf_in_embeddings(self):
        """Test WEAT handles Inf values."""
        weat = WEAT()
        
        target1 = np.random.randn(5, 10)
        target1[0, 0] = np.inf
        target2 = np.random.randn(5, 10)
        attr1 = np.random.randn(3, 10)
        attr2 = np.random.randn(3, 10)
        
        with pytest.raises(ValueError, match="contains Inf values"):
            weat.compute((target1, target2), (attr1, attr2))

    def test_weat_mismatched_dimensions(self):
        """Test WEAT with mismatched embedding dimensions."""
        weat = WEAT()
        
        target1 = np.random.randn(5, 100)
        target2 = np.random.randn(5, 50)  # Mismatch
        attr1 = np.random.randn(3, 100)
        attr2 = np.random.randn(3, 100)
        
        with pytest.raises(ValueError, match="must have same dimension"):
            weat.compute((target1, target2), (attr1, attr2))

    def test_input_types_list(self):
        """Test functions handle Python lists (type conversion coverage)."""
        weat = WEAT()
        
        target1 = [[1.0, 0.0], [1.0, 0.0]]
        target2 = [[0.0, 1.0], [0.0, 1.0]]
        attr1 = [[1.0, 0.0]]
        attr2 = [[0.0, 1.0]]
        
        # WEAT with lists
        score = weat.compute((target1, target2), (attr1, attr2))
        assert isinstance(score, float)
    
    def test_different_group_sizes(self):
        """Test WEAT with unbalanced groups."""
        weat = WEAT()
        
        target1 = np.random.randn(3, 10)
        target2 = np.random.randn(10, 10)
        attr1 = np.random.randn(5, 10)
        attr2 = np.random.randn(2, 10)
        
        score = weat.compute((target1, target2), (attr1, attr2))
        assert isinstance(score, float)
