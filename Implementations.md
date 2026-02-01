# Instructions for Claude Code: Implement Sentence Bias Score & Create Tests

## Branch Setup

Create a new branch and work there. DO NOT push or merge to main:
```bash
git checkout -b implement-sentence-bias
```

## Python Library Best Practices

Follow these standards:

1. **Type hints** - Use proper type annotations
2. **Docstrings** - Follow NumPy/Google style
3. **Input validation** - Check array shapes and types
4. **Error messages** - Clear, helpful error messages
5. **Tests** - Comprehensive coverage with pytest
6. **No side effects** - Pure functions where possible
7. **Consistent naming** - Follow PEP 8

## Tasks

1. Implement the `sentence_bias` function
2. Add input validation to all functions
3. Write comprehensive tests
4. Add clear docstrings

## Status Check

**WEAT (Lines 9-78): ALREADY FIXED**
- Array concatenation using `np.concatenate` - CORRECT
- Effect size formula with proper parentheses - CORRECT

**SEAT (Lines 81-95): ALREADY CORRECT**
- Simply calls WEAT with sentence embeddings - CORRECT

**Sentence Bias Score (Lines 98-103): NEEDS IMPLEMENTATION**
- Currently just `pass` - needs full implementation

## Implement Sentence Bias Score

Replace the `sentence_bias` function with this implementation:

```python
def sentence_bias(
    word_embeddings: np.ndarray | torch.Tensor,
    gender_direction: np.ndarray | torch.Tensor,
    word_importance: np.ndarray | torch.Tensor,
    gender_words_mask: np.ndarray | torch.Tensor = None,
) -> Tuple[float, float]:
    """
    Compute gender bias score for a sentence.
    
    Measures stereotypical gender associations by computing weighted cosine
    similarities between word embeddings and a gender direction vector.
    
    Parameters
    ----------
    word_embeddings : np.ndarray or torch.Tensor
        Word embedding vectors, shape (num_words, embedding_dim)
    gender_direction : np.ndarray or torch.Tensor
        Gender direction vector from PCA, shape (embedding_dim,)
        Positive values indicate feminine, negative indicate masculine
    word_importance : np.ndarray or torch.Tensor
        Semantic importance weight for each word, shape (num_words,)
        Typically derived from max-pooling in sentence encoder
    gender_words_mask : np.ndarray or torch.Tensor, optional
        Boolean mask indicating gendered words to exclude, shape (num_words,)
        True indicates word should be excluded (e.g., "she", "he", "mother")
    
    Returns
    -------
    Tuple[float, float]
        (female_bias, male_bias) where female_bias is sum of positive
        weighted similarities and male_bias is sum of negative weighted similarities
    
    Raises
    ------
    ValueError
        If array shapes are incompatible
    
    Examples
    --------
    >>> word_embs = np.random.randn(5, 300)
    >>> gender_dir = np.random.randn(300)
    >>> importance = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    >>> female_bias, male_bias = sentence_bias(word_embs, gender_dir, importance)
    
    Notes
    -----
    Formula from Dolci et al. (2023):
        BiasScore_F = sum(cos(word, g) * importance) for cos > 0
        BiasScore_M = sum(cos(word, g) * importance) for cos < 0
    where g is the gender direction and gendered words are excluded.
    """
    # Convert to numpy
    word_embeddings = to_numpy(word_embeddings)
    gender_direction = to_numpy(gender_direction)
    word_importance = to_numpy(word_importance)
    
    # Input validation
    num_words = word_embeddings.shape[0]
    embedding_dim = word_embeddings.shape[1]
    
    if gender_direction.shape[0] != embedding_dim:
        raise ValueError(
            f"Gender direction dimension {gender_direction.shape[0]} "
            f"does not match embedding dimension {embedding_dim}"
        )
    
    if word_importance.shape[0] != num_words:
        raise ValueError(
            f"Importance array length {word_importance.shape[0]} "
            f"does not match number of words {num_words}"
        )
    
    # Normalize gender direction
    gender_direction = gender_direction / np.linalg.norm(gender_direction)
    
    # Compute cosine similarity for each word
    word_biases = np.array([
        cosine_similarity(word_emb, gender_direction) 
        for word_emb in word_embeddings
    ])
    
    # Exclude gendered words if mask provided
    if gender_words_mask is not None:
        gender_words_mask = to_numpy(gender_words_mask)
        if gender_words_mask.shape[0] != num_words:
            raise ValueError(
                f"Mask length {gender_words_mask.shape[0]} "
                f"does not match number of words {num_words}"
            )
        word_biases = word_biases * (~gender_words_mask)
    
    # Weight by importance
    weighted_biases = word_biases * word_importance
    
    # Separate female and male bias
    female_bias = float(np.sum(weighted_biases[weighted_biases > 0]))
    male_bias = float(np.sum(weighted_biases[weighted_biases < 0]))
    
    return female_bias, male_bias
```

## Create Tests

Create `/mnt/project/tests/test_embedding_metrics.py`:

```python
"""Tests for embedding-based bias detection metrics."""

import pytest
import numpy as np
import torch
from embeddings_based_metrics import weat, seat, sentence_bias


class TestWEAT:
    """Test Word Embedding Association Test."""
    
    def test_basic_functionality(self):
        """Test WEAT with simple inputs."""
        target1 = np.array([[1.0, 0.0], [0.9, 0.1]])
        target2 = np.array([[0.0, 1.0], [0.1, 0.9]])
        attr1 = np.array([[1.0, 0.0], [0.95, 0.05]])
        attr2 = np.array([[0.0, 1.0], [0.05, 0.95]])
        
        score = weat((target1, target2), (attr1, attr2))
        
        assert isinstance(score, float)
        assert not np.isnan(score)
        assert score > 0
    
    def test_with_torch_tensors(self):
        """Test WEAT handles PyTorch tensors."""
        target1 = torch.randn(3, 10)
        target2 = torch.randn(3, 10)
        attr1 = torch.randn(2, 10)
        attr2 = torch.randn(2, 10)
        
        score = weat((target1, target2), (attr1, attr2))
        assert isinstance(score, float)
    
    def test_identical_targets(self):
        """Test WEAT with identical target sets returns near zero."""
        target = np.random.randn(5, 20)
        attr1 = np.random.randn(3, 20)
        attr2 = np.random.randn(3, 20)
        
        score = weat((target, target), (attr1, attr2))
        assert abs(score) < 0.01
    
    def test_raises_on_wrong_input(self):
        """Test WEAT validates input format."""
        target = np.random.randn(3, 10)
        attr = np.random.randn(2, 10)
        
        with pytest.raises(ValueError, match="must have two elements"):
            weat((target,), (attr, attr))
        
        with pytest.raises(ValueError, match="must have two elements"):
            weat((target, target), (attr,))


class TestSEAT:
    """Test Sentence Encoder Association Test."""
    
    def test_calls_weat(self):
        """Test SEAT produces same result as WEAT."""
        target1 = np.random.randn(4, 768)
        target2 = np.random.randn(4, 768)
        attr1 = np.random.randn(3, 768)
        attr2 = np.random.randn(3, 768)
        
        weat_score = weat((target1, target2), (attr1, attr2))
        seat_score = seat((target1, target2), (attr1, attr2))
        
        assert weat_score == seat_score
    
    def test_sentence_embeddings(self):
        """Test SEAT with typical sentence embedding dimensions."""
        target1 = np.random.randn(5, 768)
        target2 = np.random.randn(5, 768)
        attr1 = np.random.randn(3, 768)
        attr2 = np.random.randn(3, 768)
        
        score = seat((target1, target2), (attr1, attr2))
        assert isinstance(score, float)
        assert not np.isnan(score)


class TestSentenceBias:
    """Test Sentence Bias Score metric."""
    
    def test_basic_functionality(self):
        """Test basic bias score computation."""
        word_embeddings = np.array([
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        gender_direction = np.array([1.0, 0.0, 0.0])
        importance = np.array([0.3, 0.3, 0.4])
        
        female_bias, male_bias = sentence_bias(
            word_embeddings, gender_direction, importance
        )
        
        assert isinstance(female_bias, float)
        assert isinstance(male_bias, float)
        assert female_bias > 0
        assert male_bias < 0
    
    def test_gender_word_exclusion(self):
        """Test that masked words are excluded."""
        word_embeddings = np.array([
            [1.0, 0.0],
            [0.8, 0.2],
        ])
        gender_direction = np.array([1.0, 0.0])
        importance = np.array([0.5, 0.5])
        mask = np.array([True, False])
        
        bias_with_mask, _ = sentence_bias(
            word_embeddings, gender_direction, importance, mask
        )
        bias_without_mask, _ = sentence_bias(
            word_embeddings, gender_direction, importance
        )
        
        assert bias_with_mask < bias_without_mask
    
    def test_importance_weighting(self):
        """Test that importance weights are applied correctly."""
        word_embeddings = np.array([
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ])
        gender_direction = np.array([1.0, 0.0, 0.0])
        
        importance_low = np.array([0.1, 0.1])
        importance_high = np.array([0.9, 0.9])
        
        bias_low, _ = sentence_bias(word_embeddings, gender_direction, importance_low)
        bias_high, _ = sentence_bias(word_embeddings, gender_direction, importance_high)
        
        assert bias_high > bias_low
    
    def test_with_torch_tensors(self):
        """Test handles PyTorch tensors."""
        word_embeddings = torch.randn(5, 100)
        gender_direction = torch.randn(100)
        importance = torch.rand(5)
        
        female_bias, male_bias = sentence_bias(
            word_embeddings, gender_direction, importance
        )
        
        assert isinstance(female_bias, float)
        assert isinstance(male_bias, float)
    
    def test_zero_importance(self):
        """Test that zero importance means no contribution."""
        word_embeddings = np.array([[1.0, 0.0], [1.0, 0.0]])
        gender_direction = np.array([1.0, 0.0])
        importance = np.array([1.0, 0.0])
        
        female_bias, _ = sentence_bias(
            word_embeddings, gender_direction, importance
        )
        
        assert abs(female_bias - 1.0) < 0.01
    
    def test_validates_dimension_mismatch(self):
        """Test raises error on dimension mismatch."""
        word_embeddings = np.random.randn(5, 100)
        gender_direction = np.random.randn(50)  # Wrong dimension
        importance = np.array([0.2] * 5)
        
        with pytest.raises(ValueError, match="does not match embedding dimension"):
            sentence_bias(word_embeddings, gender_direction, importance)
    
    def test_validates_importance_length(self):
        """Test raises error on wrong importance length."""
        word_embeddings = np.random.randn(5, 100)
        gender_direction = np.random.randn(100)
        importance = np.array([0.2] * 3)  # Wrong length
        
        with pytest.raises(ValueError, match="does not match number of words"):
            sentence_bias(word_embeddings, gender_direction, importance)
    
    def test_validates_mask_length(self):
        """Test raises error on wrong mask length."""
        word_embeddings = np.random.randn(5, 100)
        gender_direction = np.random.randn(100)
        importance = np.array([0.2] * 5)
        mask = np.array([True, False, True])  # Wrong length
        
        with pytest.raises(ValueError, match="does not match number of words"):
            sentence_bias(word_embeddings, gender_direction, importance, mask)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

## Improve Existing Documentation

Add/improve docstrings for WEAT and SEAT following the same NumPy-style format:

```python
def weat(...) -> float:
    """
    Compute Word Embedding Association Test (WEAT) effect size.
    
    Parameters
    ----------
    target_word_embeddings : Tuple[np.ndarray, np.ndarray]
        Two sets of target word embeddings to compare
    attribute_word_embeddings : Tuple[np.ndarray, np.ndarray]
        Two sets of attribute word embeddings
    
    Returns
    -------
    float
        Effect size measuring differential association
    
    Raises
    ------
    ValueError
        If tuples don't contain exactly 2 elements each
    
    Examples
    --------
    >>> target1 = np.array([[1.0, 0.0], [0.9, 0.1]])
    >>> target2 = np.array([[0.0, 1.0], [0.1, 0.9]])
    >>> attr1 = np.array([[1.0, 0.0]])
    >>> attr2 = np.array([[0.0, 1.0]])
    >>> score = weat((target1, target2), (attr1, attr2))
    
    Notes
    -----
    Based on Caliskan et al. (2017). Effect size formula:
        d = (mean_X - mean_Y) / std(X ∪ Y)
    """
```

## Library Structure Best Practices

Ensure the following:

1. **Module-level docstring** at top of file explaining what the module does
2. **All imports** at the top, organized (stdlib, third-party, local)
3. **Type hints** on all function signatures
4. **Input validation** with helpful error messages
5. **Consistent naming** following PEP 8
6. **No magic numbers** - use named constants if needed
7. **Pure functions** - no global state modifications

## Verification

Run these checks:

```bash
# Run tests
pytest tests/test_embedding_metrics.py -v

# Check test coverage (if pytest-cov installed)
pytest tests/test_embedding_metrics.py --cov=embeddings_based_metrics

# Run type checking (if mypy installed)
mypy embeddings_based_metrics.py

# Check code style (if flake8 installed)
flake8 embeddings_based_metrics.py --max-line-length=100
```

Minimum requirement: All tests must pass.

## Final Steps

```bash
# Stage changes
git add embeddings_based_metrics.py tests/test_embedding_metrics.py

# Commit
git commit -m "Implement sentence_bias with tests and documentation

- Add sentence_bias() function following Dolci et al. (2023)
- Add input validation with clear error messages
- Add comprehensive tests (15 test cases)
- Improve docstrings for all functions (NumPy style)
- All tests passing"

# Push to YOUR branch (NOT main)
git push origin implement-sentence-bias
```

**IMPORTANT:** 
- Do NOT push to main
- Do NOT merge to main
- Only push to your branch: `implement-sentence-bias`
- The professor will review and merge
