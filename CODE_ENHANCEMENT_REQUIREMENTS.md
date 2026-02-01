# Code Enhancement Requirements for Production-Ready Library

## Context
The current implementation has critical issues that must be fixed for production use. This is a comprehensive bias detection library that researchers will rely on - it must handle edge cases gracefully and provide clear error messages.

## Critical Issues to Fix

### 1. WEAT Function - Input Validation (HIGH PRIORITY)

**Current Issue**: No validation for empty arrays, dimension mismatches, or invalid values.

**Required Fixes**:

```python
def weat(...) -> float:
    # After line 52, add comprehensive validation:
    
    # Check for empty arrays
    if len(target_word_embeddings1) == 0 or len(target_word_embeddings2) == 0:
        raise ValueError(
            "Target embeddings cannot be empty. "
            f"Got {len(target_word_embeddings1)} and {len(target_word_embeddings2)} embeddings."
        )
    
    if len(attribute_word_embeddings1) == 0 or len(attribute_word_embeddings2) == 0:
        raise ValueError(
            "Attribute embeddings cannot be empty. "
            f"Got {len(attribute_word_embeddings1)} and {len(attribute_word_embeddings2)} embeddings."
        )
    
    # Check dimension consistency
    embedding_dims = [
        target_word_embeddings1.shape[1],
        target_word_embeddings2.shape[1],
        attribute_word_embeddings1.shape[1],
        attribute_word_embeddings2.shape[1]
    ]
    
    if len(set(embedding_dims)) != 1:
        raise ValueError(
            f"All embeddings must have the same dimensionality. "
            f"Got dimensions: target1={embedding_dims[0]}, target2={embedding_dims[1]}, "
            f"attr1={embedding_dims[2]}, attr2={embedding_dims[3]}"
        )
    
    # Check for NaN/Inf
    for name, arr in [
        ("target_word_embeddings1", target_word_embeddings1),
        ("target_word_embeddings2", target_word_embeddings2),
        ("attribute_word_embeddings1", attribute_word_embeddings1),
        ("attribute_word_embeddings2", attribute_word_embeddings2)
    ]:
        if np.isnan(arr).any():
            raise ValueError(f"{name} contains NaN values")
        if np.isinf(arr).any():
            raise ValueError(f"{name} contains Inf values")
```

### 2. WEAT Function - Division by Zero (HIGH PRIORITY)

**Current Issue**: Line 91-93 will crash if std is zero or if there's only one element.

**Required Fix**:

```python
# Replace lines 91-93 with:
    std_union = np.std(cos_target_union, ddof=1)
    
    # Handle zero/near-zero standard deviation
    if std_union < 1e-10:  # Numerical threshold
        raise ValueError(
            "Standard deviation of association scores is zero or near-zero. "
            "This occurs when all target embeddings have identical associations "
            "with the attribute embeddings. Cannot compute effect size."
        )
    
    # Handle single element case (ddof=1 makes std undefined)
    if len(cos_target_union) < 2:
        raise ValueError(
            "Need at least 2 total target embeddings to compute effect size. "
            f"Got {len(cos_target_union)} embeddings total."
        )
    
    return float((np.mean(cos_target1) - np.mean(cos_target2)) / std_union)
```

### 3. Sentence Bias - Zero Vector Normalization (HIGH PRIORITY)

**Current Issue**: Line 185 will crash if gender_direction is zero vector.

**Required Fix**:

```python
# Replace line 185 with:
    # Normalize gender direction
    gender_norm = np.linalg.norm(gender_direction)
    
    if gender_norm < 1e-10:  # Numerical threshold
        raise ValueError(
            "Gender direction vector has zero or near-zero magnitude. "
            "Cannot normalize. Please provide a non-zero gender direction vector."
        )
    
    gender_direction = gender_direction / gender_norm
```

### 4. Sentence Bias - Negative Importance Validation (HIGH PRIORITY)

**Current Issue**: No validation that importance values are non-negative.

**Required Fix**:

```python
# Add after line 182:
    
    # Validate importance values are non-negative
    if (word_importance < 0).any():
        raise ValueError(
            "Word importance values must be non-negative. "
            f"Found {np.sum(word_importance < 0)} negative values. "
            f"Min value: {np.min(word_importance)}"
        )
```

### 5. Sentence Bias - NaN/Inf Validation (HIGH PRIORITY)

**Required Fix**:

```python
# Add after converting to numpy (after line 166):
    
    # Validate no NaN/Inf values
    if np.isnan(word_embeddings).any():
        raise ValueError("word_embeddings contains NaN values")
    if np.isinf(word_embeddings).any():
        raise ValueError("word_embeddings contains Inf values")
    
    if np.isnan(gender_direction).any():
        raise ValueError("gender_direction contains NaN values")
    if np.isinf(gender_direction).any():
        raise ValueError("gender_direction contains Inf values")
    
    if np.isnan(word_importance).any():
        raise ValueError("word_importance contains NaN values")
    if np.isinf(word_importance).any():
        raise ValueError("word_importance contains Inf values")
```

### 6. Sentence Bias - Mask Validation (MEDIUM PRIORITY)

**Required Fix**:

```python
# Add after line 195:
    
    if gender_words_mask is not None:
        gender_words_mask = to_numpy(gender_words_mask)
        
        # Validate mask length
        if gender_words_mask.shape[0] != num_words:
            raise ValueError(
                f"Mask length {gender_words_mask.shape[0]} "
                f"does not match number of words {num_words}"
            )
        
        # Validate mask is boolean type
        if gender_words_mask.dtype != bool:
            raise TypeError(
                f"gender_words_mask must be boolean array. "
                f"Got dtype: {gender_words_mask.dtype}. "
                f"Convert using .astype(bool) if needed."
            )
        
        # Validate not all masked
        if gender_words_mask.all():
            # This is valid - return zero bias
            return 0.0, 0.0
            
        word_biases = word_biases * (~gender_words_mask)
```

### 7. Add Usage Examples to All Docstrings (MEDIUM PRIORITY)

**WEAT Example**:
```python
Examples
--------
Basic usage with word embeddings:

>>> import numpy as np
>>> # Male/Female names vs Career/Family words
>>> male_names = np.random.randn(5, 300)  # 300-dim embeddings
>>> female_names = np.random.randn(5, 300)
>>> career_words = np.random.randn(5, 300)
>>> family_words = np.random.randn(5, 300)
>>> 
>>> effect_size = weat(
...     (male_names, female_names),
...     (career_words, family_words)
... )
>>> print(f"Gender-career bias effect size: {effect_size:.3f}")
```

**Sentence Bias Example**:
```python
Examples
--------
Compute bias for a sentence:

>>> import numpy as np
>>> # 5 words, 300-dimensional embeddings
>>> words = np.random.randn(5, 300)
>>> gender_dir = np.random.randn(300)  # From PCA
>>> importance = np.array([0.15, 0.25, 0.20, 0.30, 0.10])
>>> 
>>> # Mark first word as gendered (e.g., "she")
>>> mask = np.array([True, False, False, False, False])
>>> 
>>> female_bias, male_bias = sentence_bias(
...     words, gender_dir, importance, mask
... )
>>> print(f"Female bias: {female_bias:.4f}")
>>> print(f"Male bias: {male_bias:.4f}")
```

### 8. Performance Optimization - Vectorize Sentence Bias (LOW PRIORITY)

**Current Issue**: Lines 188-191 use list comprehension which is slower.

**Optional Optimization**:
```python
# Replace lines 188-191 with vectorized version:
    # Normalize word embeddings for cosine similarity
    word_norms = np.linalg.norm(word_embeddings, axis=1, keepdims=True)
    word_norms = np.maximum(word_norms, 1e-10)  # Avoid division by zero
    normalized_words = word_embeddings / word_norms
    
    # Compute all cosine similarities at once (vectorized)
    word_biases = normalized_words @ gender_direction
```

Note: Only implement this if the original cosine_similarity function from utils
does the same normalization. Otherwise, keep the list comprehension for correctness.

## Implementation Order

**Phase 1 - Critical Fixes (Must Do)**:
1. Add all input validation to WEAT
2. Fix division by zero in WEAT
3. Fix zero vector normalization in sentence_bias
4. Add negative importance validation
5. Add NaN/Inf validation to all functions
6. Add mask type validation

**Phase 2 - Quality Improvements (Should Do)**:
7. Add usage examples to all docstrings
8. Add mask "all True" edge case handling

**Phase 3 - Performance (Nice to Have)**:
9. Vectorize sentence_bias if safe to do so

## Testing Requirements

After making these changes, ensure:
1. All existing tests still pass
2. Add new tests for all edge cases you handle
3. Test error messages are clear and helpful
4. Test with realistic data dimensions (300, 768, 4096)

## Quality Standards

Every error message must:
1. Explain what went wrong
2. Show the problematic values
3. Suggest how to fix it (when applicable)

Example good error message:
```python
raise ValueError(
    f"Mask length {mask.shape[0]} does not match number of words {num_words}. "
    f"Please provide a mask with exactly {num_words} boolean values."
)
```

Example bad error message:
```python
raise ValueError("Invalid mask")
```

## Expected Outcome

After implementation:
- **All edge cases handled gracefully**
- **Clear, actionable error messages**
- **No crashes on invalid input**
- **Production-ready code quality**
- **Comprehensive input validation**

This library will be used by researchers who need reliability and clarity when things go wrong.
