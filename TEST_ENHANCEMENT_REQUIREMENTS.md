# Test Enhancement Requirements for Production-Ready Library

## Context
This is a comprehensive bias detection library for production use, not just a student project. Test coverage must be thorough and professional-grade.

## Current Status
- 15 tests implemented
- Basic functionality covered
- Missing critical edge cases and error conditions

## Required Test Additions

### 1. WEAT Edge Cases (Add 5 tests)

```python
def test_weat_single_embedding_per_group():
    """Test WEAT with single embedding in each group."""
    # Minimum viable input
    
def test_weat_empty_embeddings():
    """Test WEAT raises error with empty arrays."""
    # Should fail gracefully
    
def test_weat_high_dimensional_embeddings():
    """Test WEAT with realistic embedding dimensions."""
    # 300-dim (Word2Vec), 768-dim (BERT), 4096-dim (InferSent)
    
def test_weat_nan_in_embeddings():
    """Test WEAT handles NaN values."""
    # Should either raise error or handle gracefully
    
def test_weat_inf_in_embeddings():
    """Test WEAT handles Inf values."""
    # Should either raise error or handle gracefully
```

### 2. SEAT Edge Cases (Add 3 tests)

```python
def test_seat_very_long_sentences():
    """Test SEAT with long sentence embeddings."""
    # 50+ word sentences
    
def test_seat_single_sentence_per_group():
    """Test SEAT with minimal input."""
    
def test_seat_different_group_sizes():
    """Test SEAT with unbalanced groups."""
    # 3 sentences in one group, 10 in another
```

### 3. Sentence Bias Score Edge Cases (Add 8 tests)

```python
def test_sentence_bias_all_words_masked():
    """Test behavior when all words are masked."""
    # Should return (0.0, 0.0)
    
def test_sentence_bias_no_words_masked():
    """Test behavior when mask is all False."""
    # Should behave same as no mask
    
def test_sentence_bias_single_word():
    """Test with single word sentence."""
    
def test_sentence_bias_very_long_sentence():
    """Test with 50+ word sentence."""
    # Realistic long sentence
    
def test_sentence_bias_negative_importance():
    """Test raises error on negative importance values."""
    # Invalid input
    
def test_sentence_bias_importance_not_normalized():
    """Test works with importance not summing to 1.0."""
    # Should still work correctly
    
def test_sentence_bias_nan_in_embeddings():
    """Test handles NaN in word embeddings."""
    
def test_sentence_bias_zero_gender_direction():
    """Test handles zero vector as gender direction."""
    # Should raise error (can't normalize)
```

### 4. Integration Tests (Add 3 tests)

```python
def test_weat_real_world_gender_career():
    """Test WEAT on real gender-career bias example."""
    # Use actual word embeddings if available, or simulate realistic scenario
    # Male/female names vs career/family words
    
def test_sentence_bias_realistic_sentence():
    """Test Sentence Bias Score on realistic biased sentence."""
    # "The nurse prepared the medicine" vs "The doctor prepared the medicine"
    
def test_metrics_consistency():
    """Test that metrics produce consistent results across runs."""
    # With same random seed, results should be identical
```

### 5. Input Validation Tests (Add 3 tests)

```python
def test_weat_mismatched_dimensions():
    """Test WEAT with mismatched embedding dimensions."""
    # target1: 100-dim, target2: 50-dim (should fail)
    
def test_sentence_bias_non_boolean_mask():
    """Test sentence_bias with non-boolean mask values."""
    # mask with integers instead of booleans
    
def test_type_errors():
    """Test proper error messages for wrong input types."""
    # String instead of array, etc.
```

## Total Target: 37+ tests (currently 15, need 22+ more)

## Quality Standards

### Test Quality Requirements:
1. **Descriptive docstrings** - Explain what and why
2. **Clear assertions** - Specific error messages with pytest.raises
3. **Realistic data** - Use dimensions from actual models (300, 768, 4096)
4. **Edge case documentation** - Comment why edge case matters
5. **No magic numbers** - Use named constants for dimensions

### Example of High-Quality Test:

```python
def test_sentence_bias_all_words_masked(self):
    """Test that masking all words returns zero bias.
    
    This ensures the function correctly handles the edge case where
    a user might mask all words, which should result in no bias
    being computed (both female and male bias should be 0.0).
    """
    # Realistic 300-dimensional embeddings (Word2Vec size)
    EMBEDDING_DIM = 300
    NUM_WORDS = 5
    
    word_embeddings = np.random.randn(NUM_WORDS, EMBEDDING_DIM)
    gender_direction = np.random.randn(EMBEDDING_DIM)
    importance = np.random.rand(NUM_WORDS)
    
    # Mask all words
    all_masked = np.ones(NUM_WORDS, dtype=bool)
    
    female_bias, male_bias = sentence_bias(
        word_embeddings, gender_direction, importance, all_masked
    )
    
    assert female_bias == 0.0, "Expected zero female bias when all words masked"
    assert male_bias == 0.0, "Expected zero male bias when all words masked"
```

## Error Handling Requirements

All functions must:
1. **Validate input shapes** before computation
2. **Provide clear error messages** that tell users how to fix the problem
3. **Handle NaN/Inf gracefully** - either raise error or skip values
4. **Document expected behavior** in docstrings

## Implementation Instructions

1. Read existing test file: `tests/test_embedding_metrics.py`
2. Add all required tests following the patterns above
3. Ensure each test has:
   - Clear docstring explaining purpose
   - Realistic test data (proper dimensions)
   - Specific assertions with error messages
   - Comments explaining edge cases
4. Run tests and ensure all pass
5. Aim for 35-40 total tests minimum

## Verification

After implementation, verify:
```bash
# All tests pass
pytest tests/test_embedding_metrics.py -v

# Check coverage (should be >90%)
pytest tests/test_embedding_metrics.py --cov=bias_scope.embeddings_based_metrics --cov-report=term-missing

# Tests run fast (<5 seconds total)
pytest tests/test_embedding_metrics.py --durations=10
```

## Expected Outcome

- **35-40 comprehensive tests**
- **>90% code coverage**
- **All edge cases handled**
- **Clear error messages for invalid inputs**
- **Production-ready test suite**

This is a foundation for a professional library that researchers and practitioners will rely on.
