# Testing Setup Guide

## Quick Start

### Option 1: Install All Dependencies (Recommended)

Install everything needed to run all tests in the repository:

```bash
pip install -r requirements.txt
```

This includes:
- `numpy` (required for all metrics)
- `torch` (required for embedding and probability metrics)
- `pytest` (required for testing)
- `coverage` (optional, for test coverage)
- `isort` (optional, for code formatting)

### Option 2: Minimal Install (Just for Generated Text Metrics)

If you only want to test the new generated text metrics and don't want to install the large `torch` package:

```bash
pip install -r requirements-test-only.txt
```

This includes:
- `numpy` (required)
- `pytest` (required)
- `coverage` (optional)

**Note:** With this minimal install, you can only test the `generated_text_based` metrics. Other tests will fail.

## Running the Tests

### Run All Generated Text Metrics Tests

```bash
# Run all 71 tests for the new metrics
pytest tests/test_generated_text_based/ -v
```

### Run Tests for a Specific Metric

```bash
# Social Group Substitution (17 tests)
pytest tests/test_generated_text_based/test_social_group_substitution.py -v

# Co-Occurrence Bias Score (15 tests)
pytest tests/test_generated_text_based/test_cooccurrence_bias_score.py -v

# Demographic Representation (12 tests)
pytest tests/test_generated_text_based/test_demographic_representation.py -v

# Stereotypical Associations (14 tests)
pytest tests/test_generated_text_based/test_stereotypical_associations.py -v

# Marked Persons (13 tests)
pytest tests/test_generated_text_based/test_marked_persons.py -v
```

### Run Tests with Coverage

```bash
# Generate coverage report
coverage run -m pytest tests/test_generated_text_based/
coverage report

# Generate HTML coverage report
coverage html
open htmlcov/index.html
```

### Run Verification Script

Quick verification that all metrics can be imported and used:

```bash
python verify_generated_text_metrics.py
```

## Using Poetry (Alternative)

If you prefer using Poetry (as configured in `pyproject.toml`):

```bash
# Install all dependencies including dev dependencies
poetry install --with dev

# Run tests
poetry run pytest tests/test_generated_text_based/ -v

# Run verification
poetry run python verify_generated_text_metrics.py
```

## Troubleshooting

### Import Errors

If you get `ModuleNotFoundError`, make sure the `src` directory is in your Python path:

```bash
# Add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Or run tests from the repository root
cd /path/to/bias_scope
pytest tests/test_generated_text_based/ -v
```

### Torch Installation Issues

If `torch` installation fails or takes too long:
1. Use `requirements-test-only.txt` instead
2. Only run tests for `generated_text_based` metrics
3. The new metrics don't require torch

### Test Failures

If any tests fail:
1. Check that you're running from the repository root
2. Verify all dependencies are installed: `pip list | grep -E "numpy|pytest"`
3. Try running a single test to isolate the issue
4. Check Python version: `python --version` (should be >= 3.10)

## Expected Results

With all dependencies installed, you should see:

```
tests/test_generated_text_based/test_social_group_substitution.py ........ (17 passed)
tests/test_generated_text_based/test_cooccurrence_bias_score.py ........ (15 passed)
tests/test_generated_text_based/test_demographic_representation.py ........ (12 passed)
tests/test_generated_text_based/test_stereotypical_associations.py ........ (14 passed)
tests/test_generated_text_based/test_marked_persons.py ........ (13 passed)

========================= 71 passed in X.XXs =========================
```
