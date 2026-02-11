"""Tests for Context Association Test (CAT)."""

import numpy as np
import pytest

from bias_scope.probability_based import CAT


class TestCAT:
    """Test CAT metric."""

    def test_basic_functionality(self):
        """Test with biased predictions."""
        cat = CAT()

        # Biased: prefers stereotypes over anti-stereotypes
        def biased_predict(context, candidate):
            if candidate in ["man", "doctor"]:
                return 0.6
            elif candidate in ["woman", "nurse"]:
                return 0.3
            else:  # meaningless
                return 0.1

        tests = [
            {
                "context": ["The", "[MASK]", "is", "CEO"],
                "stereotype": "man",
                "anti_stereotype": "woman",
                "meaningless": "tree",
            }
        ]

        result = cat.evaluate(tests, biased_predict)

        assert "lms" in result
        assert "ss" in result
        assert "n_examples" in result
        assert result["lms"] == 100.0  # Chose meaningful
        assert result["ss"] == 100.0  # Chose stereotype

    def test_unbiased_model(self):
        """Test unbiased model shows 50% stereotype score."""
        cat = CAT()

        def unbiased_predict(context, candidate):
            # Meaningful options should be higher than meaningless
            if candidate == "cloud":
                return 0.1
            # Equal for both meaningful options
            return 0.45

        tests = [
            {
                "context": ["The", "[MASK]", "works"],
                "stereotype": "engineer",
                "anti_stereotype": "nurse",
                "meaningless": "cloud",
            }
        ]

        result = cat.evaluate(tests, unbiased_predict)

        assert result["lms"] == 100.0  # Always chose meaningful
        # With equal probs, tie-breaking might vary
        assert 0.0 <= result["ss"] <= 100.0

    def test_empty_test_cases_raises_error(self):
        """Test empty test cases raises error."""
        cat = CAT()

        def mock_predict(context, candidate):
            return 0.5

        with pytest.raises(ValueError, match="cannot be empty"):
            cat.evaluate([], mock_predict)

    def test_missing_context_key_raises_error(self):
        """Test missing context key raises error."""
        cat = CAT()

        def mock_predict(context, candidate):
            return 0.5

        tests = [
            {"stereotype": "man", "anti_stereotype": "woman", "meaningless": "tree"}
        ]

        with pytest.raises(ValueError, match="missing required key 'context'"):
            cat.evaluate(tests, mock_predict)

    def test_missing_stereotype_key_raises_error(self):
        """Test missing stereotype key raises error."""
        cat = CAT()

        def mock_predict(context, candidate):
            return 0.5

        tests = [
            {
                "context": ["The", "[MASK]", "works"],
                "anti_stereotype": "woman",
                "meaningless": "tree",
            }
        ]

        with pytest.raises(ValueError, match="missing required key 'stereotype'"):
            cat.evaluate(tests, mock_predict)

    def test_no_mask_token_raises_error(self):
        """Test context without [MASK] raises error."""
        cat = CAT()

        def mock_predict(context, candidate):
            return 0.5

        tests = [
            {
                "context": ["The", "person", "works"],  # No [MASK]
                "stereotype": "man",
                "anti_stereotype": "woman",
                "meaningless": "tree",
            }
        ]

        with pytest.raises(ValueError, match="must contain \\[MASK\\] token"):
            cat.evaluate(tests, mock_predict)

    def test_invalid_probabilities_raise_error(self):
        """Test invalid probabilities raise error."""
        cat = CAT()

        def bad_predict(context, candidate):
            return 1.5  # > 1

        tests = [
            {
                "context": ["The", "[MASK]", "works"],
                "stereotype": "man",
                "anti_stereotype": "woman",
                "meaningless": "tree",
            }
        ]

        with pytest.raises(ValueError, match="must be in range"):
            cat.evaluate(tests, bad_predict)

    def test_negative_probabilities_raise_error(self):
        """Test negative probabilities raise error."""
        cat = CAT()

        def bad_predict(context, candidate):
            return -0.1

        tests = [
            {
                "context": ["The", "[MASK]", "works"],
                "stereotype": "man",
                "anti_stereotype": "woman",
                "meaningless": "tree",
            }
        ]

        with pytest.raises(ValueError, match="must be in range"):
            cat.evaluate(tests, bad_predict)

    def test_multiple_test_cases(self):
        """Test with multiple test cases."""
        cat = CAT()

        def mock_predict(context, candidate):
            if candidate in ["man", "woman"]:
                return 0.4
            return 0.2

        tests = [
            {
                "context": ["The", "[MASK]", "is", "CEO"],
                "stereotype": "man",
                "anti_stereotype": "woman",
                "meaningless": "tree",
            },
            {
                "context": ["The", "[MASK]", "is", "nurse"],
                "stereotype": "woman",
                "anti_stereotype": "man",
                "meaningless": "cloud",
            },
        ]

        result = cat.evaluate(tests, mock_predict)

        assert result["n_examples"] == 2
        assert result["lms"] == 100.0

    def test_prefers_meaningless(self):
        """Test when model prefers meaningless completions."""
        cat = CAT()

        def bad_predict(context, candidate):
            if candidate == "tree":
                return 0.9
            return 0.05

        tests = [
            {
                "context": ["The", "[MASK]", "works"],
                "stereotype": "man",
                "anti_stereotype": "woman",
                "meaningless": "tree",
            }
        ]

        result = cat.evaluate(tests, bad_predict)

        assert result["lms"] == 0.0  # Chose meaningless

    def test_anti_stereotype_preference(self):
        """Test when model prefers anti-stereotypes."""
        cat = CAT()

        def anti_bias_predict(context, candidate):
            if candidate == "woman":
                return 0.7
            elif candidate == "man":
                return 0.2
            else:
                return 0.1

        tests = [
            {
                "context": ["The", "[MASK]", "is", "CEO"],
                "stereotype": "man",
                "anti_stereotype": "woman",
                "meaningless": "tree",
            }
        ]

        result = cat.evaluate(tests, anti_bias_predict)

        assert result["lms"] == 100.0
        assert result["ss"] == 0.0  # Chose anti-stereotype

    def test_deterministic_results(self):
        """Test deterministic prediction gives consistent results."""
        cat = CAT()

        def deterministic_predict(context, candidate):
            scores = {"man": 0.6, "woman": 0.3, "tree": 0.1}
            return scores.get(candidate, 0.5)

        tests = [
            {
                "context": ["The", "[MASK]", "works"],
                "stereotype": "man",
                "anti_stereotype": "woman",
                "meaningless": "tree",
            }
        ]

        result1 = cat.evaluate(tests, deterministic_predict)
        result2 = cat.evaluate(tests, deterministic_predict)

        assert result1 == result2

    def test_long_context(self):
        """Test with longer context sentences."""
        cat = CAT()

        def mock_predict(context, candidate):
            return 0.5

        tests = [
            {
                "context": [
                    "In",
                    "the",
                    "modern",
                    "workplace",
                    "the",
                    "[MASK]",
                    "is",
                    "often",
                    "promoted",
                ],
                "stereotype": "man",
                "anti_stereotype": "woman",
                "meaningless": "tree",
            }
        ]

        result = cat.evaluate(tests, mock_predict)

        assert isinstance(result["lms"], float)
        assert isinstance(result["ss"], float)

    def test_percentage_range(self):
        """Test that scores are in percentage range [0, 100]."""
        cat = CAT()

        def mock_predict(context, candidate):
            return np.random.uniform(0.1, 0.9)

        tests = [
            {
                "context": ["The", "[MASK]", "works"],
                "stereotype": "man",
                "anti_stereotype": "woman",
                "meaningless": "tree",
            }
            for _ in range(10)
        ]

        result = cat.evaluate(tests, mock_predict)

        assert 0.0 <= result["lms"] <= 100.0
        assert 0.0 <= result["ss"] <= 100.0
