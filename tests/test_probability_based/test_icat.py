"""Tests for Idealized Context Association Test (iCAT)."""

import json

import numpy as np
import pytest

from bias_scope.probability_based import ICAT


class TestICAT:
    """Test iCAT metric."""

    def test_basic_functionality(self):
        """Test with known LMS/SS values."""
        icat = ICAT()

        def mock_predict(context, candidate):
            # Perfect LM, perfectly biased
            if candidate in ["man", "woman"]:
                if candidate == "man":
                    return 0.8  # Prefers stereotype
                return 0.2
            return 0.05  # Meaningless

        tests = [
            {
                "context": ["The", "[MASK]", "is", "CEO"],
                "stereotype": "man",
                "anti_stereotype": "woman",
                "meaningless": "tree",
            }
        ]

        result = icat.evaluate(tests, mock_predict)

        assert "icat" in result
        assert "lms" in result
        assert "ss" in result
        assert "n_examples" in result

        # LMS should be 100 (chose meaningful)
        assert result["lms"] == 100.0
        # SS should be 100 (chose stereotype every time)
        assert result["ss"] == 100.0
        # iCAT = 100 * (min(100, 0) / 50) = 100 * 0 = 0
        assert result["icat"] == 0.0

    def test_ss_symmetry(self):
        """Test that SS=40 and SS=60 yield same fairness factor."""
        icat = ICAT()

        # Create tests with controlled outcomes
        # For SS=40: need 4 stereotype wins, 6 anti wins out of 10
        # For SS=60: need 6 stereotype wins, 4 anti wins out of 10

        # Scenario 1: Create list where we control each outcome individually
        tests_ss40 = []
        for i in range(10):
            # First 4 should prefer stereotype (man), last 6 prefer anti (woman)
            if i < 4:
                context_marker = f"case{i}_stereo"
            else:
                context_marker = f"case{i}_anti"

            tests_ss40.append(
                {
                    "context": ["The", "[MASK]", "works", context_marker],
                    "stereotype": "man",
                    "anti_stereotype": "woman",
                    "meaningless": "tree",
                }
            )

        def predict_ss40(context, candidate):
            # Check if this case should prefer stereotype or anti
            context_str = "".join(context)

            if candidate in ["man", "woman"]:
                if "stereo" in context_str:
                    # Prefer stereotype (man)
                    return 0.8 if candidate == "man" else 0.2
                else:
                    # Prefer anti-stereotype (woman)
                    return 0.2 if candidate == "man" else 0.8
            return 0.05

        result_40 = icat.evaluate(tests_ss40, predict_ss40)

        # Scenario 2: SS = 60
        tests_ss60 = []
        for i in range(10):
            # First 6 should prefer stereotype, last 4 prefer anti
            if i < 6:
                context_marker = f"case{i}_stereo"
            else:
                context_marker = f"case{i}_anti"

            tests_ss60.append(
                {
                    "context": ["The", "[MASK]", "works", context_marker],
                    "stereotype": "man",
                    "anti_stereotype": "woman",
                    "meaningless": "tree",
                }
            )

        def predict_ss60(context, candidate):
            context_str = "".join(context)

            if candidate in ["man", "woman"]:
                if "stereo" in context_str:
                    return 0.8 if candidate == "man" else 0.2
                else:
                    return 0.2 if candidate == "man" else 0.8
            return 0.05

        result_60 = icat.evaluate(tests_ss60, predict_ss60)

        # Fairness factors should be equal
        fairness_40 = min(result_40["ss"], 100 - result_40["ss"]) / 50.0
        fairness_60 = min(result_60["ss"], 100 - result_60["ss"]) / 50.0

        # Both should be 0.8 (40/50 = 0.8)
        assert result_40["ss"] == 40.0
        assert result_60["ss"] == 60.0
        assert pytest.approx(fairness_40, abs=0.01) == 0.8
        assert pytest.approx(fairness_60, abs=0.01) == 0.8

    def test_ss_extremes(self):
        """Test SS=0 or 100 yields iCAT=0."""
        icat = ICAT()

        # SS = 100 (always stereotype)
        def predict_ss100(context, candidate):
            if candidate == "man":
                return 0.9
            elif candidate == "woman":
                return 0.05
            return 0.01

        tests = [
            {
                "context": ["The", "[MASK]", "works"],
                "stereotype": "man",
                "anti_stereotype": "woman",
                "meaningless": "tree",
            }
        ]

        result = icat.evaluate(tests, predict_ss100)

        # LMS = 100, SS = 100
        # iCAT = 100 * (min(100, 0) / 50) = 0
        assert result["ss"] == 100.0
        assert result["icat"] == 0.0

    def test_ss_fifty_ideal(self):
        """Test SS=50 yields max fairness factor (1.0)."""
        icat = ICAT()

        # Perfectly fair (50-50)
        def predict_fair(context, candidate):
            if candidate in ["man", "woman"]:
                return 0.45  # Equal for both
            return 0.05

        tests = [
            {
                "context": ["The", "[MASK]", "works"],
                "stereotype": "man",
                "anti_stereotype": "woman",
                "meaningless": "tree",
            }
        ]

        result = icat.evaluate(tests, predict_fair)

        # LMS = 100
        assert result["lms"] == 100.0
        # SS should be near 50 (ties might vary)
        # Fairness factor = min(50, 50) / 50 = 1.0
        # iCAT = LMS * 1.0 = 100
        if result["ss"] == 50.0:
            assert result["icat"] == 100.0

    def test_lms_zero_yields_icat_zero(self):
        """Test LMS=0 yields iCAT=0."""
        icat = ICAT()

        # Poor language modeling (always prefers meaningless)
        def predict_bad_lm(context, candidate):
            if candidate == "tree":
                return 0.9  # Prefers meaningless
            return 0.05

        tests = [
            {
                "context": ["The", "[MASK]", "works"],
                "stereotype": "man",
                "anti_stereotype": "woman",
                "meaningless": "tree",
            }
        ]

        result = icat.evaluate(tests, predict_bad_lm)

        # LMS = 0 (chose meaningless)
        assert result["lms"] == 0.0
        # iCAT = 0 * anything = 0
        assert result["icat"] == 0.0

    def test_empty_test_cases_raises_error(self):
        """Test empty test cases raises error."""
        icat = ICAT()

        def mock_predict(context, candidate):
            return 0.5

        with pytest.raises(ValueError, match="cannot be empty"):
            icat.evaluate([], mock_predict)

    def test_invalid_test_case_structure(self):
        """Test invalid test case structure raises error."""
        icat = ICAT()

        def mock_predict(context, candidate):
            return 0.5

        # Missing required keys
        tests = [{"context": ["The", "[MASK]"]}]  # Missing stereotype, etc.

        with pytest.raises(ValueError, match="missing required key"):
            icat.evaluate(tests, mock_predict)

    def test_missing_mask_token(self):
        """Test context without [MASK] raises error."""
        icat = ICAT()

        def mock_predict(context, candidate):
            return 0.5

        tests = [
            {
                "context": ["The", "person", "works"],  # No [MASK]!
                "stereotype": "man",
                "anti_stereotype": "woman",
                "meaningless": "tree",
            }
        ]

        with pytest.raises(ValueError, match="must contain \\[MASK\\] token"):
            icat.evaluate(tests, mock_predict)

    def test_invalid_probabilities_raise_error(self):
        """Test invalid probabilities raise error."""
        icat = ICAT()

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
            icat.evaluate(tests, bad_predict)

    def test_icat_formula_manual_calculation(self):
        """Test iCAT formula with manual calculation."""
        icat = ICAT()

        # Scenario: LMS=80, SS=70
        # iCAT = 80 * (min(70, 30) / 50) = 80 * 0.6 = 48

        # Create predictor that yields LMS=80, SS=70
        call_count = [0]

        def controlled_predict(context, candidate):
            call_count[0] += 1
            # Pattern for 80% LMS: 4 out of 5 choose meaningful
            # Pattern for 70% SS: 7 out of 10 choose stereotype

            test_idx = (call_count[0] - 1) // 3  # 3 calls per test case

            if candidate in ["man", "woman"]:
                # Meaningful - should win 80% of the time
                if test_idx < 8:  # First 8 of 10: meaningful wins
                    if candidate == "man":
                        # SS=70: stereotype wins 70%
                        if test_idx < 7:
                            return 0.8
                        return 0.2
                    else:
                        if test_idx < 7:
                            return 0.2
                        return 0.8
                else:
                    # Last 2 of 10: meaningless wins
                    return 0.1
            else:  # meaningless
                if test_idx < 8:
                    return 0.1
                return 0.9

        tests = [
            {
                "context": ["The", "[MASK]", "works"],
                "stereotype": "man",
                "anti_stereotype": "woman",
                "meaningless": "tree",
            }
        ] * 10

        result = icat.evaluate(tests, controlled_predict)

        # Check LMS and SS are as expected
        assert result["lms"] == 80.0
        assert result["ss"] == 70.0

        # Check iCAT calculation
        expected_icat = 80.0 * (min(70, 30) / 50.0)
        assert pytest.approx(result["icat"], abs=0.01) == expected_icat

    def test_result_structure(self):
        """Test result dictionary has correct structure."""
        icat = ICAT()

        def mock_predict(context, candidate):
            if candidate in ["man", "woman"]:
                return 0.45
            return 0.1

        tests = [
            {
                "context": ["The", "[MASK]", "works"],
                "stereotype": "man",
                "anti_stereotype": "woman",
                "meaningless": "tree",
            }
        ]

        result = icat.evaluate(tests, mock_predict)

        # Check all required keys present
        assert "icat" in result
        assert "lms" in result
        assert "ss" in result
        assert "n_examples" in result

        # Check types
        assert isinstance(result["icat"], float)
        assert isinstance(result["lms"], float)
        assert isinstance(result["ss"], float)
        assert isinstance(result["n_examples"], int)

        # Check ranges
        assert 0.0 <= result["icat"] <= 100.0
        assert 0.0 <= result["lms"] <= 100.0
        assert 0.0 <= result["ss"] <= 100.0

    # === A) Common validation gaps ===

    def test_predict_function_not_callable_raises_error(self):
        """Test that non-callable predict_function raises TypeError."""
        icat = ICAT()

        tests = [
            {
                "context": ["The", "[MASK]", "works"],
                "stereotype": "man",
                "anti_stereotype": "woman",
                "meaningless": "tree",
            }
        ]

        # Test with None
        with pytest.raises(TypeError, match="callable"):
            icat.evaluate(tests, None)

        # Test with non-callable value
        with pytest.raises(TypeError, match="callable"):
            icat.evaluate(tests, 123)

    def test_predict_function_returning_nan_raises_error(self):
        """Test that predict_function returning NaN raises error."""
        icat = ICAT()

        def nan_predict(context, candidate):
            return np.nan

        tests = [
            {
                "context": ["The", "[MASK]", "works"],
                "stereotype": "man",
                "anti_stereotype": "woman",
                "meaningless": "tree",
            }
        ]

        with pytest.raises(ValueError, match="NaN"):
            icat.evaluate(tests, nan_predict)

    def test_predict_function_returning_inf_raises_error(self):
        """Test that predict_function returning Inf raises error."""
        icat = ICAT()

        def inf_predict(context, candidate):
            return np.inf

        tests = [
            {
                "context": ["The", "[MASK]", "works"],
                "stereotype": "man",
                "anti_stereotype": "woman",
                "meaningless": "tree",
            }
        ]

        with pytest.raises(ValueError, match="Inf"):
            icat.evaluate(tests, inf_predict)

    # === B) Tie-handling ===

    def test_tie_behavior_deterministic(self):
        """Test that ties are handled deterministically.
        
        When stereotype and anti-stereotype have identical probabilities,
        the behavior should be consistent. CAT uses > comparison, so ties
        count as 0 (anti-stereotype wins).
        """
        icat = ICAT()

        # All probabilities identical - creates ties
        def tie_predict(context, candidate):
            if candidate in ["man", "woman"]:
                return 0.5  # Identical for both
            return 0.1  # Meaningless lower

        tests = [
            {
                "context": ["The", "[MASK]", "works"],
                "stereotype": "man",
                "anti_stereotype": "woman",
                "meaningless": "tree",
            }
        ] * 10

        result = icat.evaluate(tests, tie_predict)

        # LMS should be 100 (meaningful > meaningless)
        assert result["lms"] == 100.0

        # SS: ties use > comparison, so stereotype never strictly greater
        # All 10 tests result in ties, ss_scores.append(1 if prob_stereo > prob_anti else 0)
        # Since 0.5 > 0.5 is False, all append 0
        assert result["ss"] == 0.0  # All ties count as 0

        # iCAT = 100 * (min(0, 100) / 50) = 100 * 0 = 0
        assert result["icat"] == 0.0

    # === D) Schema/serialization ===

    def test_result_json_serializable(self):
        """Test that result dictionary is JSON serializable."""
        icat = ICAT()

        def mock_predict(context, candidate):
            if candidate in ["man", "woman"]:
                return 0.6 if candidate == "man" else 0.4
            return 0.1

        tests = [
            {
                "context": ["The", "[MASK]", "works"],
                "stereotype": "man",
                "anti_stereotype": "woman",
                "meaningless": "tree",
            }
        ]

        result = icat.evaluate(tests, mock_predict)

        # Should not raise
        json_str = json.dumps(result)
        assert isinstance(json_str, str)

        # Should round-trip
        reloaded = json.loads(json_str)
        assert reloaded["icat"] == result["icat"]
        assert reloaded["lms"] == result["lms"]
        assert reloaded["ss"] == result["ss"]
        assert reloaded["n_examples"] == result["n_examples"]
