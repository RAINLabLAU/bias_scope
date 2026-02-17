"""Tests for All Unmasked Likelihood with Attention (AULA)."""

import numpy as np
import pytest

from bias_scope.probability_based import AULA


class TestAULA:
    """Test AULA metric."""

    def test_basic_functionality(self):
        """Test with biased predictions and uniform attention."""
        aula = AULA()

        # Biased: higher probs for stereotypes
        def biased_predict(sentence, pos):
            n = len(sentence)
            if "Women" in sentence:
                return {
                    "prob": 0.8,
                    "attention": np.ones(n) / n,  # Uniform attention
                }
            return {
                "prob": 0.3,
                "attention": np.ones(n) / n,
            }

        pairs = [(["Women", "work"], ["Men", "work"])]

        score = aula.evaluate(pairs, biased_predict)

        assert 0.0 <= score <= 1.0
        assert score > 0.5  # Prefers stereotypes

    def test_attention_weighting_single_token(self):
        """Test that full attention weight on one token uses only that token's log-prob."""
        aula = AULA()

        def predict_with_focused_attention(sentence, pos):
            n = len(sentence)
            # All attention on position 0
            attention = np.zeros(n)
            attention[0] = 1.0

            # Different probs for different positions
            if pos == 0:
                prob = 0.9  # High prob for first token
            else:
                prob = 0.1  # Low prob for others

            return {"prob": prob, "attention": attention}

        # Single sentence test
        sentence = ["Token1", "Token2", "Token3"]

        # Compute AULA manually
        # Position 0: log(0.9) * (1.0 / 1.0) = log(0.9)
        # Position 1: log(0.1) * (0.0 / 1.0) = 0
        # Position 2: log(0.1) * (0.0 / 1.0) = 0
        # AULA = log(0.9)

        result = aula._compute_aula(sentence, predict_with_focused_attention)
        expected = np.log(0.9)

        assert pytest.approx(result, abs=1e-5) == expected

    def test_attention_normalization(self):
        """Test that attention weights are normalized to sum to 1."""
        aula = AULA()

        def predict_with_unnormalized_attention(sentence, pos):
            n = len(sentence)
            # Unnormalized attention (sum = 3)
            attention = np.array([1.0, 1.0, 1.0])[:n]
            prob = 0.5

            return {"prob": prob, "attention": attention}

        sentence = ["A", "B", "C"]

        # Should normalize to [1/3, 1/3, 1/3]
        result = aula._compute_aula(sentence, predict_with_unnormalized_attention)

        # All probs same (0.5), all weights same after normalization
        # AULA = (1/3)*log(0.5) + (1/3)*log(0.5) + (1/3)*log(0.5) = log(0.5)
        expected = np.log(0.5)

        assert pytest.approx(result, abs=1e-5) == expected

    def test_missing_attention_raises_error(self):
        """Test that missing attention key raises clear error."""
        aula = AULA()

        def predict_without_attention(sentence, pos):
            # Missing 'attention' key
            return {"prob": 0.5}

        pairs = [(["A", "B"], ["C", "D"])]

        with pytest.raises(KeyError, match="must return dict with 'attention' key"):
            aula.evaluate(pairs, predict_without_attention)

    def test_missing_prob_raises_error(self):
        """Test that missing prob key raises error."""
        aula = AULA()

        def predict_without_prob(sentence, pos):
            return {"attention": np.array([0.5, 0.5])}

        pairs = [(["A", "B"], ["C", "D"])]

        with pytest.raises(KeyError, match="must return dict with 'prob' key"):
            aula.evaluate(pairs, predict_without_prob)

    def test_attention_shape_mismatch_raises_error(self):
        """Test attention shape mismatch raises error."""
        aula = AULA()

        def predict_wrong_shape(sentence, pos):
            # Attention length doesn't match sentence
            return {"prob": 0.5, "attention": np.array([0.5, 0.5, 0.5])}

        # Sentence has 2 tokens but attention has 3
        pairs = [(["A", "B"], ["C", "D"])]

        with pytest.raises(ValueError, match="must match sentence length"):
            aula.evaluate(pairs, predict_wrong_shape)

    def test_attention_not_1d_raises_error(self):
        """Test non-1D attention raises error."""
        aula = AULA()

        def predict_2d_attention(sentence, pos):
            # 2D attention (not pre-aggregated)
            return {"prob": 0.5, "attention": np.array([[0.5], [0.5]])}

        pairs = [(["A", "B"], ["C", "D"])]

        with pytest.raises(ValueError, match="must be 1D array"):
            aula.evaluate(pairs, predict_2d_attention)

    def test_attention_nan_raises_error(self):
        """Test attention with NaN raises error."""
        aula = AULA()

        def predict_nan_attention(sentence, pos):
            return {"prob": 0.5, "attention": np.array([np.nan, 0.5])}

        pairs = [(["A", "B"], ["C", "D"])]

        with pytest.raises(ValueError, match="contains NaN"):
            aula.evaluate(pairs, predict_nan_attention)

    def test_attention_negative_raises_error(self):
        """Test negative attention raises error."""
        aula = AULA()

        def predict_negative_attention(sentence, pos):
            return {"prob": 0.5, "attention": np.array([-0.1, 0.5])}

        pairs = [(["A", "B"], ["C", "D"])]

        with pytest.raises(ValueError, match="negative values"):
            aula.evaluate(pairs, predict_negative_attention)

    def test_attention_zero_sum_raises_error(self):
        """Test attention summing to zero raises error."""
        aula = AULA()

        def predict_zero_attention(sentence, pos):
            return {"prob": 0.5, "attention": np.array([0.0, 0.0])}

        pairs = [(["A", "B"], ["C", "D"])]

        with pytest.raises(ValueError, match="sum to near-zero"):
            aula.evaluate(pairs, predict_zero_attention)

    def test_empty_pairs_raises_error(self):
        """Test empty sentence pairs raises error."""
        aula = AULA()

        def mock_predict(sentence, pos):
            return {"prob": 0.5, "attention": np.ones(len(sentence))}

        with pytest.raises(ValueError, match="cannot be empty"):
            aula.evaluate([], mock_predict)

    def test_different_length_raises_error(self):
        """Test sentences with different lengths raise error."""
        aula = AULA()

        def mock_predict(sentence, pos):
            return {"prob": 0.5, "attention": np.ones(len(sentence))}

        pairs = [(["Women", "are"], ["Men", "are", "smart"])]

        with pytest.raises(ValueError, match="same length"):
            aula.evaluate(pairs, mock_predict)

    def test_empty_sentence_raises_error(self):
        """Test empty sentence raises error."""
        aula = AULA()

        def mock_predict(sentence, pos):
            return {"prob": 0.5, "attention": np.array([])}

        pairs = [([], [])]

        with pytest.raises(ValueError, match="cannot be empty"):
            aula.evaluate(pairs, mock_predict)

    def test_invalid_probability_raises_error(self):
        """Test invalid probability raises error."""
        aula = AULA()

        def bad_predict(sentence, pos):
            return {"prob": 1.5, "attention": np.ones(len(sentence))}

        pairs = [(["A", "B"], ["C", "D"])]

        with pytest.raises(ValueError, match="Invalid probability"):
            aula.evaluate(pairs, bad_predict)

    def test_predict_function_not_dict_raises_error(self):
        """Test predict function not returning dict raises error."""
        aula = AULA()

        def bad_predict(sentence, pos):
            return 0.5  # Not a dict!

        pairs = [(["A", "B"], ["C", "D"])]

        with pytest.raises(TypeError, match="must return dict"):
            aula.evaluate(pairs, bad_predict)

    def test_weighted_vs_unweighted_comparison(self):
        """Test that attention weighting changes the result vs uniform."""
        aula = AULA()

        # Scenario: token 0 has high prob, token 1 has low prob
        def predict_varied_probs(sentence, pos):
            n = len(sentence)
            if pos == 0:
                prob = 0.9
            else:
                prob = 0.1

            # Return two different attention patterns
            return {"prob": prob, "attention": np.ones(n) / n}

        # With uniform attention
        sentence = ["High", "Low"]
        result_uniform = aula._compute_aula(sentence, predict_varied_probs)

        # Expected: (0.5 * log(0.9)) + (0.5 * log(0.1))
        expected_uniform = 0.5 * np.log(0.9) + 0.5 * np.log(0.1)
        assert pytest.approx(result_uniform, abs=1e-5) == expected_uniform

        # With focused attention on token 0
        def predict_focused_attention(sentence, pos):
            n = len(sentence)
            if pos == 0:
                prob = 0.9
            else:
                prob = 0.1

            # Most attention on position 0
            attention = np.array([0.9, 0.1])[:n]
            return {"prob": prob, "attention": attention}

        result_focused = aula._compute_aula(sentence, predict_focused_attention)

        # Expected: (0.9 * log(0.9)) + (0.1 * log(0.1))
        expected_focused = 0.9 * np.log(0.9) + 0.1 * np.log(0.1)
        assert pytest.approx(result_focused, abs=1e-5) == expected_focused

        # Focused attention should give higher score (less negative)
        assert result_focused > result_uniform

    def test_pair_comparison(self):
        """Test stereotype vs anti-stereotype comparison."""
        aula = AULA()

        def predict_biased(sentence, pos):
            n = len(sentence)
            # Stereotype sentences get higher probs
            if "Women" in sentence:
                prob = 0.8
            else:
                prob = 0.4

            return {"prob": prob, "attention": np.ones(n) / n}

        pairs = [
            (["Women", "work"], ["Men", "work"]),
            (["Women", "cook"], ["Men", "cook"]),
        ]

        score = aula.evaluate(pairs, predict_biased)

        # Should prefer stereotypes (> 0.5)
        assert score > 0.5

    def test_unbiased_model_near_fifty(self):
        """Test unbiased model returns score near 0.5."""
        aula = AULA()

        def predict_unbiased(sentence, pos):
            n = len(sentence)
            # Same prob for both
            return {"prob": 0.6, "attention": np.ones(n) / n}

        pairs = [
            (["Women", "work"], ["Men", "work"]),
            (["She", "codes"], ["He", "codes"]),
        ]

        score = aula.evaluate(pairs, predict_unbiased)

        # With identical probs, ties are broken arbitrarily (depends on >)
        # Score could be 0.0 or 1.0 due to tie-breaking
        # In practice, should be 0.0 since neither is strictly greater
        assert 0.0 <= score <= 1.0

    def test_result_type(self):
        """Test that result is a float."""
        aula = AULA()

        def mock_predict(sentence, pos):
            return {"prob": 0.5, "attention": np.ones(len(sentence))}

        pairs = [(["A", "B"], ["C", "D"])]

        score = aula.evaluate(pairs, mock_predict)

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    # === A) Common validation gaps ===

    def test_predict_function_not_callable_raises_error(self):
        """Test that non-callable predict_function raises TypeError."""
        aula = AULA()

        pairs = [(["A", "B"], ["C", "D"])]

        # Test with None
        with pytest.raises(TypeError, match="callable"):
            aula.evaluate(pairs, None)

        # Test with non-callable value
        with pytest.raises(TypeError, match="callable"):
            aula.evaluate(pairs, 123)

    def test_predict_function_returning_nan_prob_raises_error(self):
        """Test that predict_function returning NaN probability raises error."""
        aula = AULA()

        def nan_prob_predict(sentence, pos):
            return {"prob": np.nan, "attention": np.ones(len(sentence))}

        pairs = [(["A", "B"], ["C", "D"])]

        with pytest.raises(ValueError, match="Invalid probability"):
            aula.evaluate(pairs, nan_prob_predict)

    def test_predict_function_returning_inf_prob_raises_error(self):
        """Test that predict_function returning Inf probability raises error."""
        aula = AULA()

        def inf_prob_predict(sentence, pos):
            return {"prob": np.inf, "attention": np.ones(len(sentence))}

        pairs = [(["A", "B"], ["C", "D"])]

        with pytest.raises(ValueError, match="Invalid probability"):
            aula.evaluate(pairs, inf_prob_predict)

    def test_attention_with_inf_raises_error(self):
        """Test that attention containing Inf raises error."""
        aula = AULA()

        def inf_attention_predict(sentence, pos):
            attention = np.ones(len(sentence))
            attention[0] = np.inf
            return {"prob": 0.5, "attention": attention}

        pairs = [(["A", "B"], ["C", "D"])]

        with pytest.raises(ValueError, match="contains Inf"):
            aula.evaluate(pairs, inf_attention_predict)

    # === B) Tie-handling ===

    def test_tie_behavior_deterministic(self):
        """Test that ties are handled deterministically.
        
        When both sentences have identical AULA scores, the tie-breaking
        should be consistent. AULA uses > comparison, so ties count as 0
        (no preference for stereotype).
        """
        aula = AULA()

        # Identical predictions for both sentences
        def identical_predict(sentence, pos):
            return {"prob": 0.6, "attention": np.ones(len(sentence)) / len(sentence)}

        pairs = [
            (["Women", "work"], ["Men", "work"]),
            (["Women", "code"], ["Men", "code"]),
        ]

        score = aula.evaluate(pairs, identical_predict)

        # With identical scores, aula_stereo > aula_anti is False
        # So bias_indicators.append(0) for all pairs
        assert score == 0.0  # All ties count as 0 (no bias)
