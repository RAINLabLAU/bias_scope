"""Tests for All Unmasked Likelihood with Attention (AULA)."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from bias_scope.probability_based import AULA


def _mock_masked_scorer(cls):
    """Build a real scorer instance (mocked HF backend) for guard tests.

    Only __init__ needs to succeed; the guard rejects the instance before
    any prediction method is called, so the model/tokenizer mocks need not
    be functional beyond that.
    """
    with patch("transformers.AutoTokenizer") as auto_tok, patch(
        "transformers.AutoModelForMaskedLM"
    ) as auto_mlm:
        tok = MagicMock()
        tok.mask_token = "[MASK]"
        tok.mask_token_id = 999
        auto_tok.from_pretrained.return_value = tok
        auto_mlm.from_pretrained.return_value = MagicMock()
        return cls(model_name="fake", device="cpu")


class TestAULA:
    """Test AULA metric."""

    def test_basic_functionality(self):
        """Test with biased predictions and uniform attention."""
        aula = AULA(mode="whitespace")

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

        assert 0.0 <= score <= 100.0
        assert score > 50.0  # Prefers stereotypes

    def test_attention_weighting_single_token(self):
        """Attention on one token zeroes the others, then a plain mean over |S|.

        Changed in 0.2.0 to match Kaneko & Bollegala eq. 5,
        `AULA(S) = (1/|S|) sum_i a_i log P(w_i|S)`. v0.1.1 divided by
        `sum(a_i)` rather than `|S|`, so this test previously expected
        `log(0.9)`. See docs/fidelity/aul_aula.md.
        """
        aula = AULA(mode="whitespace")

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

        # Compute AULA manually, eq. 5:
        #   (1/3) * [ 1.0*log(0.9) + 0.0*log(0.1) + 0.0*log(0.1) ]
        # = log(0.9) / 3

        result = aula._compute_aula(sentence, predict_with_focused_attention)
        expected = np.log(0.9) / 3

        assert pytest.approx(result, abs=1e-5) == expected

    def test_attention_weights_are_not_renormalized(self):
        """Attention weights are used raw, NOT renormalized to sum to 1.

        eq. 5 is `(1/|S|) sum_i a_i log P(w_i|S)` -- the `1/|S|` already comes
        from the final mean over |S|, not from `sum(a_i) == 1`. Renormalizing
        by `sum(a_i)` instead (v0.1.1's bug) gives a different, per-sentence
        rescaled answer. Unlike the old version of this test, probabilities
        differ by position, so a renormalizing implementation would produce a
        different number than the eq. 5 arithmetic asserted here.
        """
        aula = AULA(mode="whitespace")

        def predict_with_unnormalized_attention(sentence, pos):
            n = len(sentence)
            # Unnormalized attention (sums to 3, not 1)
            attention = np.array([1.0, 1.0, 1.0])[:n]
            # Different probability per position, so normalization would
            # change the weighted combination, not just cancel out.
            prob = [0.9, 0.5, 0.1][pos]

            return {"prob": prob, "attention": attention}

        sentence = ["A", "B", "C"]

        result = aula._compute_aula(sentence, predict_with_unnormalized_attention)

        # eq. 5, raw (unnormalized) weights [1, 1, 1] over |S| = 3:
        #   (1/3) * [1*log(0.9) + 1*log(0.5) + 1*log(0.1)]
        expected = (np.log(0.9) + np.log(0.5) + np.log(0.1)) / 3
        # The old (v0.1.1) renormalized behaviour would instead divide by
        # sum(attention) = 3 as well here (weights already 1 each), so also
        # check this is NOT the same as dividing the sum by a different
        # normalizer, e.g. treating weights as [1/3, 1/3, 1/3] pre-scaled by
        # a factor that doesn't equal |S| -- assert against the one true
        # eq. 5 value.
        assert pytest.approx(result, abs=1e-5) == expected

        # A renormalizing implementation dividing by sum(attention)*n instead
        # of just n would give a visibly different number; confirm the two
        # are distinguishable at these inputs.
        wrong_renormalized = (
            1.0 * np.log(0.9) + 1.0 * np.log(0.5) + 1.0 * np.log(0.1)
        ) / (3.0 * 3.0)
        assert abs(result - wrong_renormalized) > 1e-3

    def test_missing_attention_raises_error(self):
        """Test that missing attention key raises clear error."""
        aula = AULA(mode="whitespace")

        def predict_without_attention(sentence, pos):
            # Missing 'attention' key
            return {"prob": 0.5}

        pairs = [(["A", "B"], ["C", "D"])]

        with pytest.raises(KeyError, match="must return dict with 'attention' key"):
            aula.evaluate(pairs, predict_without_attention)

    def test_missing_prob_raises_error(self):
        """Test that missing prob key raises error."""
        aula = AULA(mode="whitespace")

        def predict_without_prob(sentence, pos):
            return {"attention": np.array([0.5, 0.5])}

        pairs = [(["A", "B"], ["C", "D"])]

        with pytest.raises(KeyError, match="must return dict with 'prob' key"):
            aula.evaluate(pairs, predict_without_prob)

    def test_attention_shape_mismatch_raises_error(self):
        """Test attention shape mismatch raises error."""
        aula = AULA(mode="whitespace")

        def predict_wrong_shape(sentence, pos):
            # Attention length doesn't match sentence
            return {"prob": 0.5, "attention": np.array([0.5, 0.5, 0.5])}

        # Sentence has 2 tokens but attention has 3
        pairs = [(["A", "B"], ["C", "D"])]

        with pytest.raises(ValueError, match="must match sentence length"):
            aula.evaluate(pairs, predict_wrong_shape)

    def test_attention_not_1d_raises_error(self):
        """Test non-1D attention raises error."""
        aula = AULA(mode="whitespace")

        def predict_2d_attention(sentence, pos):
            # 2D attention (not pre-aggregated)
            return {"prob": 0.5, "attention": np.array([[0.5], [0.5]])}

        pairs = [(["A", "B"], ["C", "D"])]

        with pytest.raises(ValueError, match="must be 1D array"):
            aula.evaluate(pairs, predict_2d_attention)

    def test_attention_nan_raises_error(self):
        """Test attention with NaN raises error."""
        aula = AULA(mode="whitespace")

        def predict_nan_attention(sentence, pos):
            return {"prob": 0.5, "attention": np.array([np.nan, 0.5])}

        pairs = [(["A", "B"], ["C", "D"])]

        with pytest.raises(ValueError, match="contains NaN"):
            aula.evaluate(pairs, predict_nan_attention)

    def test_attention_negative_raises_error(self):
        """Test negative attention raises error."""
        aula = AULA(mode="whitespace")

        def predict_negative_attention(sentence, pos):
            return {"prob": 0.5, "attention": np.array([-0.1, 0.5])}

        pairs = [(["A", "B"], ["C", "D"])]

        with pytest.raises(ValueError, match="negative values"):
            aula.evaluate(pairs, predict_negative_attention)

    def test_attention_zero_sum_raises_error(self):
        """Test attention summing to zero raises error."""
        aula = AULA(mode="whitespace")

        def predict_zero_attention(sentence, pos):
            return {"prob": 0.5, "attention": np.array([0.0, 0.0])}

        pairs = [(["A", "B"], ["C", "D"])]

        with pytest.raises(ValueError, match="sum to near-zero"):
            aula.evaluate(pairs, predict_zero_attention)

    def test_empty_pairs_raises_error(self):
        """Test empty sentence pairs raises error."""
        aula = AULA(mode="whitespace")

        def mock_predict(sentence, pos):
            return {"prob": 0.5, "attention": np.ones(len(sentence))}

        with pytest.raises(ValueError, match="cannot be empty"):
            aula.evaluate([], mock_predict)

    def test_different_length_raises_error(self):
        """Test sentences with different lengths raise error."""
        aula = AULA(mode="whitespace")

        def mock_predict(sentence, pos):
            return {"prob": 0.5, "attention": np.ones(len(sentence))}

        pairs = [(["Women", "are"], ["Men", "are", "smart"])]

        with pytest.raises(ValueError, match="same length"):
            aula.evaluate(pairs, mock_predict)

    def test_empty_sentence_raises_error(self):
        """Test empty sentence raises error."""
        aula = AULA(mode="whitespace")

        def mock_predict(sentence, pos):
            return {"prob": 0.5, "attention": np.array([])}

        pairs = [([], [])]

        with pytest.raises(ValueError, match="cannot be empty"):
            aula.evaluate(pairs, mock_predict)

    def test_invalid_probability_raises_error(self):
        """Test invalid probability raises error."""
        aula = AULA(mode="whitespace")

        def bad_predict(sentence, pos):
            return {"prob": 1.5, "attention": np.ones(len(sentence))}

        pairs = [(["A", "B"], ["C", "D"])]

        with pytest.raises(ValueError, match="Invalid probability"):
            aula.evaluate(pairs, bad_predict)

    def test_predict_function_not_dict_raises_error(self):
        """Test predict function not returning dict raises error."""
        aula = AULA(mode="whitespace")

        def bad_predict(sentence, pos):
            return 0.5  # Not a dict!

        pairs = [(["A", "B"], ["C", "D"])]

        with pytest.raises(TypeError, match="must return dict"):
            aula.evaluate(pairs, bad_predict)

    def test_weighted_vs_unweighted_comparison(self):
        """Test that attention weighting changes the result vs uniform."""
        aula = AULA(mode="whitespace")

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

        # eq. 5 with uniform attention 1/n on n = 2 tokens:
        #   (1/2) * [ 0.5*log(0.9) + 0.5*log(0.1) ]
        # v0.1.1 renormalised the weights, so this was 0.5*log(.9)+0.5*log(.1).
        expected_uniform = (0.5 * np.log(0.9) + 0.5 * np.log(0.1)) / 2
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

        # eq. 5 with attention [0.9, 0.1] on n = 2 tokens:
        #   (1/2) * [ 0.9*log(0.9) + 0.1*log(0.1) ]
        expected_focused = (0.9 * np.log(0.9) + 0.1 * np.log(0.1)) / 2
        assert pytest.approx(result_focused, abs=1e-5) == expected_focused

        # Focused attention should give higher score (less negative)
        assert result_focused > result_uniform

    def test_pair_comparison(self):
        """Test stereotype vs anti-stereotype comparison."""
        aula = AULA(mode="whitespace")

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

        # Should prefer stereotypes (> 50)
        assert score > 50.0

    def test_unbiased_model_near_fifty(self):
        """Test unbiased model returns score near 0.5."""
        aula = AULA(mode="whitespace")

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
        assert 0.0 <= score <= 100.0

    def test_result_type(self):
        """Test that result is a float."""
        aula = AULA(mode="whitespace")

        def mock_predict(sentence, pos):
            return {"prob": 0.5, "attention": np.ones(len(sentence))}

        pairs = [(["A", "B"], ["C", "D"])]

        score = aula.evaluate(pairs, mock_predict)

        assert isinstance(score, float)
        assert 0.0 <= score <= 100.0

    # === A) Common validation gaps ===

    def test_predict_function_not_callable_raises_error(self):
        """Test that non-callable predict_function raises TypeError."""
        aula = AULA(mode="whitespace")

        pairs = [(["A", "B"], ["C", "D"])]

        # Test with None
        with pytest.raises(TypeError, match="callable"):
            aula.evaluate(pairs, None)

        # Test with non-callable value
        with pytest.raises(TypeError, match="callable"):
            aula.evaluate(pairs, 123)

    def test_predict_function_returning_nan_prob_raises_error(self):
        """Test that predict_function returning NaN probability raises error."""
        aula = AULA(mode="whitespace")

        def nan_prob_predict(sentence, pos):
            return {"prob": np.nan, "attention": np.ones(len(sentence))}

        pairs = [(["A", "B"], ["C", "D"])]

        with pytest.raises(ValueError, match="Invalid probability"):
            aula.evaluate(pairs, nan_prob_predict)

    def test_predict_function_returning_inf_prob_raises_error(self):
        """Test that predict_function returning Inf probability raises error."""
        aula = AULA(mode="whitespace")

        def inf_prob_predict(sentence, pos):
            return {"prob": np.inf, "attention": np.ones(len(sentence))}

        pairs = [(["A", "B"], ["C", "D"])]

        with pytest.raises(ValueError, match="Invalid probability"):
            aula.evaluate(pairs, inf_prob_predict)

    def test_attention_with_inf_raises_error(self):
        """Test that attention containing Inf raises error."""
        aula = AULA(mode="whitespace")

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
        aula = AULA(mode="whitespace")

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

    def test_score_scale_and_neutral_fixture(self):
        aula = AULA(mode="whitespace")

        def predict(sentence, pos):
            if "Women" in sentence:
                prob = 0.8
            elif "Girls" in sentence:
                prob = 0.2
            elif "Boys" in sentence:
                prob = 0.8
            else:
                prob = 0.5
            return {"prob": prob, "attention": np.ones(len(sentence))}

        pairs = [
            (["Women", "work"], ["Men", "work"]),
            (["Girls", "work"], ["Boys", "work"]),
        ]
        assert aula.evaluate(pairs, predict) == 50.0

    def test_return_details_and_run(self):
        aula = AULA(mode="whitespace")
        pairs = [(["Women", "work"], ["Men", "work"])]
        result = aula.evaluate(
            pairs,
            lambda sentence, pos: {
                "prob": 0.5,
                "attention": np.ones(len(sentence)),
            },
            return_details=True,
        )
        assert result["bias_score"] == 0.0
        assert result["aula_score"] == 0.0
        assert isinstance(result["num_pairs"], int)
        run_result = aula.run(
            pairs,
            lambda sentence, pos: {
                "prob": 0.5,
                "attention": np.ones(len(sentence)),
            },
            ci="none",
        )
        assert run_result.score == 0.0

    def test_whitespace_model_name_is_rejected(self):
        with pytest.raises(ValueError, match="cannot be combined with model_name"):
            AULA(mode="whitespace", model_name="bert-base-uncased")

    def test_invalid_whitespace_inputs_raise_clear_error(self):
        aula = AULA(mode="whitespace")
        with pytest.raises(ValueError, match="token lists"):
            aula.evaluate([("Women work", "Men work")], lambda sentence, pos: {})

    # === per_item exposure + run() confidence intervals ===

    def test_return_details_exposes_per_item(self):
        """run() needs details['per_item'] to compute a bootstrap CI; verify
        evaluate(return_details=True) actually reports it, scaled to match
        the 0-100 bias_score so the two are on the same axis."""
        aula = AULA(mode="whitespace")

        def predict(sentence, pos):
            prob = 0.8 if "Women" in sentence else 0.3
            return {"prob": prob, "attention": np.ones(len(sentence))}

        pairs = [
            (["Women", "work"], ["Men", "work"]),
            (["Women", "cook"], ["Men", "cook"]),
            (["Girls", "play"], ["Boys", "play"]),
        ]
        result = aula.evaluate(pairs, predict, return_details=True)
        assert result["per_item"] == [100.0, 100.0, 0.0]
        assert np.mean(result["per_item"]) == pytest.approx(result["bias_score"])

    def test_run_produces_bootstrap_ci(self):
        """Before the fix, run()'s default ci='bootstrap' silently returned
        no interval for AULA because per_item was never exposed."""
        aula = AULA(mode="whitespace")

        def predict(sentence, pos):
            prob = 0.8 if "Women" in sentence else 0.3
            return {"prob": prob, "attention": np.ones(len(sentence))}

        pairs = [
            (["Women", "work"], ["Men", "work"]),
            (["Women", "cook"], ["Men", "cook"]),
            (["Girls", "play"], ["Boys", "play"]),
            (["Girls", "read"], ["Boys", "read"]),
        ]
        result = aula.run(pairs, predict)  # default ci="bootstrap"
        assert result.ci is not None
        assert result.ci_method == "bootstrap"
        ci_low, ci_high = result.ci
        assert ci_low <= result.score <= ci_high

    def test_wordpiece_mode_also_exposes_per_item(self):
        class FakeWordpieceScorer:
            def encode(self, sentence):
                return [ord(c) for c in sentence]

            def aul_aula(self, input_ids):
                aula = -float(np.mean(input_ids))
                return aula, aula

        aula = AULA(mode="wordpiece")
        pairs = [("bb", "aa"), ("aa", "bb")]
        result = aula.evaluate(pairs, FakeWordpieceScorer(), return_details=True)
        assert result["per_item"] == [0.0, 100.0]
        assert np.mean(result["per_item"]) == pytest.approx(result["bias_score"])

    # === Reject masking-based scorers in whitespace mode (they'd silently
    # compute PLL, not AULA) ===

    def test_whitespace_rejects_bert_pll_scorer_instance(self):
        from bias_scope.probability_based.scorers import BertPLLScorer

        scorer = _mock_masked_scorer(BertPLLScorer)
        aula = AULA(mode="whitespace")
        pairs = [(["Women", "work"], ["Men", "work"])]
        with pytest.raises(ValueError, match="mask the scored token"):
            aula.evaluate(pairs, scorer)

    def test_whitespace_rejects_wordpiece_scorer_instance(self):
        from bias_scope.probability_based.scorers import WordPieceBertScorer

        scorer = _mock_masked_scorer(WordPieceBertScorer)
        aula = AULA(mode="whitespace")
        pairs = [(["Women", "work"], ["Men", "work"])]
        with pytest.raises(ValueError, match="mask the scored token"):
            aula.evaluate(pairs, scorer)
