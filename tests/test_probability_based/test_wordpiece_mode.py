"""Interface tests for the wordpiece-mode paths of CrowSPairs / AUL / AULA.

These tests use a fake scorer that exposes the WordPieceBertScorer API
(encode / align_unmodified / pll_over_positions / aul_aula). No BERT
model is downloaded.
"""

from __future__ import annotations

from typing import List, Tuple

import pytest

from bias_scope.probability_based import AUL, AULA, CrowSPairs


class FakeWordPieceScorer:
    """Deterministic, mock-friendly stand-in for WordPieceBertScorer.

    Behaviour is chosen so that when the first sentence in a pair contains
    the substring "STEREO", its PLL/AUL/AULA are strictly greater than the
    second — which lets the tests assert bias_score == 1.0.
    """

    def __init__(self, stereo_marker: str = "STEREO"):
        self.stereo_marker = stereo_marker
        self._call_log: list[tuple[str, tuple]] = []

    def encode(self, sentence: str) -> List[int]:
        self._call_log.append(("encode", (sentence,)))
        # Encode as list of char codes — deterministic and comparable.
        return [ord(c) for c in sentence]

    def align_unmodified(
        self, ids_a: List[int], ids_b: List[int]
    ) -> Tuple[List[int], List[int]]:
        self._call_log.append(("align", (len(ids_a), len(ids_b))))
        # For the tests we use pairs of equal length; identical positions
        # are those where the char codes match.
        pos_a, pos_b = [], []
        for i, (a, b) in enumerate(zip(ids_a, ids_b)):
            if a == b:
                pos_a.append(i)
                pos_b.append(i)
        return pos_a, pos_b

    def pll_over_positions(
        self, input_ids: List[int], positions: List[int]
    ) -> float:
        self._call_log.append(("pll", (len(input_ids), len(positions))))
        s = "".join(chr(i) for i in input_ids)
        # Higher PLL for the stereotype side.
        return 1.0 if self.stereo_marker in s else 0.0

    def aul_aula(self, input_ids: List[int]) -> Tuple[float, float]:
        self._call_log.append(("aul_aula", (len(input_ids),)))
        s = "".join(chr(i) for i in input_ids)
        val = 1.0 if self.stereo_marker in s else 0.0
        return val, val


PAIRS = [
    ("STEREOtypical sentence one", "neutral sentence one         "),
    ("STEREOtypical sentence two", "neutral sentence two         "),
    ("STEREOtypical sentence x", "neutral sentence x         "),
]


class TestCrowSPairsWordpieceMode:
    def test_dispatches_to_wordpiece_scorer(self):
        crows = CrowSPairs(mode="wordpiece")
        scorer = FakeWordPieceScorer()
        score = crows.evaluate(PAIRS, predict_masked_token=scorer)
        assert score == 1.0  # every stereo side wins

    def test_return_details_includes_mode(self):
        crows = CrowSPairs(mode="wordpiece")
        scorer = FakeWordPieceScorer()
        result = crows.evaluate(PAIRS, predict_masked_token=scorer, return_details=True)
        assert result["mode"] == "wordpiece"
        assert result["crows_pairs_score"] == 1.0
        assert result["num_pairs"] == float(len(PAIRS))

    def test_requires_string_pairs_in_wordpiece_mode(self):
        crows = CrowSPairs(mode="wordpiece")
        scorer = FakeWordPieceScorer()
        bad_pairs = [(["tokens", "not", "strings"], ["also", "not", "strings"])]
        with pytest.raises(ValueError):
            crows.evaluate(bad_pairs, predict_masked_token=scorer)

    def test_raises_without_scorer(self):
        crows = CrowSPairs(mode="wordpiece")
        with pytest.raises(TypeError):
            crows.evaluate(PAIRS)  # no scorer, no model_name

    def test_invalid_mode_string(self):
        with pytest.raises(ValueError):
            CrowSPairs(mode="totally-bogus")

    def test_default_mode_is_whitespace(self):
        crows = CrowSPairs()
        assert crows.mode == "whitespace"


class TestAULWordpieceMode:
    def test_dispatches_to_wordpiece_scorer(self):
        aul = AUL(mode="wordpiece")
        scorer = FakeWordPieceScorer()
        score = aul.evaluate(PAIRS, predict_token_given_sentence=scorer)
        assert score == 1.0

    def test_return_details_shape(self):
        aul = AUL(mode="wordpiece")
        scorer = FakeWordPieceScorer()
        result = aul.evaluate(
            PAIRS, predict_token_given_sentence=scorer, return_details=True
        )
        assert result["mode"] == "wordpiece"
        assert result["aul_score"] == 1.0

    def test_requires_scorer_or_model_name(self):
        aul = AUL(mode="wordpiece")
        with pytest.raises(TypeError):
            aul.evaluate(PAIRS)


class TestAULAWordpieceMode:
    def test_dispatches_to_wordpiece_scorer(self):
        aula = AULA(mode="wordpiece")
        scorer = FakeWordPieceScorer()
        score = aula.evaluate(PAIRS, predict_with_attention=scorer)
        assert score == 1.0

    def test_return_details_shape(self):
        aula = AULA(mode="wordpiece")
        scorer = FakeWordPieceScorer()
        result = aula.evaluate(
            PAIRS, predict_with_attention=scorer, return_details=True
        )
        assert result["mode"] == "wordpiece"
        assert result["aula_score"] == 1.0

    def test_requires_scorer_or_model_name(self):
        aula = AULA(mode="wordpiece")
        with pytest.raises(TypeError):
            aula.evaluate(PAIRS)
