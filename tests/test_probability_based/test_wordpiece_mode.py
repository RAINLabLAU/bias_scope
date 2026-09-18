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
    second — which lets the tests assert bias_score == 100.0 (percent, RL-060).
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
        assert score == 100.0  # every stereo side wins, as a percentage (RL-060)

    def test_return_details_includes_mode(self):
        crows = CrowSPairs(mode="wordpiece")
        scorer = FakeWordPieceScorer()
        result = crows.evaluate(PAIRS, predict_masked_token=scorer, return_details=True)
        assert result["mode"] == "wordpiece"
        assert result["crows_pairs_score"] == 100.0
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

    def test_default_mode_is_wordpiece_the_faithful_path(self):
        """The default must be the path the registered fidelity refers to.

        Through v0.1.x the default was `whitespace`, which scores whole
        whitespace words — not the authors' pseudo-log-likelihood over WordPiece
        tokens. The metric was registered `faithful` and its fidelity note said
        wordpiece "is the default" while the code said otherwise, so the default
        run was not the protocol the status claimed. See CHANGELOG 0.2.0
        (Breaking) and REVIEW_LATER RL-037.
        """
        assert CrowSPairs().mode == "wordpiece"

    def test_the_faithful_default_holds_for_the_whole_pll_family(self):
        """AUL and AULA had the same split, and PLAN.md 5.2 names all three."""
        from bias_scope.probability_based import AUL, AULA

        assert AUL().mode == "wordpiece"
        assert AULA().mode == "wordpiece"

    def test_the_whitespace_path_is_still_reachable(self):
        """Kept for continuity with v0.1.x; it is not CrowS-Pairs."""
        assert CrowSPairs(mode="whitespace").mode == "whitespace"


class TestAULWordpieceMode:
    def test_dispatches_to_wordpiece_scorer(self):
        aul = AUL(mode="wordpiece")
        scorer = FakeWordPieceScorer()
        score = aul.evaluate(PAIRS, predict_token_given_sentence=scorer)
        assert score == 100.0

    def test_return_details_shape(self):
        aul = AUL(mode="wordpiece")
        scorer = FakeWordPieceScorer()
        result = aul.evaluate(
            PAIRS, predict_token_given_sentence=scorer, return_details=True
        )
        assert result["mode"] == "wordpiece"
        assert result["aul_score"] == 100.0

    def test_requires_scorer_or_model_name(self):
        aul = AUL(mode="wordpiece")
        with pytest.raises(TypeError):
            aul.evaluate(PAIRS)


class TestAULAWordpieceMode:
    def test_dispatches_to_wordpiece_scorer(self):
        aula = AULA(mode="wordpiece")
        scorer = FakeWordPieceScorer()
        score = aula.evaluate(PAIRS, predict_with_attention=scorer)
        assert score == 100.0

    def test_return_details_shape(self):
        aula = AULA(mode="wordpiece")
        scorer = FakeWordPieceScorer()
        result = aula.evaluate(
            PAIRS, predict_with_attention=scorer, return_details=True
        )
        assert result["mode"] == "wordpiece"
        assert result["aula_score"] == 100.0

    def test_requires_scorer_or_model_name(self):
        aula = AULA(mode="wordpiece")
        with pytest.raises(TypeError):
            aula.evaluate(PAIRS)


class TestPairsMayBeListsNotOnlyTuples:
    """JSON has no tuple type, so every pair that reaches these metrics
    through a tool call — bias_scope_agent's `run_suite`, or any JSON API —
    arrives as a list. A hard `isinstance(pair, tuple)` check therefore makes
    wordpiece mode unreachable from the agent no matter what the caller does.
    A list of two strings is the same pair; a bare string is not
    (REVIEW_LATER RL-050).
    """

    def test_a_pair_given_as_a_list_is_accepted(self):
        scorer = FakeWordPieceScorer()
        pairs = [["STEREO sentence one", "plain sentence one"]]
        score = CrowSPairs(mode="wordpiece").evaluate(
            sentence_pairs=pairs, predict_masked_token=scorer
        )
        assert score == pytest.approx(100.0)

    def test_lists_and_tuples_give_the_same_score(self):
        as_tuples = [("STEREO a", "plain a"), ("STEREO b", "plain b")]
        as_lists = [list(pair) for pair in as_tuples]
        tuple_score = CrowSPairs(mode="wordpiece").evaluate(
            sentence_pairs=as_tuples, predict_masked_token=FakeWordPieceScorer()
        )
        list_score = CrowSPairs(mode="wordpiece").evaluate(
            sentence_pairs=as_lists, predict_masked_token=FakeWordPieceScorer()
        )
        assert tuple_score == pytest.approx(list_score)

    @pytest.mark.parametrize(
        "metric_cls,scorer_kwarg",
        [(AUL, "predict_token_given_sentence"), (AULA, "predict_with_attention")],
    )
    def test_aul_and_aula_accept_lists_too(self, metric_cls, scorer_kwarg):
        pairs = [["STEREO sentence", "plain sentence"]]
        score = metric_cls(mode="wordpiece").evaluate(
            sentence_pairs=pairs, **{scorer_kwarg: FakeWordPieceScorer()}
        )
        assert score == pytest.approx(100.0)

    def test_a_two_character_string_is_still_rejected(self):
        """"ab" has len 2 but is not a pair of sentences."""
        with pytest.raises(ValueError, match="sentence_pair"):
            CrowSPairs(mode="wordpiece").evaluate(
                sentence_pairs=["ab"], predict_masked_token=FakeWordPieceScorer()
            )

    def test_a_three_element_pair_is_still_rejected(self):
        with pytest.raises(ValueError, match="sentence_pair"):
            CrowSPairs(mode="wordpiece").evaluate(
                sentence_pairs=[["a", "b", "c"]],
                predict_masked_token=FakeWordPieceScorer(),
            )
