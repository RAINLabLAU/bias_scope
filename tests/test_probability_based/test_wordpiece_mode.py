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
        crows = CrowSPairs(mode="wordpiece", percentage=False)
        scorer = FakeWordPieceScorer()
        score = crows.evaluate(PAIRS, predict_masked_token=scorer)
        assert score == 1.0  # every stereo side wins

    def test_return_details_includes_mode(self):
        crows = CrowSPairs(mode="wordpiece", percentage=False)
        scorer = FakeWordPieceScorer()
        result = crows.evaluate(PAIRS, predict_masked_token=scorer, return_details=True)
        assert result["mode"] == "wordpiece"
        assert result["crows_pairs_score"] == 1.0
        assert result["num_pairs"] == float(len(PAIRS))

    def test_requires_string_pairs_in_wordpiece_mode(self):
        crows = CrowSPairs(mode="wordpiece", percentage=False)
        scorer = FakeWordPieceScorer()
        bad_pairs = [(["tokens", "not", "strings"], ["also", "not", "strings"])]
        with pytest.raises(ValueError):
            crows.evaluate(bad_pairs, predict_masked_token=scorer)

    def test_raises_without_scorer(self):
        crows = CrowSPairs(mode="wordpiece", percentage=False)
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
        assert CrowSPairs(mode="whitespace", percentage=False).mode == "whitespace"


class TestAULWordpieceMode:
    def test_dispatches_to_wordpiece_scorer(self):
        aul = AUL(mode="wordpiece", percentage=False)
        scorer = FakeWordPieceScorer()
        score = aul.evaluate(PAIRS, predict_token_given_sentence=scorer)
        assert score == 1.0

    def test_return_details_shape(self):
        aul = AUL(mode="wordpiece", percentage=False)
        scorer = FakeWordPieceScorer()
        result = aul.evaluate(
            PAIRS, predict_token_given_sentence=scorer, return_details=True
        )
        assert result["mode"] == "wordpiece"
        assert result["aul_score"] == 1.0

    def test_requires_scorer_or_model_name(self):
        aul = AUL(mode="wordpiece", percentage=False)
        with pytest.raises(TypeError):
            aul.evaluate(PAIRS)


class TestAULAWordpieceMode:
    def test_dispatches_to_wordpiece_scorer(self):
        aula = AULA(mode="wordpiece", percentage=False)
        scorer = FakeWordPieceScorer()
        score = aula.evaluate(PAIRS, predict_with_attention=scorer)
        assert score == 1.0

    def test_return_details_shape(self):
        aula = AULA(mode="wordpiece", percentage=False)
        scorer = FakeWordPieceScorer()
        result = aula.evaluate(
            PAIRS, predict_with_attention=scorer, return_details=True
        )
        assert result["mode"] == "wordpiece"
        assert result["aula_score"] == 1.0

    def test_requires_scorer_or_model_name(self):
        aula = AULA(mode="wordpiece", percentage=False)
        with pytest.raises(TypeError):
            aula.evaluate(PAIRS)


class TestPercentageScale:
    """These three report a percentage, matching their papers and MetricInfo.

    Nangia et al. report 60.5 for BERT, not 0.605; `MetricInfo` declares
    `neutral_value=50.0, value_range=(0.0, 100.0)`; and
    `results/emnlp/crows_pairs.json` records our reproduction as 58.62. Until
    0.2.0 `evaluate()` alone returned a fraction, which put every default run
    on the wrong side of neutral:

        0.6667 (fraction)   normalized_deviation = -0.99   <- reads unbiased-ish
        66.67  (percent)    normalized_deviation = +0.33   <- correct

    A model preferring the stereotyping sentence in 4 of 6 pairs plotted as
    *anti*-stereotypical in every profile and dumbbell figure. See
    REVIEW_LATER RL-038.
    """

    @staticmethod
    def _four_of_six(cls):
        """Four stereotyping sentences preferred out of six pairs.

        The pair shares every token but one. CrowS-Pairs scores the
        *unmodified* tokens conditioned on the modified one, so the shared
        tokens must be the thing whose likelihood differs — which is exactly
        what the metric measures. AULA takes a different callback that returns
        a probability and an attention weight per position.
        """
        pairs = [(["the", "person", "was", f"m{i}"],
                  ["the", "person", "was", f"w{i}"]) for i in range(6)]

        def probability(tokens, index):
            marker = next(t for t in tokens
                          if t[:1] in ("m", "w") and t[1:].isdigit())
            stereotype_should_win = int(marker[1:]) < 4
            on_stereotype_side = marker.startswith("m")
            favoured = on_stereotype_side == stereotype_should_win
            return 0.9 if favoured else 0.1

        if cls is AULA:
            def with_attention(tokens, index):
                # Uniform attention: AULA's weighting then reduces to AUL's
                # plain mean, so the two must agree on this fixture.
                return {"prob": probability(tokens, index),
                        "attention": [1.0 / len(tokens)] * len(tokens)}
            return pairs, with_attention
        return pairs, probability

    @pytest.mark.parametrize("cls", [CrowSPairs, AUL, AULA])
    def test_a_two_thirds_preference_scores_sixty_six_not_zero_point_six(self, cls):
        pairs, predict = self._four_of_six(cls)
        score = cls(mode="whitespace").evaluate(pairs, predict)
        assert score == pytest.approx(100 * 4 / 6, abs=1e-9)

    @pytest.mark.parametrize("cls", [CrowSPairs, AUL, AULA])
    def test_the_scale_matches_the_declared_value_range(self, cls):
        """A metric whose score scale disagrees with its own MetricInfo puts
        `normalized_deviation` — and every figure — on the wrong side of 0."""
        from bias_scope.metadata import normalized_deviation

        pairs, predict = self._four_of_six(cls)
        score = cls(mode="whitespace").evaluate(pairs, predict)
        low, high = cls.info.value_range
        assert low <= score <= high
        assert score > cls.info.neutral_value
        assert normalized_deviation(score, cls.info) > 0

    def test_run_refuses_the_fraction_scale(self):
        """A docstring warning is not a guard.

        `percentage=False` is a v0.1.x compatibility path for `evaluate()`. Fed
        to `run()` it would produce a score near 0 against a declared neutral of
        50, which is exactly the sign inversion this flag exists to avoid. So
        `run()` refuses it by name rather than returning a wrong number.
        """
        from bias_scope.base import BiasScopeError

        pairs, predict = self._four_of_six(CrowSPairs)
        with pytest.raises(BiasScopeError, match="percentage=False"):
            CrowSPairs(mode="whitespace", percentage=False).run(
                sentence_pairs=pairs, predict_masked_token=predict)

    def test_run_accepts_the_default_percentage_scale(self):
        pairs, predict = self._four_of_six(CrowSPairs)
        result = CrowSPairs(mode="whitespace").run(
            sentence_pairs=pairs, predict_masked_token=predict)
        assert result.score == pytest.approx(100 * 4 / 6, abs=1e-9)
        assert result.normalized_deviation() > 0
