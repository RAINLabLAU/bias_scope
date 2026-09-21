"""LPBS per Kurita et al. 2019, and the original metric it displaced.

Known-answer cases are hand-derived from the paper's §2 procedure, with the
arithmetic written out in the test body (PLAN.md Section 1).

Definition being tested:

    p_tgt(t)   = P([MASK] = t | template with TARGET masked, ATTRIBUTE filled)
    p_prior(t) = P([MASK] = t | template with BOTH masked, read at the TARGET slot)
    LPBS(t1, t2) = [log p_tgt(t1) − log p_prior(t1)] − [log p_tgt(t2) − log p_prior(t2)]

With target *sets*, probabilities are summed within a set before the log, as the
authors' code does (`lib/bias_calculator.py:51-65`).
"""

import math

import pytest

from bias_scope.probability_based import LPBS, PairwiseLikelihoodPreference

TEMPLATE = "[TARGET] is a [ATTRIBUTE]."


def make_filler(table):
    """Build a fill_probabilities callable from a {(sentence, ordinal): {word: p}} table."""

    def fill_probabilities(sentence, candidates, mask_ordinal=0):
        probs = table[(sentence, mask_ordinal)]
        return {w: probs[w] for w in candidates}

    return fill_probabilities


class TestLPBSKnownAnswers:
    def test_hand_computed_single_template(self):
        """A case where every number is chosen so the answer is exact.

        Target sentence  "[MASK] is a programmer."   p(he)=0.4, p(she)=0.1
        Prior sentence   "[MASK] is a [MASK]."       p(he)=0.2, p(she)=0.2

          increased log prob (he)  = log(0.4) − log(0.2) = log 2
          increased log prob (she) = log(0.1) − log(0.2) = log 0.5 = −log 2
          LPBS = log 2 − (−log 2) = 2·log 2 ≈ 1.3862943611
        """
        table = {
            ("[MASK] is a programmer.", 0): {"he": 0.4, "she": 0.1},
            ("[MASK] is a [MASK].", 0): {"he": 0.2, "she": 0.2},
        }
        score = LPBS().evaluate(
            templates=[TEMPLATE],
            target_a=["he"],
            target_b=["she"],
            attributes=["programmer"],
            fill_probabilities=make_filler(table),
        )
        assert score == pytest.approx(2 * math.log(2))

    def test_no_association_gives_exactly_zero(self):
        """When the target and prior ratios agree, the prior cancels them out.

        This is the metric's whole point: p_tgt alone would report a 4:1
        preference for "he", but the model prefers "he" 4:1 in the prior too, so
        the attribute contributes nothing.
        """
        table = {
            ("[MASK] is a programmer.", 0): {"he": 0.4, "she": 0.1},
            ("[MASK] is a [MASK].", 0): {"he": 0.8, "she": 0.2},
        }
        score = LPBS().evaluate(
            templates=[TEMPLATE], target_a=["he"], target_b=["she"],
            attributes=["programmer"], fill_probabilities=make_filler(table),
        )
        assert score == pytest.approx(0.0, abs=1e-12)

    def test_target_sets_sum_before_the_log(self):
        """Kurita's code sums probabilities within a target set, then logs.

        target_a = {he, him}: p_tgt = 0.3 + 0.1 = 0.4, p_prior = 0.1 + 0.1 = 0.2
        target_b = {she, her}: p_tgt = 0.05 + 0.05 = 0.1, p_prior = 0.1 + 0.1 = 0.2
        Same numbers as the first case, so the same answer: 2·log 2.
        """
        table = {
            ("[MASK] is a programmer.", 0):
                {"he": 0.3, "him": 0.1, "she": 0.05, "her": 0.05},
            ("[MASK] is a [MASK].", 0):
                {"he": 0.1, "him": 0.1, "she": 0.1, "her": 0.1},
        }
        score = LPBS().evaluate(
            templates=[TEMPLATE], target_a=["he", "him"], target_b=["she", "her"],
            attributes=["programmer"], fill_probabilities=make_filler(table),
        )
        assert score == pytest.approx(2 * math.log(2))

    def test_prior_is_read_at_the_target_slot(self):
        """§2 step 3 reads the prior at the TARGET slot, mask ordinal 0.

        The filler returns different values at ordinal 0 and 1; a metric reading
        the wrong slot gets a different answer. This pins the choice discussed
        in docs/fidelity/lpbs.md and REVIEW_LATER RL-012.
        """
        table = {
            ("[MASK] is a programmer.", 0): {"he": 0.4, "she": 0.1},
            ("[MASK] is a [MASK].", 0): {"he": 0.2, "she": 0.2},   # target slot
            ("[MASK] is a [MASK].", 1): {"he": 0.9, "she": 0.01},  # attribute slot
        }
        score = LPBS().evaluate(
            templates=[TEMPLATE], target_a=["he"], target_b=["she"],
            attributes=["programmer"], fill_probabilities=make_filler(table),
        )
        assert score == pytest.approx(2 * math.log(2))


class TestLPBSProperties:
    def _table(self):
        return {
            ("[MASK] is a programmer.", 0): {"he": 0.4, "she": 0.1},
            ("[MASK] is a nurse.", 0): {"he": 0.1, "she": 0.4},
            ("[MASK] is a [MASK].", 0): {"he": 0.2, "she": 0.2},
        }

    def test_swapping_targets_flips_the_sign(self):
        """Property 2 (swap antisymmetry): LPBS is exactly antisymmetric."""
        filler = make_filler(self._table())
        kwargs = dict(templates=[TEMPLATE], attributes=["programmer"],
                      fill_probabilities=filler)
        forward = LPBS().evaluate(target_a=["he"], target_b=["she"], **kwargs)
        backward = LPBS().evaluate(target_a=["she"], target_b=["he"], **kwargs)
        assert backward == pytest.approx(-forward)

    def test_identical_targets_give_zero(self):
        """Property 1 (null): a target compared with itself has no association."""
        filler = make_filler(self._table())
        score = LPBS().evaluate(
            templates=[TEMPLATE], target_a=["he"], target_b=["he"],
            attributes=["programmer"], fill_probabilities=filler,
        )
        assert score == pytest.approx(0.0, abs=1e-12)

    def test_attribute_order_does_not_change_the_score(self):
        """Property 4 (permutation invariance) over attributes."""
        filler = make_filler(self._table())
        kwargs = dict(templates=[TEMPLATE], target_a=["he"], target_b=["she"],
                      fill_probabilities=filler)
        forward = LPBS().evaluate(attributes=["programmer", "nurse"], **kwargs)
        reverse = LPBS().evaluate(attributes=["nurse", "programmer"], **kwargs)
        assert forward == pytest.approx(reverse)

    def test_opposing_attributes_cancel(self):
        """"programmer" favours he by exactly what "nurse" favours she."""
        filler = make_filler(self._table())
        score = LPBS().evaluate(
            templates=[TEMPLATE], target_a=["he"], target_b=["she"],
            attributes=["programmer", "nurse"], fill_probabilities=filler,
        )
        assert score == pytest.approx(0.0, abs=1e-12)


class TestLPBSDetails:
    def test_details_expose_per_item_scores(self):
        table = {
            ("[MASK] is a programmer.", 0): {"he": 0.4, "she": 0.1},
            ("[MASK] is a nurse.", 0): {"he": 0.1, "she": 0.4},
            ("[MASK] is a [MASK].", 0): {"he": 0.2, "she": 0.2},
        }
        details = LPBS().evaluate(
            templates=[TEMPLATE], target_a=["he"], target_b=["she"],
            attributes=["programmer", "nurse"],
            fill_probabilities=make_filler(table), return_details=True,
        )
        assert details["per_item"] == pytest.approx(
            [2 * math.log(2), -2 * math.log(2)]
        )
        assert details["n"] == 2
        assert details["bias_score"] == pytest.approx(0.0, abs=1e-12)

    def test_per_attribute_breakdown_is_reported(self):
        table = {
            ("[MASK] is a programmer.", 0): {"he": 0.4, "she": 0.1},
            ("[MASK] is a [MASK].", 0): {"he": 0.2, "she": 0.2},
        }
        details = LPBS().evaluate(
            templates=[TEMPLATE], target_a=["he"], target_b=["she"],
            attributes=["programmer"], fill_probabilities=make_filler(table),
            return_details=True,
        )
        assert details["breakdown"]["programmer"] == pytest.approx(2 * math.log(2))
        assert details["increased_log_prob_a"][0] == pytest.approx(math.log(2))
        assert details["increased_log_prob_b"][0] == pytest.approx(-math.log(2))

    def test_run_returns_a_signed_unbounded_result(self):
        table = {
            ("[MASK] is a programmer.", 0): {"he": 0.4, "she": 0.1},
            ("[MASK] is a [MASK].", 0): {"he": 0.2, "she": 0.2},
        }
        result = LPBS().run(
            templates=[TEMPLATE], target_a=["he"], target_b=["she"],
            attributes=["programmer"], fill_probabilities=make_filler(table),
        )
        assert result.score == pytest.approx(2 * math.log(2))
        assert result.n == 1
        assert result.info.neutral_value == 0.0
        assert result.info.direction == "signed"


class TestLPBSValidation:
    def _filler(self):
        return make_filler({
            ("[MASK] is a programmer.", 0): {"he": 0.4, "she": 0.1},
            ("[MASK] is a [MASK].", 0): {"he": 0.2, "she": 0.2},
        })

    def test_a_template_missing_target_raises(self):
        with pytest.raises(ValueError, match=r"exactly one \[TARGET\]"):
            LPBS().evaluate(templates=["A [ATTRIBUTE] works."], target_a=["he"],
                            target_b=["she"], attributes=["programmer"],
                            fill_probabilities=self._filler())

    def test_a_template_missing_attribute_raises(self):
        with pytest.raises(ValueError, match=r"exactly one \[ATTRIBUTE\]"):
            LPBS().evaluate(templates=["[TARGET] works."], target_a=["he"],
                            target_b=["she"], attributes=["programmer"],
                            fill_probabilities=self._filler())

    def test_a_template_with_duplicate_target_raises(self):
        with pytest.raises(ValueError, match=r"exactly one \[TARGET\].*found 2"):
            LPBS().evaluate(
                templates=["[TARGET] told [TARGET] about [ATTRIBUTE]."],
                target_a=["he"], target_b=["she"], attributes=["programmer"],
                fill_probabilities=self._filler(),
            )

    def test_a_template_with_duplicate_attribute_raises(self):
        with pytest.raises(ValueError, match=r"exactly one \[ATTRIBUTE\].*found 2"):
            LPBS().evaluate(
                templates=["[TARGET] is a [ATTRIBUTE] and [ATTRIBUTE]."],
                target_a=["he"], target_b=["she"], attributes=["programmer"],
                fill_probabilities=self._filler(),
            )

    def test_empty_templates_raises(self):
        with pytest.raises(ValueError, match="templates"):
            LPBS().evaluate(templates=[], target_a=["he"], target_b=["she"],
                            attributes=["programmer"], fill_probabilities=self._filler())

    def test_empty_target_set_raises(self):
        with pytest.raises(ValueError, match="target_a"):
            LPBS().evaluate(templates=[TEMPLATE], target_a=[], target_b=["she"],
                            attributes=["programmer"], fill_probabilities=self._filler())

    def test_empty_attributes_raises(self):
        with pytest.raises(ValueError, match="attributes"):
            LPBS().evaluate(templates=[TEMPLATE], target_a=["he"], target_b=["she"],
                            attributes=[], fill_probabilities=self._filler())

    def test_a_zero_probability_raises_rather_than_returning_minus_inf(self):
        table = {
            ("[MASK] is a programmer.", 0): {"he": 0.0, "she": 0.1},
            ("[MASK] is a [MASK].", 0): {"he": 0.2, "she": 0.2},
        }
        with pytest.raises(ValueError, match="positive"):
            LPBS().evaluate(templates=[TEMPLATE], target_a=["he"], target_b=["she"],
                            attributes=["programmer"],
                            fill_probabilities=make_filler(table))

    def test_a_non_callable_filler_raises(self):
        with pytest.raises(ValueError, match="fill_probabilities"):
            LPBS().evaluate(templates=[TEMPLATE], target_a=["he"], target_b=["she"],
                            attributes=["programmer"], fill_probabilities="nope")

    def test_multi_word_target_raises(self):
        with pytest.raises(ValueError, match="multi-word"):
            LPBS().evaluate(
                templates=[TEMPLATE], target_a=["New York"], target_b=["Paris"],
                attributes=["home"], fill_probabilities=self._filler(),
            )

    def test_multi_word_attribute_raises(self):
        with pytest.raises(ValueError, match="multi-word"):
            LPBS().evaluate(
                templates=[TEMPLATE], target_a=["he"], target_b=["she"],
                attributes=["software engineer"], fill_probabilities=self._filler(),
            )

    def test_continuation_wordpiece_target_raises(self):
        with pytest.raises(ValueError, match="continuation wordpiece"):
            LPBS().evaluate(
                templates=[TEMPLATE], target_a=["##er"], target_b=["she"],
                attributes=["programmer"], fill_probabilities=self._filler(),
            )


class TestPairwiseLikelihoodPreference:
    """The v0.1.1 statistic, preserved under its own name (PLAN.md Section 12)."""

    def test_it_still_computes_the_old_win_rate(self):
        pairs = [(["a"], ["b"]), (["c"], ["d"])]
        scores = {("a",): 1.0, ("b",): 0.0, ("c",): 0.0, ("d",): 1.0}
        metric = PairwiseLikelihoodPreference()
        score = metric.evaluate(
            sentence_pairs=pairs, logprob_fn=lambda s: scores[tuple(s)]
        )
        assert score == pytest.approx(0.5)

    def test_it_is_registered_as_original_not_as_kuritas_metric(self):
        info = PairwiseLikelihoodPreference.info
        assert info.fidelity == "original"
        assert info.neutral_value == 0.5
        assert "Kurita" not in info.reference

    def test_lpbs_no_longer_accepts_the_old_signature(self):
        """The old behaviour is not available under the LPBS name (Section 1)."""
        with pytest.raises(TypeError):
            LPBS().evaluate(sentence_pairs=[(["a"], ["b"])], logprob_fn=lambda s: 1.0)
