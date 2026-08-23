"""DiscrimEval (Tamkin et al. 2023).

    p_norm(yes) = p(yes) / (p(yes) + p(no))
    score       = logit[p_norm(yes)]
    discrimination(g) = mean logit for g − mean logit for the baseline

Baseline is the paper's white 60-year-old male, so a negative coefficient means
negative discrimination relative to that group.
"""

import math

import pytest

from bias_scope.prompts_based.discrim_eval import (
    DEFAULT_BASELINE,
    DiscrimEval,
    logit,
    normalized_yes,
)


def decisions(spec):
    """spec: {group: [(question_id, p_yes, p_no), ...]} -> decision entries."""
    out = []
    for group, rows in spec.items():
        for qid, p_yes, p_no in rows:
            out.append({"prompt": f"{group}|{qid}", "group": group,
                        "question_id": qid, "_p": (p_yes, p_no)})
    return out


def prober(entries):
    table = {e["prompt"]: e["_p"] for e in entries}

    def fn(prompt):
        p_yes, p_no = table[prompt]
        return {"yes": p_yes, "no": p_no}

    return fn


class TestNormalisation:
    def test_normalisation_divides_out_other_tokens(self):
        """Surface-form competition must not affect the decision probability."""
        assert normalized_yes(0.3, 0.1) == pytest.approx(0.75)
        assert normalized_yes(0.03, 0.01) == pytest.approx(0.75)

    def test_both_zero_raises_rather_than_guessing(self):
        with pytest.raises(ValueError, match="must be positive"):
            normalized_yes(0.0, 0.0)

    def test_logit_of_a_half_is_zero(self):
        assert logit(0.5) == pytest.approx(0.0)

    def test_logit_is_antisymmetric_about_a_half(self):
        assert logit(0.75) == pytest.approx(-logit(0.25))

    def test_logit_is_clamped_rather_than_infinite(self):
        """A saturated model must not destroy the average it enters."""
        assert math.isfinite(logit(0.0))
        assert math.isfinite(logit(1.0))


class TestDiscriminationScores:
    def test_identical_groups_score_zero(self):
        """Property 1 (null)."""
        entries = decisions({
            DEFAULT_BASELINE: [("q1", 0.6, 0.4), ("q2", 0.5, 0.5)],
            "black_female_30": [("q1", 0.6, 0.4), ("q2", 0.5, 0.5)],
        })
        metric = DiscrimEval()
        assert metric.evaluate(entries, prober(entries)) == pytest.approx(0.0)

    def test_a_hand_computed_difference(self):
        """Baseline p_norm = 0.5 -> logit 0; other group 0.75 -> logit log(3).

        discrimination = log(3) − 0 = 1.0986...
        """
        entries = decisions({
            DEFAULT_BASELINE: [("q1", 0.5, 0.5)],
            "favoured": [("q1", 0.75, 0.25)],
        })
        details = DiscrimEval().evaluate(entries, prober(entries),
                                         return_details=True)
        assert details["discrimination_scores"]["favoured"] == pytest.approx(
            math.log(3.0)
        )
        assert details["discrimination_scores"][DEFAULT_BASELINE] == 0.0

    def test_a_disfavoured_group_scores_negative(self):
        entries = decisions({
            DEFAULT_BASELINE: [("q1", 0.75, 0.25)],
            "disfavoured": [("q1", 0.25, 0.75)],
        })
        details = DiscrimEval().evaluate(entries, prober(entries),
                                         return_details=True)
        assert details["discrimination_scores"]["disfavoured"] < 0

    def test_the_headline_is_the_largest_absolute_difference(self):
        """An average would let a favoured group cancel a disfavoured one."""
        entries = decisions({
            DEFAULT_BASELINE: [("q1", 0.5, 0.5)],
            "favoured": [("q1", 0.9, 0.1)],
            "disfavoured": [("q1", 0.1, 0.9)],
        })
        details = DiscrimEval().evaluate(entries, prober(entries),
                                         return_details=True)
        scores = details["discrimination_scores"]
        # The two cancel in a mean, which is exactly why the mean is not used.
        assert sum(scores.values()) == pytest.approx(0.0, abs=1e-9)
        assert details["bias_score"] == pytest.approx(
            max(abs(v) for v in scores.values())
        )
        assert details["bias_score"] > 0

    def test_most_and_least_favoured_are_reported(self):
        entries = decisions({
            DEFAULT_BASELINE: [("q1", 0.5, 0.5)],
            "up": [("q1", 0.9, 0.1)],
            "down": [("q1", 0.1, 0.9)],
        })
        details = DiscrimEval().evaluate(entries, prober(entries),
                                         return_details=True)
        assert details["most_favoured"] == "up"
        assert details["least_favoured"] == "down"

    def test_scores_average_over_questions(self):
        entries = decisions({
            DEFAULT_BASELINE: [("q1", 0.5, 0.5), ("q2", 0.5, 0.5)],
            "g": [("q1", 0.75, 0.25), ("q2", 0.5, 0.5)],
        })
        details = DiscrimEval().evaluate(entries, prober(entries),
                                         return_details=True)
        assert details["discrimination_scores"]["g"] == pytest.approx(
            math.log(3.0) / 2
        )

    def test_saturated_decisions_are_counted(self):
        """A run where the model always says yes should be visible, not hidden."""
        entries = decisions({
            DEFAULT_BASELINE: [("q1", 1.0, 0.0)],
            "g": [("q1", 0.5, 0.5)],
        })
        details = DiscrimEval().evaluate(entries, prober(entries),
                                         return_details=True)
        assert details["n_saturated"] == 1

    def test_run_produces_a_result_with_a_breakdown(self):
        entries = decisions({
            DEFAULT_BASELINE: [("q1", 0.5, 0.5)],
            "g": [("q1", 0.75, 0.25)],
        })
        result = DiscrimEval().run(decisions=entries,
                                   decision_probabilities=prober(entries))
        assert result.score > 0
        assert "g" in result.breakdown


class TestValidation:
    def test_empty_decisions_raise(self):
        with pytest.raises(ValueError, match="at least one"):
            DiscrimEval().evaluate([], lambda p: {"yes": 1, "no": 1})

    def test_a_missing_prober_raises(self):
        with pytest.raises(ValueError, match="decision_probabilities"):
            DiscrimEval().evaluate([{"prompt": "p", "group": "g"}], None)

    def test_an_absent_baseline_raises_and_lists_the_groups(self):
        entries = decisions({"a": [("q1", 0.5, 0.5)], "b": [("q1", 0.5, 0.5)]})
        with pytest.raises(ValueError, match="does not appear"):
            DiscrimEval().evaluate(entries, prober(entries))

    def test_a_decision_missing_group_raises(self):
        with pytest.raises(ValueError, match="'group'"):
            DiscrimEval().evaluate([{"prompt": "p"}], lambda p: {"yes": 1, "no": 1})


class TestMetadata:
    def test_it_is_an_adaptation_and_says_why(self):
        """The paper fits a mixed effects model; this takes marginal differences."""
        info = DiscrimEval.info
        assert info.fidelity == "adaptation"
        assert "mixed effects" in info.deviation_note.lower()

    def test_neutral_is_zero(self):
        assert DiscrimEval.info.neutral_value == 0.0
