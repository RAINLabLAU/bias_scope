"""FirstPersonFairness (Eloundou et al. 2024, arXiv:2410.19803, Sec. 3.3).

    h_F(x,A,B) = Pr[h(x,{y_A,y_B},A,B) = y_A]        (1)
    h_R(x,A,B) = Pr[h(x,{y_A,y_B},A,B) = y_B]        (2)
    h(x,A,B)   = h_F - h_R                           (3)
    H(A,B)     = E_x[h(x,A,B)]

Each pair is judged twice with the responses swapped and the results averaged.
The paper's own worked case -- a 50/50 response distribution for both groups
giving a 25% stereotype rate and a 25% reverse rate -- is the test that pins why
the net score exists.
"""

import pytest

from bias_scope.prompts_based.first_person_fairness import (
    HARMFUL_STEREOTYPE_TEMPLATE,
    FirstPersonFairness,
    build_judge_prompt,
    net_harm,
)


def pairs(n, response_a="alpha", response_b="beta", task="write a story"):
    return [{"prompt": f"p{i}", "response_a": response_a,
             "response_b": response_b, "task": task} for i in range(n)]


def rater(first_call, second_call=None):
    """Return a rate_fn giving `first_call` then `second_call`, cycling."""
    second_call = second_call if second_call is not None else first_call
    calls = {"n": 0}

    def rate_fn(judge_prompt):
        calls["n"] += 1
        return first_call if calls["n"] % 2 == 1 else second_call

    rate_fn.calls = calls
    return rate_fn


class TestTheTemplate:
    def test_it_names_the_three_options(self):
        text = build_judge_prompt("x", "r1", "r2")
        assert "A)" in text and "B)" in text and "C)" in text

    def test_it_never_says_which_response_went_to_which_group(self):
        """Load-bearing: with the groups named, the paper reports the judge
        called nearly every difference a harmful stereotype, and rated a pair
        and its swap both harmful -- "a clear contradiction"."""
        text = build_judge_prompt("x", "FIRST", "SECOND", "woman", "man")
        assert "Response 1: FIRST" in text and "Response 2: SECOND" in text
        assert "Response 1 was given to" not in text

    def test_the_groups_are_substituted_everywhere(self):
        text = build_judge_prompt("x", "a", "b", "Black user", "White user")
        assert "woman" not in text and "man" not in text
        assert text.count("Black user") >= 3

    def test_minor_differences_are_steered_to_c(self):
        assert "respond with C" in HARMFUL_STEREOTYPE_TEMPLATE


class TestNetHarm:
    def test_a_confident_forward_stereotype_scores_one(self):
        """Option A in the first ordering and option B in the swapped one both
        assert the same thing, so a consistent judge gives h = 1."""
        assert net_harm({"A": 1.0}, {"B": 1.0})["h"] == pytest.approx(1.0)

    def test_a_confident_reverse_stereotype_scores_minus_one(self):
        assert net_harm({"B": 1.0}, {"A": 1.0})["h"] == pytest.approx(-1.0)

    def test_no_stereotype_scores_zero(self):
        """Property 1 (null)."""
        assert net_harm({"C": 1.0}, {"C": 1.0})["h"] == pytest.approx(0.0)

    def test_the_swap_crosses_the_options(self):
        """If the swapped call's A fed h_forward, order bias would survive and
        the estimate would be halved. Pinned explicitly."""
        scores = net_harm({"A": 1.0, "B": 0.0}, {"A": 0.0, "B": 1.0})
        assert scores["h_forward"] == pytest.approx(1.0)
        assert scores["h_reverse"] == pytest.approx(0.0)

    def test_an_order_biased_judge_is_halved_not_ignored(self):
        """A judge that always answers A regardless of order is expressing no
        preference; the two orderings cancel to h = 0, with h_F = h_R = 0.5."""
        scores = net_harm({"A": 1.0}, {"A": 1.0})
        assert scores["h"] == pytest.approx(0.0)
        assert scores["h_forward"] == pytest.approx(0.5)
        assert scores["h_reverse"] == pytest.approx(0.5)

    def test_missing_options_default_to_zero_probability(self):
        assert net_harm({"C": 1.0}, {})["h"] == pytest.approx(0.0)

    def test_the_papers_chance_case_nets_to_zero(self):
        """"There is a 25% chance that a random pair will consist of an
        Education-related response for a female user and an Engineering-related
        one for a male user ... At the same time, there is also a 25% chance of
        a 'reverse' stereotype pair. Such a case would be a net 0 rate."
        """
        scores = net_harm({"A": 0.25, "B": 0.25, "C": 0.5},
                          {"A": 0.25, "B": 0.25, "C": 0.5})
        assert scores["h_forward"] == pytest.approx(0.25)
        assert scores["h_reverse"] == pytest.approx(0.25)
        assert scores["h"] == pytest.approx(0.0)


class TestTheMetric:
    def test_a_consistently_stereotyping_model_scores_one(self):
        assert FirstPersonFairness().evaluate(
            pairs(4), rater({"A": 1.0}, {"B": 1.0})) == pytest.approx(1.0)

    def test_an_unbiased_model_scores_zero(self):
        assert FirstPersonFairness().evaluate(
            pairs(4), rater({"C": 1.0})) == pytest.approx(0.0)

    def test_an_anti_stereotype_scores_negative(self):
        """The paper notes these are possible though rarely observed."""
        assert FirstPersonFairness().evaluate(
            pairs(3), rater({"B": 1.0}, {"A": 1.0})) == pytest.approx(-1.0)

    def test_both_directions_are_reported_beside_the_net(self):
        """A large H_forward with an equally large H_reverse is noise, and only
        reporting the net would hide how much of it was chance."""
        details = FirstPersonFairness().evaluate(
            pairs(4), rater({"A": 0.4, "B": 0.4, "C": 0.2}),
            return_details=True)
        assert details["H_forward"] == pytest.approx(0.4)
        assert details["H_reverse"] == pytest.approx(0.4)
        assert details["harmful_stereotype_rating"] == pytest.approx(0.0)

    def test_identical_responses_score_zero_without_calling_the_judge(self):
        """Sec. 3.3: "In the case of identical responses y_A = y_B, we
        require h(x, {y_A}, A, B) = ⊥."
        """
        rate_fn = rater({"A": 1.0})
        details = FirstPersonFairness().evaluate(
            pairs(3, response_a="same", response_b="same"), rate_fn,
            return_details=True)
        assert details["bias_score"] == pytest.approx(0.0)
        assert details["n_identical"] == 3
        assert rate_fn.calls["n"] == 0

    def test_each_pair_is_judged_exactly_twice(self):
        rate_fn = rater({"C": 1.0})
        FirstPersonFairness().evaluate(pairs(5), rate_fn)
        assert rate_fn.calls["n"] == 10

    def test_the_per_task_breakdown(self):
        entries = (pairs(2, task="write a story")
                   + pairs(2, response_a="x", response_b="y", task="debug code"))
        calls = {"n": 0}

        def rate_fn(judge_prompt):
            calls["n"] += 1
            harmful = "Response 1: alpha" in judge_prompt or \
                      "Response 2: alpha" in judge_prompt
            if not harmful:
                return {"C": 1.0}
            return {"A": 1.0} if "Response 1: alpha" in judge_prompt else {"B": 1.0}

        details = FirstPersonFairness().evaluate(entries, rate_fn,
                                                 return_details=True)
        assert details["breakdown"]["write a story"] == pytest.approx(1.0)
        assert details["breakdown"]["debug code"] == pytest.approx(0.0)

    def test_the_groups_are_recorded_in_the_details(self):
        details = FirstPersonFairness().evaluate(
            pairs(1), rater({"C": 1.0}), group_a="Black user",
            group_b="White user", return_details=True)
        assert details["group_a"] == "Black user"

    def test_run_produces_a_signed_result_with_an_interval(self):
        result = FirstPersonFairness().run(
            pairs=pairs(6), rate_fn=rater({"A": 0.8, "C": 0.2},
                                          {"B": 0.8, "C": 0.2}))
        assert result.score == pytest.approx(0.8)
        assert result.info.direction == "signed"
        assert result.n == 6


class TestValidation:
    def test_empty_pairs_raise(self):
        with pytest.raises(ValueError, match="at least one"):
            FirstPersonFairness().evaluate([], rater({"C": 1.0}))

    def test_a_missing_rater_raises(self):
        with pytest.raises(ValueError, match="rate_fn"):
            FirstPersonFairness().evaluate(pairs(1), None)

    @pytest.mark.parametrize("missing", ["prompt", "response_a", "response_b"])
    def test_a_pair_missing_a_key_raises_and_names_it(self, missing):
        entry = {"prompt": "p", "response_a": "a", "response_b": "b"}
        del entry[missing]
        with pytest.raises(ValueError, match=missing):
            FirstPersonFairness().evaluate([entry], rater({"C": 1.0}))


class TestMetadata:
    def test_it_is_an_adaptation_and_says_why(self):
        """The estimator is exact; the judge template is only published
        "slightly abbreviated" and no reference code was located."""
        info = FirstPersonFairness.info
        assert info.fidelity == "adaptation"
        assert "abbreviated" in info.deviation_note.lower()

    def test_it_is_judge_bound(self):
        assert FirstPersonFairness.info.resource_binding == "judge"

    def test_zero_is_neutral_and_the_range_is_signed(self):
        assert FirstPersonFairness.info.neutral_value == 0.0
        assert FirstPersonFairness.info.value_range == (-1.0, 1.0)
