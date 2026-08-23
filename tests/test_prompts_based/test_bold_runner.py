"""BOLD as a benchmark runner (Dhamala et al. 2021).

BOLD is a dataset plus five metrics, not a single score. v0.1.1 shipped an
ad-hoc "lexical bias heuristic" under this name that the paper does not define;
it was removed rather than renamed (docs/fidelity/bold.md).
"""

import pytest

from bias_scope.prompts_based.bold import BOLD, BOLD_DOMAINS, BOLD_METRICS

PROMPTS = {
    "gender": {
        "American_actors": ["The actor was", "He was known for"],
        "American_actresses": ["The actress was", "She was known for"],
    },
    "profession": {
        "nursing": ["The nurse was"],
        "engineering": ["The engineer was"],
    },
}


def echo(prompt):
    return prompt + " good"


def constant(value):
    return lambda texts: [value for _ in texts]


class TestRunner:
    def test_reports_per_domain_per_metric_per_group(self):
        result = BOLD().evaluate(
            prompts=PROMPTS, generate_fn=echo,
            scorers={"sentiment": constant(0.5)},
        )
        assert set(result["scores"]) == {"gender", "profession"}
        assert set(result["scores"]["gender"]["sentiment"]) == {
            "American_actors", "American_actresses"
        }
        assert result["scores"]["gender"]["sentiment"]["American_actors"] == 0.5

    def test_no_aggregate_score_is_produced(self):
        """The paper never collapses the five metrics; neither do we."""
        result = BOLD().evaluate(
            prompts=PROMPTS, generate_fn=echo, scorers={"regard": constant(1.0)},
        )
        assert "bias_score" not in result
        assert "score" not in result

    def test_the_gap_is_the_spread_across_groups(self):
        def by_group(texts):
            # "actor" prompts score 1.0, everything else 0.0.
            return [1.0 if "actor was" in t else 0.0 for t in texts]

        result = BOLD().evaluate(
            prompts=PROMPTS, generate_fn=echo, scorers={"sentiment": by_group},
        )
        actors = result["scores"]["gender"]["sentiment"]["American_actors"]
        actresses = result["scores"]["gender"]["sentiment"]["American_actresses"]
        assert result["gaps"]["gender"]["sentiment"] == pytest.approx(
            abs(actors - actresses)
        )

    def test_a_single_group_has_a_zero_gap(self):
        result = BOLD().evaluate(
            prompts={"gender": {"only": ["a"]}}, generate_fn=echo,
            scorers={"sentiment": constant(0.3)},
        )
        assert result["gaps"]["gender"]["sentiment"] == 0.0

    def test_multiple_scorers_run_independently(self):
        result = BOLD().evaluate(
            prompts=PROMPTS, generate_fn=echo,
            scorers={"sentiment": constant(0.5), "toxicity": constant(0.1)},
        )
        assert result["scores"]["gender"]["sentiment"]["American_actors"] == 0.5
        assert result["scores"]["gender"]["toxicity"]["American_actors"] == 0.1

    def test_domains_can_be_restricted(self):
        result = BOLD().evaluate(
            prompts=PROMPTS, generate_fn=echo, scorers={"regard": constant(0.0)},
            domains=["profession"],
        )
        assert result["domains"] == ["profession"]
        assert "gender" not in result["scores"]

    def test_generate_fn_is_called_once_per_prompt(self):
        calls = []

        def counting(prompt):
            calls.append(prompt)
            return prompt

        BOLD().evaluate(prompts=PROMPTS, generate_fn=counting,
                        scorers={"regard": constant(0.0)})
        assert len(calls) == 6

    def test_details_expose_the_generations(self):
        result = BOLD().evaluate(
            prompts=PROMPTS, generate_fn=echo, scorers={"regard": constant(0.0)},
            return_details=True,
        )
        assert result["generations"]["gender"]["American_actors"] == [
            "The actor was good", "He was known for good"
        ]

    def test_an_unknown_scorer_name_is_flagged_not_silently_accepted(self):
        """A typo must not masquerade as one of BOLD's five metrics."""
        result = BOLD().evaluate(
            prompts=PROMPTS, generate_fn=echo, scorers={"sentimnet": constant(0.0)},
        )
        assert result["unknown_scorers"] == ["sentimnet"]

    def test_the_five_metric_names_are_the_papers(self):
        assert set(BOLD_METRICS) == {
            "sentiment", "toxicity", "regard",
            "psycholinguistic_norms", "gender_polarity",
        }

    def test_the_five_domains_are_the_papers(self):
        assert set(BOLD_DOMAINS) == {
            "gender", "race", "profession",
            "religious_ideology", "political_ideology",
        }


class TestValidation:
    def test_empty_prompts_raises(self):
        with pytest.raises(ValueError, match="prompts"):
            BOLD().evaluate(prompts={}, generate_fn=echo,
                            scorers={"regard": constant(0.0)})

    def test_missing_generate_fn_raises(self):
        with pytest.raises(ValueError, match="generate_fn"):
            BOLD().evaluate(prompts=PROMPTS, generate_fn=None,
                            scorers={"regard": constant(0.0)})

    def test_no_scorers_raises_and_names_the_five(self):
        with pytest.raises(ValueError, match="five metrics"):
            BOLD().evaluate(prompts=PROMPTS, generate_fn=echo, scorers={})

    def test_an_unknown_domain_raises(self):
        with pytest.raises(ValueError, match="not present"):
            BOLD().evaluate(prompts=PROMPTS, generate_fn=echo,
                            scorers={"regard": constant(0.0)}, domains=["religion"])

    def test_a_scorer_returning_the_wrong_count_raises(self):
        with pytest.raises(ValueError, match="returned"):
            BOLD().evaluate(prompts=PROMPTS, generate_fn=echo,
                            scorers={"regard": lambda texts: [0.0]})

    def test_the_lexical_heuristic_is_gone(self):
        """v0.1.1's `_bias_score` was removed, not renamed."""
        assert not hasattr(BOLD, "_bias_score")
