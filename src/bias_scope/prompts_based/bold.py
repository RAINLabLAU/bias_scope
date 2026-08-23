"""BOLD — a benchmark runner over five metrics (Dhamala et al. 2021).

BOLD is a **dataset plus a suite of five metrics**, not a single score. The
class that shipped under this name through v0.1.1 computed an ad-hoc "lexical
bias heuristic" that appears nowhere in the paper; it was removed rather than
renamed, because unlike the other v0.1.1 mismatches it had no defensible
statistic behind it. See `docs/fidelity/bold.md`.

This module runs BOLD's protocol: load prompts by domain, generate a
continuation for each, and score the continuations with whichever of the
paper's five metrics the caller supplies. Results are reported **per domain per
metric** — the paper never aggregates them into one number, and neither does
this.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from bias_scope.base import PromptBasedMetric

#: The five domains of the BOLD prompt set.
BOLD_DOMAINS = (
    "gender",
    "race",
    "profession",
    "religious_ideology",
    "political_ideology",
)

#: The five metrics BOLD defines (paper §"Bias metrics"). Names map onto the
#: library classes that implement them; `sentiment` is VADER, which BOLD uses
#: directly rather than via a BiasScope metric.
BOLD_METRICS = (
    "sentiment",
    "toxicity",
    "regard",
    "psycholinguistic_norms",
    "gender_polarity",
)


class BOLD(PromptBasedMetric):
    """
    BOLD benchmark runner.

    Not a metric: `BOLD` orchestrates generation over the BOLD prompt set and
    hands the continuations to scorers. `evaluate` returns a table of
    ``{domain: {metric_name: score}}`` plus the raw per-group values.

    The paper's five metrics are sentiment (VADER, with ±0.5 thresholds),
    toxicity, regard (Sheng et al. 2019's classifier), psycholinguistic norms
    (NRC-VAD), and gender polarity (unigram and embedding variants). Each is
    implemented elsewhere in this library; pass the ones you want as `scorers`.

    Because scoring depends on classifiers and lexicons that are separate
    downloads, nothing is wired in by default: a caller must say which scorers
    to run. That keeps the runner honest about what produced each number, and
    keeps the protocol block accurate.

    Example
    -------
    >>> prompts = {"gender": {"American_actors": ["The actor was"],
    ...                       "American_actresses": ["The actress was"]}}
    >>> runner = BOLD(model_name="stub/model")
    >>> table = runner.evaluate(
    ...     prompts=prompts,
    ...     generate_fn=lambda p: p + " praised",
    ...     scorers={"sentiment": lambda texts: [1.0 for _ in texts]},
    ... )
    >>> table["scores"]["gender"]["sentiment"]["American_actors"]
    1.0
    """

    def __init__(self, model_name: str = "", api_key: Optional[str] = None) -> None:
        """
        Args:
            model_name (str): Identifier recorded in the protocol block. Not
                used when `generate_fn` is supplied.
            api_key (str | None): Passed to LiteLLM when generating.
        """
        self.model_name = model_name
        self.api_key = api_key
        self.dataset_name = "AlexaAI/bold"

    def evaluate(
        self,
        prompts: Mapping[str, Mapping[str, Sequence[str]]],
        generate_fn: Optional[Callable[[str], str]] = None,
        scorers: Optional[Mapping[str, Callable[[Sequence[str]], Sequence[float]]]] = None,
        *,
        domains: Optional[Sequence[str]] = None,
        return_details: bool = False,
    ) -> Dict[str, Any]:
        """
        Run BOLD over the supplied prompts.

        Args:
            prompts (Mapping): ``{domain: {group: [prompt, ...]}}``. BOLD's own
                groups are e.g. ``American_actors`` / ``American_actresses``
                within the ``gender`` domain.
            generate_fn (Callable): Maps a prompt to a continuation. Required.
            scorers (Mapping): ``{metric_name: fn}`` where ``fn`` takes a list
                of continuations and returns one score per continuation. Names
                should come from `BOLD_METRICS`; unknown names are accepted but
                flagged in the result so a typo cannot masquerade as a metric.
            domains (Sequence[str], optional): Restrict to these domains.
            return_details (bool): Include the generations themselves.

        Returns:
            dict: ``scores[domain][metric][group]`` = mean score, plus
            ``gaps[domain][metric]`` = max-minus-min across groups, the
            per-domain group counts, and the list of unknown scorer names.

        Raises:
            ValueError: If `prompts` is empty, `generate_fn` is missing, or
                `scorers` is empty.

        Notes:
            No aggregate "BOLD score" is produced. The paper reports per-domain
            per-metric values and so does this; collapsing them would be the
            composite score PLAN.md lists as a non-goal.
        """
        if not prompts:
            raise ValueError("prompts must contain at least one domain")
        if generate_fn is None or not callable(generate_fn):
            raise ValueError("generate_fn is required and must be callable")
        if not scorers:
            raise ValueError(
                "scorers must name at least one of BOLD's five metrics: "
                f"{BOLD_METRICS}. Nothing is wired in by default so that the "
                "protocol records exactly which classifier produced each number."
            )

        selected = list(domains) if domains else list(prompts)
        unknown_domains = [d for d in selected if d not in prompts]
        if unknown_domains:
            raise ValueError(f"domains not present in prompts: {unknown_domains}")

        unknown_scorers = [n for n in scorers if n not in BOLD_METRICS]

        generations: Dict[str, Dict[str, List[str]]] = {}
        for domain in selected:
            generations[domain] = {
                group: [generate_fn(p) for p in group_prompts]
                for group, group_prompts in prompts[domain].items()
            }

        scores: Dict[str, Dict[str, Dict[str, float]]] = {}
        gaps: Dict[str, Dict[str, float]] = {}
        for domain in selected:
            scores[domain] = {
                name: self._score_groups(name, scorer, generations[domain], domain)
                for name, scorer in scorers.items()
            }
            gaps[domain] = {
                name: self._spread(per_group)
                for name, per_group in scores[domain].items()
            }

        result: Dict[str, Any] = {
            "scores": scores,
            "gaps": gaps,
            "domains": selected,
            "metrics": list(scorers),
            "unknown_scorers": unknown_scorers,
            "num_prompts": {
                d: sum(len(v) for v in prompts[d].values()) for d in selected
            },
            "groups": {d: sorted(prompts[d]) for d in selected},
            "metric": "BOLD",
            "category": self.category,
        }
        if return_details:
            result["generations"] = generations
        return result

    @staticmethod
    def _score_groups(
        name: str,
        scorer: Callable[[Sequence[str]], Sequence[float]],
        by_group: Mapping[str, Sequence[str]],
        domain: str,
    ) -> Dict[str, float]:
        """Mean score per group for one metric, with an arity check."""
        per_group: Dict[str, float] = {}
        for group, texts in by_group.items():
            if not texts:
                continue
            values = [float(v) for v in scorer(texts)]
            if len(values) != len(texts):
                raise ValueError(
                    f"scorer {name!r} returned {len(values)} scores for "
                    f"{len(texts)} generations in {domain}/{group}"
                )
            per_group[group] = sum(values) / len(values)
        return per_group

    @staticmethod
    def _spread(per_group: Mapping[str, float]) -> float:
        """Max minus min across groups; 0 when there is nothing to compare."""
        if len(per_group) < 2:
            return 0.0
        return max(per_group.values()) - min(per_group.values())
