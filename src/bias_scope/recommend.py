"""Metric selection (PLAN.md 5.4).

`recommend_metrics` answers "which metrics can I actually run on this model, for
this axis, in this language, and how much should I trust each one?"

The access filter is a hard constraint derived from the backend, not a
preference: a metric needing `logits` is not offered for a chat API because the
API cannot provide them. The fidelity information is advisory but always
attached, so a caller cannot pick a metric without seeing whether it is faithful
to its cited paper.
"""

from __future__ import annotations

from typing import Dict, List, NamedTuple, Optional, Sequence

from bias_scope.metadata import ACCESS_MODES, MetricInfo, list_metrics

#: Metrics that measure something other than social bias and so are excluded
#: from a bias profile by default. PLAN.md Section 12 pre-decides this for
#: TruthfulQA ("truthfulness (not social bias)"); TofNof measures sycophancy,
#: which is the same kind of scope question.
NOT_SOCIAL_BIAS = {"TruthfulQA", "TofNof"}


class Recommendation(NamedTuple):
    """One offered metric, with the reason it is offered."""

    metric: str
    info: MetricInfo
    reason: str

    @property
    def fidelity(self) -> str:
        return self.info.fidelity


def _reason(info: MetricInfo, axis: Optional[str], language: str) -> str:
    """A one-line justification a caller can read without the docs."""
    bits = [f"{info.family} family", f"needs {'+'.join(info.access)}"]
    if language != "en" or info.resource_binding != "language_agnostic":
        bits.append(f"{info.resource_binding}-bound, {'/'.join(info.languages)}")
    if axis:
        bits.append(f"axis {axis}")

    caveat = {
        "faithful": "faithful to its cited paper",
        "adaptation": "ADAPTATION — deviates from the paper; see deviation_note",
        "original": "ORIGINAL — BiasScope's own metric, no published reference",
        "mismatch": "MISMATCH — does not implement its cited metric; do not use",
        "unaudited": "UNAUDITED — sources not read; no fidelity claim",
    }[info.fidelity]
    bits.append(caveat)
    return "; ".join(bits)


def recommend_metrics(
    access: Sequence[str],
    axis: Optional[str] = None,
    language: str = "en",
    *,
    include_mismatch: bool = False,
    include_unaudited: bool = True,
    include_non_bias: bool = False,
    family: Optional[str] = None,
) -> List[Recommendation]:
    """
    Metrics runnable with the given access, sorted by trustworthiness.

    Args:
        access (Sequence[str]): What the backend provides, e.g.
            ``("completions", "chat")``. Take it from `Backend.access` rather
            than typing it, so it cannot drift from reality.
        axis (str, optional): Bias axis, e.g. "gender". Recorded in the reason;
            axis-specific filtering arrives when metrics declare their axes
            (PLAN.md 5.4 says not to add the field before it is needed).
        language (str): Language code. A metric is offered only if it declares
            support, so a `lexicon`-bound English metric is not offered for
            French.
        include_mismatch (bool): Offer metrics known to implement a different
            statistic than they cite. Default False — there is no good reason
            to run one, and they are all scheduled for repair.
        include_unaudited (bool): Offer metrics whose sources have not been
            read. Default True, because excluding them would hide them; the
            reason string carries the warning.
        include_non_bias (bool): Offer TruthfulQA and TofNof, which measure
            truthfulness and sycophancy rather than social bias.
        family (str, optional): Restrict to one family.

    Returns:
        list of Recommendation: Sorted faithful → adaptation → original →
        unaudited → mismatch, then alphabetically. Trustworthy metrics first.

    Raises:
        ValueError: If `access` is empty or contains an unknown mode.

    Examples:
        >>> names = [r.metric for r in recommend_metrics(("embeddings",))]
        >>> "WEAT" in names
        True
        >>> # A chat API cannot run a metric that needs logits.
        >>> "CrowSPairs" in [r.metric for r in recommend_metrics(("chat",))]
        False
    """
    if not access:
        raise ValueError("access must name at least one mode")
    unknown = set(access) - set(ACCESS_MODES)
    if unknown:
        raise ValueError(f"unknown access modes: {sorted(unknown)}; expected {ACCESS_MODES}")

    order = {"faithful": 0, "adaptation": 1, "original": 2, "unaudited": 3, "mismatch": 4}
    available = set(access)
    out: List[Recommendation] = []

    for name, info in list_metrics(family=family).items():
        if not set(info.access) <= available:
            continue
        if language not in info.languages:
            continue
        if info.fidelity == "mismatch" and not include_mismatch:
            continue
        if info.fidelity == "unaudited" and not include_unaudited:
            continue
        if name in NOT_SOCIAL_BIAS and not include_non_bias:
            continue
        out.append(Recommendation(name, info, _reason(info, axis, language)))

    out.sort(key=lambda r: (order[r.info.fidelity], r.metric))
    return out


def explain_exclusions(
    access: Sequence[str],
    language: str = "en",
) -> Dict[str, str]:
    """
    Why each *excluded* metric was not offered.

    A recommender that silently drops two thirds of the library is hard to
    trust. This says, per metric, which constraint removed it.

    Returns:
        dict: metric name -> reason for exclusion.
    """
    available = set(access)
    reasons: Dict[str, str] = {}
    for name, info in list_metrics().items():
        missing = set(info.access) - available
        if missing:
            reasons[name] = f"needs {'+'.join(sorted(missing))}, which this backend lacks"
        elif language not in info.languages:
            reasons[name] = f"no {language} resources ({info.resource_binding}-bound)"
        elif info.fidelity == "mismatch":
            reasons[name] = "mismatch: does not implement its cited metric"
        elif name in NOT_SOCIAL_BIAS:
            reasons[name] = "measures truthfulness/sycophancy, not social bias"
    return reasons


__all__ = ["recommend_metrics", "explain_exclusions", "Recommendation", "NOT_SOCIAL_BIAS"]
