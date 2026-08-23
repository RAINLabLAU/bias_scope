"""Metric metadata (PLAN.md Section 5.1).

A flat frozen dataclass of plain strings, plus one normalisation function and a
registry. Deliberately dumb: the profile view, `recommend_metrics`, and the
validation table read it, and nothing else should need to.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Dict, List, Tuple

# ── Vocabularies ───────────────────────────────────────────────────────────
FAMILIES = ("embedding", "probability", "generated_text", "prompt")
ACCESS_MODES = ("embeddings", "logits", "completions", "chat")
DIRECTIONS = ("higher_more_biased", "lower_more_biased", "signed")
RESOURCE_BINDINGS = ("language_agnostic", "lexicon", "dataset", "classifier", "judge")

#: Fidelity statuses. The first four are PLAN.md 4.1's.
#:
#: ``unaudited`` is a fifth, added here deliberately. PLAN.md Section 4.0
#: forbids assigning a fidelity status before the metric's paper and reference
#: code have been read, and that audit is in progress — 8 of 40 done. The
#: alternative was to guess a status for the other 32, which would put exactly
#: the kind of unverified claim into the public API that v0.2 exists to remove.
#: ``unaudited`` makes the gap visible instead, and `tests/test_metadata.py`
#: pins it to `sources/SOURCES.yaml`, so a metric cannot quietly stay unaudited
#: once its sources are marked read, nor claim a status without them.
#:
#: Section 13's definition of done requires zero ``mismatch``; this adds the
#: requirement of zero ``unaudited`` for 0.2.0.
FIDELITIES = ("faithful", "adaptation", "original", "mismatch", "unaudited")

#: Statuses that oblige a non-empty `deviation_note`.
FIDELITIES_NEEDING_NOTE = ("adaptation", "original", "mismatch", "unaudited")


@dataclass(frozen=True)
class MetricInfo:
    """
    What the framework layer needs to know about a metric.

    Every field is a plain string, number, or tuple of strings — no enums, no
    validation logic beyond `validate()` and the one test that calls it.

    Attributes
    ----------
    name : str
        Human-readable name, e.g. "CrowS-Pairs".
    family : str
        One of `FAMILIES`.
    access : tuple of str
        Which model access modes the metric needs; a subset of `ACCESS_MODES`.
        `recommend_metrics` will not offer a metric whose access a backend
        cannot provide.
    neutral_value : float
        The score meaning "no bias": 0.5 for CrowS-Pairs, 0 for WEAT, 100 for
        ICAT.
    direction : str
        One of `DIRECTIONS`.
    value_range : tuple of float
        Inclusive bounds. Use ``float("-inf")`` / ``float("inf")`` for
        unbounded metrics.
    fidelity : str
        One of `FIDELITIES`. See the note on `unaudited` above.
    reference : str
        Citation plus URL.
    reference_impl : str
        URL @ SHA of the authors' code, empty if none.
    languages : tuple of str
        Language codes the metric's resources support.
    resource_binding : str
        One of `RESOURCE_BINDINGS`: what ties the metric to a language.
    deviation_note : str
        Required (non-empty) when `fidelity` is not ``faithful``. Says what
        differs and why, in one or two sentences.
    fidelity_note : str
        Path to `docs/fidelity/<metric>.md`, empty until the audit is done.
    """

    name: str
    family: str
    access: Tuple[str, ...]
    neutral_value: float
    direction: str
    value_range: Tuple[float, float]
    fidelity: str
    reference: str
    reference_impl: str = ""
    languages: Tuple[str, ...] = ("en",)
    resource_binding: str = "dataset"
    deviation_note: str = ""
    fidelity_note: str = ""

    def _validate_vocabularies(self) -> List[str]:
        """Every enumerated field holds a value from its vocabulary."""
        problems: List[str] = []
        if self.family not in FAMILIES:
            problems.append(f"family must be one of {FAMILIES}, got {self.family!r}")
        if not self.access:
            problems.append("access must name at least one mode")
        for mode in self.access:
            if mode not in ACCESS_MODES:
                problems.append(f"access mode {mode!r} is not one of {ACCESS_MODES}")
        if self.direction not in DIRECTIONS:
            problems.append(f"direction must be one of {DIRECTIONS}, got {self.direction!r}")
        if self.fidelity not in FIDELITIES:
            problems.append(f"fidelity must be one of {FIDELITIES}, got {self.fidelity!r}")
        if self.resource_binding not in RESOURCE_BINDINGS:
            problems.append(
                f"resource_binding must be one of {RESOURCE_BINDINGS}, "
                f"got {self.resource_binding!r}"
            )
        return problems

    def _validate_range(self) -> List[str]:
        """The neutral value sits inside an increasing range."""
        low, high = self.value_range
        if not low < high:
            return [f"value_range must be increasing, got {self.value_range}"]
        if not low <= self.neutral_value <= high:
            return [
                f"neutral_value {self.neutral_value} is outside value_range "
                f"{self.value_range}"
            ]
        return []

    def validate(self) -> List[str]:
        """Return a list of problems; empty means the metadata is coherent."""
        problems = self._validate_vocabularies() + self._validate_range()

        if self.fidelity in FIDELITIES_NEEDING_NOTE and not self.deviation_note.strip():
            problems.append(
                f"fidelity {self.fidelity!r} requires a non-empty deviation_note "
                "saying what differs from the cited source and why"
            )
        if not self.reference.strip():
            problems.append("reference is required")
        if not self.languages:
            problems.append("languages must name at least one language")

        return problems

    def to_dict(self) -> Dict[str, object]:
        """JSON-serialisable view, for `BiasResult.to_dict`."""
        return {f.name: getattr(self, f.name) for f in fields(self)}


def normalized_deviation(score: float, info: MetricInfo) -> float:
    """
    Map any metric's score onto a signed, comparable scale.

    Zero is neutral and **positive means more stereotyped or more harmful**,
    regardless of the metric's own direction. Formula:

        raw   = score − info.neutral_value
        raw   = −raw                       if direction is "lower_more_biased"
        scale = max( neutral − low , high − neutral )
        result = raw / scale

    `scale` is the larger of the two distances from the neutral value to the
    range bounds, so an asymmetric range (CrowS-Pairs' 0.5 in [0, 100], say)
    does not make one direction look bigger than the other.

    Returns the raw signed deviation when the range is unbounded — there is no
    defensible denominator, and inventing one would let unbounded metrics be
    plotted against bounded ones as though the units matched.

    **This is a display transform, not an aggregate.** It exists so the profile
    plot can put metrics on one axis. Averaging it across metrics or families
    is exactly the composite bias score PLAN.md lists as a non-goal.

    Examples
    --------
    >>> info = MetricInfo(name="CrowS-Pairs", family="probability",
    ...                   access=("logits",), neutral_value=50.0,
    ...                   direction="higher_more_biased", value_range=(0.0, 100.0),
    ...                   fidelity="faithful", reference="Nangia et al. 2020")
    >>> normalized_deviation(75.0, info)
    0.5
    >>> normalized_deviation(50.0, info)
    0.0
    """
    raw = float(score) - float(info.neutral_value)
    if info.direction == "lower_more_biased":
        raw = -raw

    low, high = info.value_range
    if low == float("-inf") or high == float("inf"):
        return raw

    scale = max(info.neutral_value - low, high - info.neutral_value)
    if scale == 0:
        return 0.0
    return raw / scale


# ── Registry ───────────────────────────────────────────────────────────────
# Populated at import time by each family package. Kept as a plain dict rather
# than an import-side-effect plugin registry (PLAN.md Section 1 forbids those);
# `register` is called explicitly from one place per metric.
_REGISTRY: Dict[str, MetricInfo] = {}


def register(class_name: str, info: MetricInfo) -> MetricInfo:
    """Record a metric's metadata under its class name."""
    if class_name in _REGISTRY and _REGISTRY[class_name] != info:
        raise ValueError(f"{class_name} is already registered with different metadata")
    _REGISTRY[class_name] = info
    return info


def list_metrics(
    family: str | None = None,
    fidelity: str | None = None,
) -> Dict[str, MetricInfo]:
    """
    Every registered metric, optionally filtered.

    Parameters
    ----------
    family : str, optional
        Restrict to one of `FAMILIES`.
    fidelity : str, optional
        Restrict to one of `FIDELITIES`.

    Returns
    -------
    dict
        Class name -> `MetricInfo`, sorted by class name.
    """
    if family is not None and family not in FAMILIES:
        raise ValueError(f"family must be one of {FAMILIES}, got {family!r}")
    if fidelity is not None and fidelity not in FIDELITIES:
        raise ValueError(f"fidelity must be one of {FIDELITIES}, got {fidelity!r}")

    return {
        name: info
        for name, info in sorted(_REGISTRY.items())
        if (family is None or info.family == family)
        and (fidelity is None or info.fidelity == fidelity)
    }


def fidelity_counts() -> Dict[str, int]:
    """How many registered metrics carry each fidelity status."""
    counts = {status: 0 for status in FIDELITIES}
    for info in _REGISTRY.values():
        counts[info.fidelity] += 1
    return counts


__all__ = [
    "MetricInfo",
    "normalized_deviation",
    "register",
    "list_metrics",
    "fidelity_counts",
    "FAMILIES",
    "ACCESS_MODES",
    "DIRECTIONS",
    "FIDELITIES",
    "RESOURCE_BINDINGS",
]
