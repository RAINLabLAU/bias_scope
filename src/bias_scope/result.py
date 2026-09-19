"""`BiasResult` and the protocol block (PLAN.md Section 5.3).

`evaluate()` keeps returning what it returns today — a float, or a dict with
`return_details=True` — so every existing script, example, and test is
untouched. The framework layer uses `run()`, which returns one of these.

No operator overloading and no float look-alike: a `BiasResult` is a record,
and reading `.score` is how you get the number.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from bias_scope.metadata import MetricInfo, normalized_deviation
from bias_scope.utils import protocol_hash

#: Keys `make_protocol` always emits, so a protocol block has a fixed shape
#: whether or not the caller knew every value.
PROTOCOL_KEYS = (
    "metric", "model_id", "dtype", "seed", "dataset", "dataset_revision",
    "decoding", "resources", "judge_model", "judge_prompt_version", "permutation_seed",
    "random_seed", "library_version", "timestamp", "hash",
)


def _library_version() -> str:
    """The installed version, preferring package metadata over the attribute.

    `bias_scope.__version__` and `pyproject.toml` have drifted apart before
    (REVIEW_LATER RL-004), so the distribution metadata wins where available.
    """
    try:
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version("bias-scope")
        except PackageNotFoundError:
            pass
    except ImportError:  # pragma: no cover - importlib.metadata is stdlib in 3.10+
        pass

    import bias_scope

    return getattr(bias_scope, "__version__", "unknown")


def make_protocol(
    metric: str,
    model_id: Optional[str] = None,
    dtype: Optional[str] = None,
    seed: int = 42,
    dataset: Optional[str] = None,
    dataset_revision: Optional[str] = None,
    decoding: Optional[Dict[str, Any]] = None,
    resources: Optional[List[Dict[str, str]]] = None,
    judge_model: Optional[str] = None,
    judge_prompt_version: Optional[str] = None,
    permutation_seed: Optional[int] = None,
    random_seed: Optional[int] = None,
    timestamp: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build the protocol block that travels with every result.

    Section 1 requires every result written to `results/` to carry one, because
    a bias number without its protocol is not reproducible: the same metric on
    the same model moves with dtype, decoding, lexicon version, and judge.

    `hash` is `protocol_hash` over every field **except** `timestamp`, so two
    runs of the same protocol at different times share a hash and the
    generation cache can key on it.

    Parameters
    ----------
    metric : str
        Metric class name.
    model_id : str, optional
        Model identifier, e.g. "gpt2" or "openai/gpt-4o".
    dtype : str, optional
        "fp32", "bf16", … Record it: AWQ-INT4 moved BBQ by 17 points.
    seed : int
        The seed passed to `seed_everything`.
    dataset, dataset_revision : str, optional
        Dataset name and pinned revision.
    decoding : dict, optional
        Decoding parameters (top_p, top_k, temperature, num_return_sequences).
    resources : list of dict, optional
        Lexicons and classifiers used, each with a name and a sha256.
    judge_model, judge_prompt_version : str, optional
        For judge-based metrics; the prompt version is a filename plus hash.
    permutation_seed : int, optional
        Effective random seed for a metric's permutation procedure, when used.
    random_seed : int, optional
        Effective random seed for a metric's own Monte Carlo sampling (e.g.
        CEAT's context resampling), when used and distinct from a permutation
        test.
    timestamp : str, optional
        ISO-8601. Defaults to now, in UTC. Excluded from the hash.

    Returns
    -------
    dict
        Plain, JSON-serialisable, with exactly `PROTOCOL_KEYS`.

    Examples
    --------
    >>> a = make_protocol("WEAT", model_id="gpt2", timestamp="2020-01-01T00:00:00Z")
    >>> b = make_protocol("WEAT", model_id="gpt2", timestamp="2026-08-22T00:00:00Z")
    >>> a["hash"] == b["hash"]          # timestamp is excluded from the hash
    True
    """
    protocol: Dict[str, Any] = {
        "metric": metric,
        "model_id": model_id,
        "dtype": dtype,
        "seed": seed,
        "dataset": dataset,
        "dataset_revision": dataset_revision,
        "decoding": decoding or {},
        "resources": resources or [],
        "judge_model": judge_model,
        "judge_prompt_version": judge_prompt_version,
        "permutation_seed": permutation_seed,
        "random_seed": random_seed,
        "library_version": _library_version(),
    }
    protocol["hash"] = protocol_hash(protocol)
    protocol["timestamp"] = timestamp or dt.datetime.now(dt.timezone.utc).isoformat()
    return {key: protocol[key] for key in PROTOCOL_KEYS}


@dataclass
class BiasResult:
    """
    One metric's result, with everything needed to interpret and reproduce it.

    Attributes
    ----------
    metric : str
        Metric class name.
    score : float
        The headline number, in `info.value_range`.
    n : int
        Number of items actually scored — not the number requested.
    ci : (float, float) or None
        95% interval, or None when the metric has no item-level scores and the
        paper defines no interval.
    ci_method : str
        "bootstrap" | "wald" | "hedges_olkin" | "permutation" | "none".
    per_item : list of float or None
        Item-level scores, when the metric has them.
    breakdown : dict
        Per-group or per-category scores. May be empty.
    details : dict
        Whatever `evaluate(return_details=True)` returns.
    protocol : dict
        From `make_protocol`.
    info : MetricInfo
        The metric's metadata.
    """

    metric: str
    score: float
    n: int
    ci: Optional[Tuple[float, float]]
    ci_method: str
    per_item: Optional[List[float]]
    breakdown: Dict[str, float]
    details: Dict[str, Any]
    protocol: Dict[str, Any]
    info: MetricInfo
    p_value: Optional[float] = field(default=None)

    def normalized_deviation(self) -> float:
        """Signed deviation from neutral; positive means more biased."""
        return normalized_deviation(self.score, self.info)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable view. `from_dict` round-trips it."""
        return {
            "metric": self.metric,
            "score": self.score,
            "n": self.n,
            "ci": list(self.ci) if self.ci is not None else None,
            "ci_method": self.ci_method,
            "per_item": list(self.per_item) if self.per_item is not None else None,
            "breakdown": dict(self.breakdown),
            "details": self.details,
            "protocol": self.protocol,
            "info": self.info.to_dict(),
            "p_value": self.p_value,
            "normalized_deviation": self.normalized_deviation(),
        }

    def __repr__(self) -> str:
        ci = "" if self.ci is None else f", ci=[{self.ci[0]:.4g}, {self.ci[1]:.4g}]"
        return (
            f"BiasResult({self.metric}, score={self.score:.4g}, n={self.n}{ci}, "
            f"fidelity={self.info.fidelity})"
        )


def from_dict(payload: Dict[str, Any]) -> BiasResult:
    """
    Rebuild a `BiasResult` from `to_dict` output.

    Module-level rather than a classmethod, to keep `BiasResult` a plain record.

    Examples
    --------
    >>> from bias_scope.metadata import MetricInfo
    >>> info = MetricInfo(name="X", family="embedding", access=("embeddings",),
    ...                   neutral_value=0.0, direction="signed",
    ...                   value_range=(-2.0, 2.0), fidelity="faithful",
    ...                   reference="ref")
    >>> r = BiasResult("X", 1.0, 8, (0.5, 1.5), "bootstrap", [1.0], {}, {}, {}, info)
    >>> from_dict(r.to_dict()).score
    1.0
    """
    info_fields = dict(payload["info"])
    for key in ("access", "languages"):
        if key in info_fields and info_fields[key] is not None:
            info_fields[key] = tuple(info_fields[key])
    if "value_range" in info_fields and info_fields["value_range"] is not None:
        info_fields["value_range"] = tuple(info_fields["value_range"])

    ci = payload.get("ci")
    return BiasResult(
        metric=payload["metric"],
        score=payload["score"],
        n=payload["n"],
        ci=tuple(ci) if ci is not None else None,
        ci_method=payload.get("ci_method", "none"),
        per_item=payload.get("per_item"),
        breakdown=payload.get("breakdown", {}),
        details=payload.get("details", {}),
        protocol=payload.get("protocol", {}),
        info=MetricInfo(**info_fields),
        p_value=payload.get("p_value"),
    )


def _as_dict(result: BiasResult) -> Dict[str, Any]:
    """`dataclasses.asdict`, kept for symmetry in tests."""
    return asdict(result)


__all__ = ["BiasResult", "make_protocol", "from_dict", "PROTOCOL_KEYS"]
