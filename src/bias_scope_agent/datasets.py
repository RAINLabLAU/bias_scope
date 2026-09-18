"""Metric inputs by reference, so the agent LLM never retypes evaluation data.

`run_suite` originally took its data as a tool argument, which meant every
item a metric scores had to pass through the model's own output tokens. Two
different frontier models corrupted a 20-pair list that way: one rewrote
"one of the best engineers in *her* field" as "*his* field" - destroying the
minimal pair on exactly the token CrowS-Pairs measures - and the other silently
dropped the last pair and scored 19 items where 20 were asked for. Both scores
were honest outputs of the metric on data nobody chose (REVIEW_LATER.md
RL-053). Embeddings arrays cannot survive that path at all.

Here the agent *names* a dataset and the harness loads it. Nothing but counts
and provenance crosses the tool boundary, so the data a metric scores is
byte-identical to the file on disk, and `protocol_hash`'s promise that a result
is tied to a known input means something again.

Sources are the authors' own releases, vendored at pinned SHAs by
`scripts/sources/fetch_sources.py` (see `sources/SOURCES.yaml`). They are
git-ignored, so every loader fails with the command that restores them rather
than with a stack trace.
"""

from __future__ import annotations

import csv
import hashlib
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from bias_scope.backends import Backend

_THIRD_PARTY = Path("third_party/code")

_CROWS_RELATIVE = "crows-pairs/data/crows_pairs_anonymized.csv"
_SENT_BIAS_TESTS = "sent-bias/tests"

# CrowS-Pairs' own `bias_type` spellings, for the axis names this library uses.
_CROWS_AXIS = {
    "gender": "gender",
    "race": "race-color",
    "religion": "religion",
    "age": "age",
    "nationality": "nationality",
    "socioeconomic": "socioeconomic",
    "sexual-orientation": "sexual-orientation",
    "physical-appearance": "physical-appearance",
    "disability": "disability",
}

# Which of Caliskan's ten tests measures which axis. Only the ones whose target
# and attribute categories actually name the axis are listed: WEAT 1 and 2
# (flowers/insects, instruments/weapons) measure no social axis, and no WEAT
# test covers religion - `available_datasets` says so rather than substituting.
_WEAT_BY_AXIS = {"gender": "weat6", "race": "weat3", "age": "weat10"}
_SEAT_BY_AXIS = {"gender": "sent-weat6", "race": "sent-weat3", "age": "sent-weat10"}


@dataclass(frozen=True)
class DatasetSpec:
    """One named body of evaluation data and the metrics it can feed."""

    name: str
    description: str
    metrics: Tuple[str, ...]
    axes: Tuple[str, ...]
    source: str
    #: Constructor arguments to fill from the backend. Declared per provider,
    #: never inferred from a signature: `model_name` means the model under test
    #: for CrowSPairs, the classifier Sheng et al. require for RegardScore, and
    #: the sentence encoder for WEAT/SEAT. A provider serving a metric of the
    #: second kind must set this to () (REVIEW_LATER RL-052).
    init_from_backend: Tuple[str, ...] = ("model_name", "device")


DATASETS: Dict[str, DatasetSpec] = {
    "crows_pairs": DatasetSpec(
        name="crows_pairs",
        description=(
            "Nangia et al. 2020 CrowS-Pairs, the authors' own anonymized CSV. "
            "Each item is a (more-stereotypical, less-stereotypical) minimal "
            "pair, which is the order these metrics expect."
        ),
        # Deliberately not LMB or PairwiseLikelihoodPreference: both require the
        # two sentences of a pair to tokenize to the same length (RedditBias's
        # protocol), and CrowS-Pairs pairs routinely do not - they fail with
        # "Sentence pairs must have same length". Feeding them this data would
        # be using the wrong dataset for the metric, not a convenience.
        metrics=("CrowSPairs", "AUL", "AULA"),
        axes=tuple(_CROWS_AXIS),
        source=_CROWS_RELATIVE,
    ),
    "weat": DatasetSpec(
        name="weat",
        description=(
            "Caliskan et al. 2017 word sets, as vendored in May et al.'s "
            "sent-bias repository. Word lists, embedded by the metric itself "
            "using the model under evaluation."
        ),
        # CEAT is deliberately absent: it reports `n_samples` (its permutation
        # sample count) rather than an items-scored count, so `run()`'s guard
        # rejects it with "n must be positive, got 0" for every input. That is
        # REVIEW_LATER RL-048, an open question about Guo & Caliskan's
        # definition - not something a dataset provider may paper over.
        metrics=("WEAT",),
        axes=tuple(_WEAT_BY_AXIS),
        source=f"{_SENT_BIAS_TESTS}/weat<n>.jsonl",
    ),
    "seat": DatasetSpec(
        name="seat",
        description=(
            "May et al. 2019 SEAT sentence sets - the WEAT words placed in "
            "semantically bleached template sentences."
        ),
        metrics=("SEAT",),
        axes=tuple(_SEAT_BY_AXIS),
        source=f"{_SENT_BIAS_TESTS}/sent-weat<n>.jsonl",
    ),
}


def _require(path: Path, metric_hint: str) -> Path:
    if not path.exists():
        raise ValueError(
            f"{path} is not present. third_party/ is git-ignored; restore it with\n"
            f"    python scripts/sources/fetch_sources.py --metric {metric_hint}"
        )
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _init_kwargs(metric_name: str, backend: Backend, allowed: Tuple[str, ...]) -> Dict[str, Any]:
    """Constructor arguments naming the model under evaluation.

    `allowed` comes from the provider's own `init_from_backend`, so a provider
    serving a metric whose `model_name` is a fixed resource rather than the
    model under test declares `()` and nothing is filled. The signature is
    consulted only to avoid passing an argument the metric does not accept.
    """
    from bias_scope_agent.introspection import _agent_metric_classes

    cls = _agent_metric_classes().get(metric_name)
    if cls is None:
        return {}
    accepted = inspect.signature(cls.__init__).parameters
    kwargs: Dict[str, Any] = {}
    if "model_name" in allowed and "model_name" in accepted:
        kwargs["model_name"] = backend.model_id
    device = getattr(backend, "device", None)
    if "device" in allowed and "device" in accepted and device:
        kwargs["device"] = device
    return kwargs


def _crows_pairs(root: Path, axis: str, limit: Optional[int]) -> List[List[str]]:
    bias_type = _CROWS_AXIS.get(axis)
    if bias_type is None:
        raise ValueError(f"crows_pairs has no axis {axis!r}; available: {sorted(_CROWS_AXIS)}")
    path = _require(root / _CROWS_RELATIVE, "CrowSPairs")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["bias_type"] == bias_type]
    if limit is not None:
        rows = rows[:limit]
    return [[row["sent_more"], row["sent_less"]] for row in rows]


def _word_sets(path: Path) -> Tuple[List[str], List[str], List[str], List[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return tuple(data[key]["examples"] for key in ("targ1", "targ2", "attr1", "attr2"))


def _association_test(root: Path, axis: str, by_axis: Dict[str, str], hint: str) -> Path:
    test = by_axis.get(axis)
    if test is None:
        raise ValueError(
            f"no {hint} test measures axis {axis!r}; available: {sorted(by_axis)}. "
            f"Caliskan's tests 1-2 measure no social axis and none covers religion."
        )
    return _require(root / _SENT_BIAS_TESTS / f"{test}.jsonl", hint.upper())


def _build_crows(backend, metrics, axis, limit, root, allowed) -> Tuple[Dict, Dict]:
    pairs = _crows_pairs(root, axis, limit)
    inputs = {
        name: {"__init__": _init_kwargs(name, backend, allowed), "sentence_pairs": pairs}
        for name in metrics
    }
    path = root / _CROWS_RELATIVE
    provenance = {
        "source": str(path),
        "sha256": _sha256(path),
        "pairs": len(pairs),
        "axis": axis,
        "note": "order is (more-stereotypical, less-stereotypical), the file's own column order",
    }
    return inputs, provenance


def _build_association(
    backend, metrics, axis, limit, root, allowed, by_axis, hint
) -> Tuple[Dict, Dict]:
    path = _association_test(root, axis, by_axis, hint)
    targ1, targ2, attr1, attr2 = _word_sets(path)
    if limit is not None:
        targ1, targ2 = targ1[:limit], targ2[:limit]
        attr1, attr2 = attr1[:limit], attr2[:limit]
    inputs = {
        name: {
            "__init__": _init_kwargs(name, backend, allowed),
            "target_embeddings": (targ1, targ2),
            "attribute_embeddings": (attr1, attr2),
        }
        for name in metrics
    }
    provenance = {
        "source": str(path),
        "sha256": _sha256(path),
        "targets": [len(targ1), len(targ2)],
        "attributes": [len(attr1), len(attr2)],
        "axis": axis,
        "note": "word/sentence lists; the metric embeds them with the model under evaluation",
    }
    return inputs, provenance


_BUILDERS: Dict[str, Callable[..., Tuple[Dict, Dict]]] = {
    "crows_pairs": _build_crows,
    "weat": lambda b, m, a, lim, r, al: _build_association(
        b, m, a, lim, r, al, _WEAT_BY_AXIS, "weat"
    ),
    "seat": lambda b, m, a, lim, r, al: _build_association(
        b, m, a, lim, r, al, _SEAT_BY_AXIS, "seat"
    ),
}


def available_datasets(metric_names: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    """Every dataset the harness can load, optionally filtered to metrics."""
    wanted = set(metric_names) if metric_names else None
    rows = []
    for spec in DATASETS.values():
        served = [m for m in spec.metrics if wanted is None or m in wanted]
        if not served:
            continue
        rows.append(
            {
                "dataset": spec.name,
                "description": spec.description,
                "metrics": served,
                "axes": list(spec.axes),
                "source": spec.source,
            }
        )
    return rows


def build_inputs(
    backend: Backend,
    dataset: str,
    metric_names: Sequence[str],
    axis: str = "gender",
    limit: Optional[int] = None,
    root: Path = _THIRD_PARTY,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """Load `dataset` and shape it as `run_suite` inputs for `metric_names`.

    Returns (inputs, provenance). The inputs never leave the process; the
    provenance is what the agent and the user see.
    """
    spec = DATASETS.get(dataset)
    if spec is None:
        raise ValueError(f"unknown dataset {dataset!r}; available: {sorted(DATASETS)}")
    unsupported = [name for name in metric_names if name not in spec.metrics]
    if unsupported:
        raise ValueError(
            f"dataset {dataset!r} does not serve {unsupported}; it serves "
            f"{list(spec.metrics)}. Call list_datasets to see which dataset feeds a metric."
        )
    inputs, provenance = _BUILDERS[dataset](
        backend, list(metric_names), axis, limit, root, spec.init_from_backend
    )
    provenance |= {"dataset": dataset, "metrics": list(metric_names)}
    return inputs, provenance
