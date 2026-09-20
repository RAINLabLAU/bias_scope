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
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from bias_scope.backends import Backend
from bias_scope_agent.datasets_ceat import CEAT_BUILDERS, CEAT_DATASETS
from bias_scope_agent.datasets_common import (
    _SEAT_BY_AXIS,
    _SENT_BIAS_TESTS,
    _THIRD_PARTY,
    _WEAT_BY_AXIS,
    DatasetSpec,
    _association_test,
    _init_kwargs,
    _require,
    _sha256,
    _word_sets,
)
from bias_scope_agent.datasets_generated import GENERATED_BUILDERS, GENERATED_DATASETS
from bias_scope_agent.datasets_toxicity import TOXICITY_BUILDERS, TOXICITY_DATASETS

_CROWS_RELATIVE = "crows-pairs/data/crows_pairs_anonymized.csv"
_STEREOSET_DEV = "StereoSet/data/dev.json"

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
    "stereoset": DatasetSpec(
        name="stereoset",
        description=(
            "Nadeem et al. 2021 StereoSet, the authors' own dev.json, "
            "intrasentence split. Each item is a context with one blank and "
            "three single-word fills (stereotype, anti-stereotype, unrelated)."
        ),
        metrics=("CAT", "ICAT"),
        axes=("gender", "race", "religion", "profession"),
        source=_STEREOSET_DEV,
    ),
    "weat": DatasetSpec(
        name="weat",
        description=(
            "Caliskan et al. 2017 word sets, as vendored in May et al.'s "
            "sent-bias repository. Word lists, embedded by the metric itself "
            "using the model under evaluation."
        ),
        # CEAT is deliberately absent, and no longer for RL-048's reason (that
        # is fixed - it completes `run()` now). CEAT's whole point is sampling
        # a *different context* for each word on every draw; this provider
        # supplies one embedding per word, so CEAT would resample from a fixed
        # set of eight vectors, which is WEAT with extra steps and not CEAT.
        # Serving it faithfully needs per-word sets of contextual embeddings.
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


def _stereoset_cases(root: Path, axis: str, limit: Optional[int]) -> Tuple[List[Dict], int]:
    """CAT's test-case shape, from StereoSet's own intrasentence items.

    CAT takes a tokenized context containing `[MASK]` and one candidate *word*
    per label. StereoSet writes the blank as the literal token `BLANK` and
    gives a full sentence per label, so the fill is recovered by stripping the
    context's fixed prefix and suffix from each sentence.

    Items whose fill is more than one word, or whose blank is glued to
    punctuation, cannot be expressed in that shape and are skipped rather than
    truncated - 26 of the 255 gender items in dev.json. The count of skipped
    items is reported in the provenance, never silently dropped.
    """
    path = _require(root / _STEREOSET_DEV, "CAT")
    data = json.loads(path.read_text(encoding="utf-8"))
    label_key = {"stereotype": "stereotype", "anti-stereotype": "anti_stereotype",
                 "unrelated": "meaningless"}
    cases: List[Dict] = []
    skipped = 0
    for item in data["data"]["intrasentence"]:
        if item["bias_type"] != axis:
            continue
        context = item["context"]
        # Pad the substitution so a blank glued to punctuation ("BLANK.")
        # tokenizes as ["[MASK]", "."] instead of being skipped. The fill is
        # still recovered from the original context, so nothing else shifts.
        tokens = context.replace("BLANK", " [MASK] ").split()
        if context.count("BLANK") != 1 or tokens.count("[MASK]") != 1:
            skipped += 1
            continue
        prefix, suffix = context.split("BLANK")
        case: Dict[str, Any] = {"context": tokens}
        for sentence in item["sentences"]:
            text = sentence["sentence"]
            if not (text.startswith(prefix) and text.endswith(suffix)):
                break
            fill = text[len(prefix): len(text) - len(suffix)] if suffix else text[len(prefix):]
            if len(fill.split()) != 1:
                break
            case[label_key[sentence["gold_label"]]] = fill.strip()
        if len(case) != 4:
            skipped += 1
            continue
        cases.append(case)
    if limit is not None:
        cases = cases[:limit]
    return cases, skipped


def _build_stereoset(backend, metrics, axis, limit, root, allowed) -> Tuple[Dict, Dict]:
    cases, skipped = _stereoset_cases(root, axis, limit)
    if not cases:
        raise ValueError(
            f"stereoset has no usable intrasentence items for axis {axis!r}; "
            f"available axes: gender, race, religion, profession"
        )
    inputs = {
        name: {"__init__": _init_kwargs(name, backend, allowed), "test_cases": cases}
        for name in metrics
    }
    path = root / _STEREOSET_DEV
    provenance = {
        "source": str(path),
        "sha256": _sha256(path),
        "test_cases": len(cases),
        "skipped_multi_word_or_glued_blank": skipped,
        "axis": axis,
        "note": "intrasentence split; BLANK rendered as [MASK], fills recovered by diff",
    }
    return inputs, provenance


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
    "stereoset": _build_stereoset,
    "weat": lambda b, m, a, lim, r, al: _build_association(
        b, m, a, lim, r, al, _WEAT_BY_AXIS, "weat"
    ),
    "seat": lambda b, m, a, lim, r, al: _build_association(
        b, m, a, lim, r, al, _SEAT_BY_AXIS, "seat"
    ),
}


# Providers that generate with the model under evaluation live in
# datasets_generated.py (module size). Same table, merged explicitly here.
DATASETS.update(GENERATED_DATASETS)
_BUILDERS.update(GENERATED_BUILDERS)
DATASETS.update(TOXICITY_DATASETS)
_BUILDERS.update(TOXICITY_BUILDERS)
DATASETS.update(CEAT_DATASETS)
_BUILDERS.update(CEAT_BUILDERS)


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
    missing_access = [mode for mode in spec.requires_access if mode not in backend.access]
    if missing_access:
        raise ValueError(
            f"dataset {dataset!r} needs backend access {list(spec.requires_access)} "
            f"but {backend.model_id} provides {list(backend.access)}; it is missing "
            f"{missing_access}. This dataset generates text with the model under "
            f"evaluation, which an encoder cannot do."
        )
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
