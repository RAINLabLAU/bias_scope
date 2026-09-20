"""Primitives shared by the dataset providers (datasets.py, datasets_generated.py).

Kept apart so the provider modules can each stay under PLAN.md's 400-line
limit and import these without importing each other.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from bias_scope.backends import Backend, cached_generate

_THIRD_PARTY = Path("third_party/code")
_GENERATION_CACHE = Path("cache/generations")
_SENT_BIAS_TESTS = "sent-bias/tests"

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
    #: Access modes the backend must provide. A provider that generates needs
    #: "completions"; the rest need nothing the planner has not already checked.
    requires_access: Tuple[str, ...] = ()



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



def generate_for(
    backend: Backend, prompts: Sequence[str], decoding: Dict[str, Any], dataset: str, seed: int
) -> List[str]:
    """One continuation per prompt from the model under evaluation, cached.

    Cached under cache/generations/<model>/<dataset>/ keyed by the protocol
    hash (PLAN.md Section 1), so re-running a provider on the same model and
    decoding regenerates nothing. Seeded first: sampled decoding is otherwise
    irreproducible, and the seed is recorded in the provenance by the caller.
    """
    from bias_scope.utils import seed_everything

    seed_everything(seed)
    safe_model = re.sub(r"[^A-Za-z0-9_.-]", "_", backend.model_id)
    return cached_generate(
        backend, list(prompts), decoding, cache_dir=_GENERATION_CACHE / safe_model / dataset
    )


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


def access_note(backend: Backend) -> str:
    """How the model under evaluation produced its text, for the provenance.

    A chat API answers a prompt as a message rather than continuing it, which
    is a further access-mode adaptation of continuation-style protocols
    (BOLD, HONEST, RealToxicityPrompts) on top of the causal-LM one.
    """
    if "chat" in getattr(backend, "access", ()):
        return ("chat API: each prompt sent as a user message; the model answers, it does "
                "not continue")
    return "local causal LM: each prompt continued token by token"

