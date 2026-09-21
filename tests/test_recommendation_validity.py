"""Every metric `recommend_metrics` can return must actually be runnable.

The gate this project did not have. Its absence is why a recommendation could
name a metric that cannot complete: `CrowSPairs` on a causal LM (RL-057),
every masked-LM metric on a checkpoint with no LM head (RL-058), `CAT`/`ICAT`
with no declarable headline (RL-061), `EMT`/`GenderPolarity`/`HONEST` with no
item count (RL-063). Each was found by a live run rather than by the suite,
which is the wrong order.

What this asserts: for each backend kind, every recommended metric either
completes `run()` on minimal inputs, or appears in `KNOWN_UNRUNNABLE` with a
reason. The exception list is the point - it makes "we recommend things that
cannot run" a visible, reviewed set rather than a runtime surprise.

Inputs here are minimal synthetic stand-ins of the right *shape*. They measure
reachability, never bias: no number this file produces is a result, and none
is reported anywhere.
"""

from __future__ import annotations

import contextlib
import inspect
import io

import numpy as np
import pytest
from tests.conftest import TINY_ENCODER_ID

from bias_scope.backends import HuggingFaceBackend
from bias_scope.recommend import recommend_metrics
from bias_scope_agent.introspection import _agent_metric_classes, metrics_needing_data

# Why each metric cannot be run here. Every entry is a claim a maintainer can
# check, not a shrug - and several are tracked as their own REVIEW_LATER items.
KNOWN_UNRUNNABLE = {
    "ToxicityFraction": "constructor requires a Perspective API key; no offline path",
    "ToxicityProbability": "constructor requires a Perspective API key; no offline path",
    "MeanScoreGap": "rejects an empty api_key; needs a live classifier service",
    "MarkedPersons": "reports no scalar; PLAN.md 4.2 still asks what the reported scalar is",
    "SentenceBiasScore": "fidelity unaudited - paper paywalled with no preprint (RL-029)",
    "SocialGroupSubstitution": "requires generate_fn/score_fn callables; no stand-in supplies them",
    "CounterfactualSentimentBias": "needs paired sentiment arrays this harness cannot build",
    "PsycholinguisticNorms": "needs an NRC-VAD lexicon covering the completions",
    "StereotypeRuleHitRate": "rule objects need fields no generic stand-in can invent",
    "CBS": "templates must carry a {attr} placeholder this harness does not construct",
    "TopKFillDivergence": "template must carry a {attr} placeholder",
    "DisCoMetric": "expects a mapping of slot values, not a flat word list",
    "LPBS": "requires a fill_probabilities callable",
    "LMB": "requires equal-token-length pairs; generic pairs do not satisfy it",
    "PairwiseLikelihoodPreference": "requires equal-token-length pairs",
}

_RNG = np.random.RandomState(0)


def _embeddings(shift: float) -> np.ndarray:
    return _RNG.randn(6, 16) + shift


_TEXTS = [["a text one"], ["a text two"], ["a text three"]]
_MISSING = object()

_STAND_INS = {
    "sentence_pairs": [["a b c", "d b c"], ["e f g", "h f g"]],
    "target_embeddings": (_embeddings(1.0), _embeddings(-1.0)),
    "attribute_embeddings": (_embeddings(0.7), _embeddings(-0.7)),
    "test_cases": [
        {
            "context": "The [MASK] one",
            "stereotype": "good",
            "anti_stereotype": "bad",
            "meaningless": "cloud",
        }
    ],
    "group_a_texts": _TEXTS,
    "group_b_texts": _TEXTS,
    "generated_texts": _TEXTS,
    "completions": _TEXTS,
    "generations": ["he is a doctor", "she is a nurse"],
    "group_lexicons": {"m": ["he"], "f": ["she"]},
    "masculine_terms": ["he"],
    "feminine_terms": ["she"],
    "hurtlex": {"text"},
    "toxicity_scores": [[0.1], [0.2], [0.3]],
    "target_words": ["doctor"],
    "marked_generations": ["she is a nurse"] * 6,
    "unmarked_generations": ["he is a nurse"] * 6,
}

# CEAT (as audited 2026-09) takes, per stimulus, a matrix of that word's
# contextual token embeddings, not the flat arrays WEAT and SEAT take.
_STAND_IN_OVERRIDES = {
    "CEAT": {
        "target_embeddings": ({"x1": _embeddings(1.0), "x2": _embeddings(1.0)},
                              {"y1": _embeddings(-1.0), "y2": _embeddings(-1.0)}),
        "attribute_embeddings": ({"a1": _embeddings(0.7), "a2": _embeddings(0.7)},
                                 {"b1": _embeddings(-0.7), "b2": _embeddings(-0.7)}),
    },
}

BACKENDS = [
    ("encoder", TINY_ENCODER_ID, "encoder"),
    ("causal", "sshleifer/tiny-gpt2", "causal"),
]


def _init_kwargs(cls) -> dict:
    """Constructor arguments, or a dict carrying the reason none could be built."""
    kwargs = {}
    for param, spec in inspect.signature(cls.__init__).parameters.items():
        if param == "self" or spec.kind in (spec.VAR_POSITIONAL, spec.VAR_KEYWORD):
            continue
        if param == "model_name":
            kwargs[param] = TINY_ENCODER_ID
        elif spec.default is spec.empty:
            value = _STAND_INS.get(param, _MISSING)
            if value is _MISSING:
                return {"__reason__": f"no stand-in for constructor argument {param!r}"}
            kwargs[param] = value
    return kwargs


def _try_run(name: str) -> str:
    """"" if the metric completed run(), else a one-line reason."""
    cls = _agent_metric_classes().get(name)
    if cls is None:
        return "class not importable"
    init_kwargs = _init_kwargs(cls)
    if "__reason__" in init_kwargs:
        return init_kwargs["__reason__"]
    kwargs = {}
    for param in metrics_needing_data([name])[name]:
        if param.startswith("__init__."):
            continue
        value = _STAND_IN_OVERRIDES.get(name, {}).get(param, _STAND_INS.get(param, _MISSING))
        if value is _MISSING:
            return f"no stand-in for {param!r}"
        kwargs[param] = value
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            cls(**init_kwargs).run(**kwargs)
    except Exception as exc:  # the point of the gate is to catch every kind
        return f"{type(exc).__name__}: {str(exc).splitlines()[0][:160]}"
    return ""


def _recommended(kind: str, model_id: str):
    backend = HuggingFaceBackend(model_id, kind=kind, dtype="fp32")
    return [rec.metric for rec in recommend_metrics(access=backend.access, language="en")]


@pytest.mark.slow
@pytest.mark.parametrize("label,model_id,kind", BACKENDS)
def test_every_recommended_metric_runs_or_is_a_known_exception(label, model_id, kind):
    failures = {}
    for name in _recommended(kind, model_id):
        if name in KNOWN_UNRUNNABLE:
            continue
        reason = _try_run(name)
        if reason:
            failures[name] = reason
    assert not failures, (
        f"{label}: recommended metrics that cannot run and are not listed in "
        f"KNOWN_UNRUNNABLE: {failures}"
    )


@pytest.mark.slow
@pytest.mark.parametrize("label,model_id,kind", BACKENDS)
def test_the_exception_list_has_no_stale_entries(label, model_id, kind):
    """An entry that now runs must be removed, or the list stops meaning anything."""
    recommended = set(_recommended(kind, model_id))
    now_running = [
        name
        for name in sorted(KNOWN_UNRUNNABLE)
        if name in recommended and not _try_run(name)
    ]
    assert not now_running, (
        f"{label}: these are listed as unrunnable but now run - remove them from "
        f"KNOWN_UNRUNNABLE: {now_running}"
    )
