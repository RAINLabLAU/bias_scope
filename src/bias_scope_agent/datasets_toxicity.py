"""The RealToxicityPrompts provider for EMT (Gehman et al. 2020).

Gehman et al. draw K=25 nucleus-sampled continuations per prompt and score
each with the Perspective API; EMT is the mean over prompts of the maximum
toxicity among the K. The prompts here are the authors' own release on the
Hugging Face Hub at a pinned revision. The scorer is not Perspective: no key
is available, so a local classifier stands in, and the substitution is
written into the result's protocol as a deviation (see `_DEVIATION`). EMT's
class-level fidelity is `faithful`; a run made this way is an adaptation, and
the report says so next to the score.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from bias_scope_agent.datasets_common import DatasetSpec, _init_kwargs, access_note, generate_for

_RTP_DATASET = "allenai/real-toxicity-prompts"
_RTP_REVISION = "f21629712ffd6a3d13a54fd2807ccd521c55ef74"
_RTP_K = 25  # Gehman et al. 2020, Sec. 3.2
_RTP_DECODING = {"max_new_tokens": 20, "do_sample": True, "top_p": 0.9}  # Sec. 4, nucleus p=0.9
_RTP_DEFAULT_LIMIT = 25  # prompts; K=25 continuations each
_TOXICITY_CLASSIFIER = "unitary/toxic-bert"
_TOXICITY_CLASSIFIER_REVISION = "4d6c22e74ba2fdd26bc4f7238f50766b045a0d94"
_SEED = 42
_DEVIATION = (
    f"toxicity scored by the local classifier {_TOXICITY_CLASSIFIER}, not the Perspective "
    "API Gehman et al. used; EMT values are not comparable to the paper's"
)

TOXICITY_DATASETS: Dict[str, DatasetSpec] = {
    "rtp_toxicity": DatasetSpec(
        name="rtp_toxicity",
        description=(
            "Gehman et al. 2020 RealToxicityPrompts, the authors' release on the "
            "Hugging Face Hub at a pinned revision, prompts in file order. K=25 "
            "nucleus-sampled continuations per prompt from the model under "
            "evaluation, each scored for toxicity. SCORED BY A LOCAL CLASSIFIER "
            f"({_TOXICITY_CLASSIFIER}), not the Perspective API the paper uses, "
            "which the result's protocol records as a deviation. Toxicity is "
            "measured irrespective of any demographic axis, so this dataset "
            "applies under every axis (axes: any). 25 generations per prompt "
            "plus classification: keep `limit` modest."
        ),
        metrics=("EMT",),
        axes=("any",),
        source=f"hf://datasets/{_RTP_DATASET}@{_RTP_REVISION[:12]}",
        init_from_backend=(),
        requires_access=("completions",),
    ),
}


def _rtp_prompts(limit: Optional[int]) -> Tuple[str, List[str]]:
    """(revision, prompt texts) from the pinned Hub release, file order."""
    from datasets import load_dataset

    rows = load_dataset(_RTP_DATASET, split="train", revision=_RTP_REVISION)
    take = _RTP_DEFAULT_LIMIT if limit is None else limit
    return _RTP_REVISION, [row["prompt"]["text"] for row in rows.select(range(take))]


def _build_classifier():
    """The local toxicity classifier, on the GPU when there is one."""
    import torch
    from transformers import pipeline

    return pipeline(
        "text-classification",
        model=_TOXICITY_CLASSIFIER,
        revision=_TOXICITY_CLASSIFIER_REVISION,
        top_k=None,
        truncation=True,
        device=0 if torch.cuda.is_available() else -1,
    )


@lru_cache(maxsize=1)
def _classifier():
    # Built once per process: RealToxicityPrompts scores one text at a time,
    # hundreds of times, and a pipeline per call would take minutes.
    return _build_classifier()


def _toxicity_scores(texts: Sequence[str]) -> List[float]:
    """P(toxic) per text from the local classifier's `toxic` head."""
    outputs = _classifier()([text if text else " " for text in texts], batch_size=32)
    return [
        float(next(item["score"] for item in scores if item["label"] == "toxic"))
        for scores in outputs
    ]


def toxicity_scorer() -> Callable[[str], float]:
    """A single-text scorer for metrics that take one (RealToxicityPrompts)."""
    return lambda text: _toxicity_scores([text])[0]


def _build_rtp_toxicity(backend, metrics, axis, limit, root, allowed) -> Tuple[Dict, Dict]:
    revision, prompts = _rtp_prompts(limit)
    repeated = [prompt for prompt in prompts for _ in range(_RTP_K)]
    texts = generate_for(backend, repeated, _RTP_DECODING, "rtp_toxicity", _SEED)
    scores = _toxicity_scores(texts)
    groups = [slice(i * _RTP_K, (i + 1) * _RTP_K) for i in range(len(prompts))]
    protocol = {
        "dataset": _RTP_DATASET,
        "dataset_revision": revision,
        "resources": [{
            "name": "toxicity_classifier",
            "source": f"{_TOXICITY_CLASSIFIER}@{_TOXICITY_CLASSIFIER_REVISION}",
            "deviation": _DEVIATION,
        }],
    }
    inputs = {
        name: {
            "__init__": _init_kwargs(name, backend, allowed),
            "completions": [texts[g] for g in groups],
            "toxicity_scores": [scores[g] for g in groups],
            "__protocol__": protocol,
        }
        for name in metrics
    }
    provenance = {
        "source": f"hf://datasets/{_RTP_DATASET}",
        "dataset_revision": revision,
        "prompts": len(prompts),
        "k": _RTP_K,
        "generated_by": backend.model_id,
        "access_mode": access_note(backend),
        "decoding": _RTP_DECODING,
        "seed": _SEED,
        "toxicity_classifier": _TOXICITY_CLASSIFIER,
        "classifier_revision": _TOXICITY_CLASSIFIER_REVISION,
        "deviation": _DEVIATION,
        "note": "prompts in the release's own order; Gehman et al. sample 10K stratified "
                "by prompt toxicity",
    }
    return inputs, provenance


TOXICITY_BUILDERS: Dict[str, Callable[..., Tuple[Dict, Dict]]] = {
    "rtp_toxicity": _build_rtp_toxicity,
}
