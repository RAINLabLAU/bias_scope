"""Providers for the prompt family, for a target served through a chat API.

Three shapes of provider live here:

* `prompt_benchmarks` - metrics that load their own benchmark and call the
  model themselves (BBQ, StereoSet as a prompt task, identity swaps,
  WinoBias pronoun skew). They need the backend's model name and a bounded
  size; the harness supplies both and nothing else.
* `winobias_coref`, `decodingtrust_stereotype` - metrics that take model
  answers to the authors' items. The harness loads the authors' files and
  has the backend answer each item; the callables it builds never leave
  the process.
* `rtp_prompt_runner` - RealToxicityPrompts, handed the same local toxicity
  scorer EMT uses, as a recorded deviation from the Perspective API.

Only a backend with `chat` access is offered these: the metrics drive the
model through litellm by name, which a local checkpoint cannot satisfy.
"""

from __future__ import annotations

import csv
from typing import Any, Callable, Dict, Sequence, Tuple

from bias_scope_agent.datasets_common import DatasetSpec, _require, _sha256, access_note

_WINO_DIR = "corefBias/WinoBias/wino/data"
_DT_DIR = "DecodingTrust/data/stereotype/dataset"
_CHAT = ("chat",)
_SEED = 42

# Each self-loading metric: the benchmark it loads, its own subset name per
# axis, and the size arguments that keep a run to a few hundred requests.
_BBQ_SUBSET = {"gender": "Gender_identity", "race": "Race_ethnicity", "religion": "Religion",
               "age": "Age", "nationality": "Nationality", "disability": "Disability_status"}
_SELF_LOADING: Dict[str, Dict[str, Any]] = {
    "BBQMetric": {
        "benchmark": "Elfsong/BBQ (Parrish et al. 2022), ambiguous contexts",
        "subset_by_axis": _BBQ_SUBSET,
        "size": {"num_samples": 200},
    },
    "StereoSetMetric": {
        "benchmark": "StereoSet intersentence, validation split (Nadeem et al. 2021)",
        "subset_by_axis": {a: a for a in ("gender", "race", "religion", "profession")},
        "size": {"num_samples": 200},
    },
    "IdentitySwapConsistency": {
        "benchmark": "StereoSet contexts with the identity term swapped (IDENTITY_SWAP_PAIRS "
                     "covers race and religion terms)",
        "subset_by_axis": {"race": "race", "religion": "religion"},
        "size": {"num_samples": 100},
    },
    "OccupationPronounSkew": {
        "benchmark": "uclanlp/wino_bias type1_pro templates (Zhao et al. 2018)",
        "subset_by_axis": {"gender": None},  # no subset argument; gender is the only axis
        "size": {"num_templates": 20, "num_samples": 10},
    },
}

PROMPT_DATASETS: Dict[str, DatasetSpec] = {
    "prompt_benchmarks": DatasetSpec(
        name="prompt_benchmarks",
        description=(
            "Prompt-family metrics that load their own benchmark and query the "
            "model themselves: BBQMetric, StereoSetMetric, IdentitySwapConsistency, "
            "OccupationPronounSkew. The harness supplies the backend's model name, "
            "the benchmark subset for the axis, and a bounded size (200 BBQ items, "
            "200 StereoSet items, 100 identity-swap pairs, 20 WinoBias templates x 10 "
            "samples). Chat-API backends only. A metric that does not cover the "
            "axis is refused by name."
        ),
        metrics=tuple(_SELF_LOADING),
        axes=("gender", "race", "religion", "age", "profession"),
        source="each metric's own loader (Hugging Face Hub datasets)",
        init_from_backend=("model_name",),
        requires_access=_CHAT,
    ),
    "winobias_coref": DatasetSpec(
        name="winobias_coref",
        description=(
            "Zhao et al. 2018 WinoBias, the authors' type-1 pro- and anti-stereotypical "
            "test files, paired one to one. The harness asks the model under evaluation "
            "each coreference question and maps its reply to one of the two options. "
            "Default 100 pairs (200 requests). Chat-API backends only."
        ),
        metrics=("WinoBias",),
        axes=("gender",),
        source=f"{_WINO_DIR}/{{pro,anti}}_stereotyped_type1.txt.test",
        init_from_backend=("model_name",),
        requires_access=_CHAT,
    ),
    "decodingtrust_stereotype": DatasetSpec(
        name="decodingtrust_stereotype",
        description=(
            "Wang et al. 2023 DecodingTrust stereotype statements, the authors' "
            "user_prompts.csv, under the 'benign' system prompt. The model under "
            "evaluation answers each statement; the metric reads agree/disagree. "
            "Default 120 prompts in file order. Chat-API backends only."
        ),
        metrics=("DecodingTrustStereotype",),
        axes=("gender", "race", "religion", "age", "any"),
        source=f"{_DT_DIR}/user_prompts.csv + system_prompts.csv",
        init_from_backend=("model_name",),
        requires_access=_CHAT,
    ),
    "rtp_prompt_runner": DatasetSpec(
        name="rtp_prompt_runner",
        description=(
            "RealToxicityPrompts as the prompt-family metric runs it: it loads the "
            "prompts and generates itself. Toxicity is SCORED BY A LOCAL CLASSIFIER "
            "(unitary/toxic-bert), not the Perspective API, recorded as a deviation. "
            "Default 10 prompts x 25 generations. Chat-API backends only."
        ),
        metrics=("RealToxicityPrompts",),
        axes=("any",),
        source="hf://datasets/allenai/real-toxicity-prompts, loaded by the metric",
        init_from_backend=("model_name",),
        requires_access=_CHAT,
    ),
}


def _model_init(backend, allowed: Tuple[str, ...]) -> Dict[str, Any]:
    return {"model_name": backend.model_id} if "model_name" in allowed else {}


def _build_prompt_benchmarks(backend, metrics, axis, limit, root, allowed) -> Tuple[Dict, Dict]:
    inputs: Dict[str, Dict[str, Any]] = {}
    benchmarks: Dict[str, str] = {}
    for name in metrics:
        spec = _SELF_LOADING[name]
        if axis not in spec["subset_by_axis"]:
            raise ValueError(
                f"{name} does not cover axis {axis!r}; it covers "
                f"{sorted(spec['subset_by_axis'])}. Drop it from this axis's plan."
            )
        block: Dict[str, Any] = {"__init__": _model_init(backend, allowed), **spec["size"]}
        subset = spec["subset_by_axis"][axis]
        if subset is not None:
            block["subset"] = subset
        inputs[name] = block
        benchmarks[name] = spec["benchmark"]
    provenance = {
        "source": "each metric's own loader",
        "benchmarks": benchmarks,
        "sizes": {name: _SELF_LOADING[name]["size"] for name in metrics},
        "model": backend.model_id,
        "access_mode": access_note(backend),
        "axis": axis,
        "note": "these metrics load their benchmark and call the model themselves; the "
                "harness only names the model, the subset and the size",
    }
    return inputs, provenance


def _winobias_answer_fn(backend) -> Callable[[str, Sequence[str]], str]:
    """Ask the backend, then map its reply to one of the offered options."""

    def answer(prompt: str, choices: Sequence[str]) -> str:
        reply = backend.generate([prompt], max_new_tokens=20, do_sample=False)[0]
        lowered = reply.lower()
        hits = [c for c in choices if c.lower() in lowered]
        return hits[0] if len(hits) == 1 else reply

    return answer


_WINO_DEFAULT_PAIRS = 100


def _build_winobias(backend, metrics, axis, limit, root, allowed) -> Tuple[Dict, Dict]:
    from bias_scope.prompts_based.winobias import load_winobias_file

    pro_path = _require(root / _WINO_DIR / "pro_stereotyped_type1.txt.test", "WinoBias")
    anti_path = _require(root / _WINO_DIR / "anti_stereotyped_type1.txt.test", "WinoBias")
    take = _WINO_DEFAULT_PAIRS if limit is None else limit
    pro = load_winobias_file(pro_path)[:take]
    anti = load_winobias_file(anti_path)[:take]
    n = min(len(pro), len(anti))
    inputs = {
        name: {
            "__init__": _model_init(backend, allowed),
            "pro_items": pro[:n],
            "anti_items": anti[:n],
            "answer_fn": _winobias_answer_fn(backend),
            "sentence_type": 1,
        }
        for name in metrics
    }
    provenance = {
        "source": [str(pro_path), str(anti_path)],
        "sha256": [_sha256(pro_path), _sha256(anti_path)],
        "pairs": n,
        "sentence_type": 1,
        "answered_by": backend.model_id,
        "access_mode": access_note(backend),
        "decoding": {"max_new_tokens": 20, "do_sample": False},
        "note": "the model's reply is mapped to the option whose text it contains; an "
                "ambiguous reply is scored as given",
    }
    return inputs, provenance


_DT_DEFAULT_PROMPTS = 120


def _build_decodingtrust(backend, metrics, axis, limit, root, allowed) -> Tuple[Dict, Dict]:
    from bias_scope_agent.datasets_common import generate_for

    users = _require(root / _DT_DIR / "user_prompts.csv", "DecodingTrustStereotype")
    systems = _require(root / _DT_DIR / "system_prompts.csv", "DecodingTrustStereotype")
    with users.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    take = _DT_DEFAULT_PROMPTS if limit is None else limit
    rows = rows[:take]
    with systems.open(encoding="utf-8", newline="") as handle:
        benign = next(r["system_prompt"] for r in csv.DictReader(handle) if r["type"] == "benign")
    decoding = {"max_new_tokens": 60, "do_sample": False}
    replies = generate_for(backend, [r["user_prompt"] for r in rows], decoding,
                           "decodingtrust_stereotype", _SEED)
    generations = [
        {"response": reply, "target_group": r["target_group"], "topic": r["stereotype_topic"],
         "scenario": "benign"}
        for r, reply in zip(rows, replies)
    ]
    inputs = {
        name: {"__init__": _model_init(backend, allowed), "generations": generations}
        for name in metrics
    }
    provenance = {
        "source": [str(users), str(systems)],
        "sha256": [_sha256(users), _sha256(systems)],
        "prompts": len(rows),
        "scenario": "benign",
        "generated_by": backend.model_id,
        "access_mode": access_note(backend),
        "decoding": decoding,
        "seed": _SEED,
        "note": f"the benign system prompt ({benign!r}) is the API's default assistant role; "
                "it is not sent explicitly. The targeted/untargeted scenarios are not run.",
    }
    return inputs, provenance


def _build_rtp_runner(backend, metrics, axis, limit, root, allowed) -> Tuple[Dict, Dict]:
    from bias_scope_agent import datasets_toxicity as tox

    scorer_name = f"{tox._TOXICITY_CLASSIFIER}@{tox._TOXICITY_CLASSIFIER_REVISION[:12]}"
    num_prompts = 10 if limit is None else limit
    protocol = {"resources": [{"name": "toxicity_classifier", "source": scorer_name,
                               "deviation": tox._DEVIATION}]}
    inputs = {
        name: {
            "__init__": {**_model_init(backend, allowed), "scorer": tox.toxicity_scorer(),
                         "scorer_name": scorer_name},
            "num_prompts": num_prompts,
            "num_generations": 25,
            "__protocol__": protocol,
        }
        for name in metrics
    }
    provenance = {
        "source": "hf://datasets/allenai/real-toxicity-prompts (loaded by the metric)",
        "prompts": num_prompts,
        "generations_per_prompt": 25,
        "generated_by": backend.model_id,
        "access_mode": access_note(backend),
        "toxicity_classifier": scorer_name,
        "deviation": tox._DEVIATION,
    }
    return inputs, provenance


PROMPT_BUILDERS: Dict[str, Callable[..., Tuple[Dict, Dict]]] = {
    "prompt_benchmarks": _build_prompt_benchmarks,
    "winobias_coref": _build_winobias,
    "decodingtrust_stereotype": _build_decodingtrust,
    "rtp_prompt_runner": _build_rtp_runner,
}
