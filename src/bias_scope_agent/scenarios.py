"""The scripted evaluation: three user turns per kind of target model.

Used by scripts/agent/live_conversation.py (recorded, comparable runs) and by
the autonomous mode of the terminal UI, which asks for a model id and nothing
else - so the kind of target is inferred here the way `inspect_model` does it
and turned into the same three turns. Confirmation of the plan is the third
turn: in these modes it is given on the user's behalf.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from bias_scope_agent.introspection import inspect_model

SCENARIOS: Dict[str, Dict[str, str]] = {
    "encoder": {
        "model_id": "bert-base-uncased",
        "backend_kind": "encoder",
        "dtype": "fp32",
        "described_as": "a BERT masked language model",
    },
    "causal": {
        "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "backend_kind": "causal",
        "dtype": "bf16",
        "described_as": "an instruction-tuned decoder-only (causal) LM",
    },
    "embedding": {
        "model_id": "sentence-transformers/all-MiniLM-L6-v2",
        "backend_kind": "encoder",
        "dtype": "fp32",
        "described_as": "a sentence-embedding model",
    },
    # A target served by an API through litellm. The `openrouter/` prefix
    # makes litellm read OPENROUTER_API_KEY itself; no key enters the
    # conversation. Only completions/chat access, so the generation-based
    # providers are what can be fed.
    "api": {
        "model_id": "openrouter/meta-llama/llama-3.1-8b-instruct",
        "backend_kind": "litellm",
        "dtype": "api",
        "described_as": "a chat model served through OpenRouter",
    },
}


def scenario_turns(scenario: str, device: str) -> List[str]:
    """Three turns: what can run, plan it, run it and summarize."""
    spec = SCENARIOS[scenario]
    if spec["backend_kind"] == "litellm":
        first = (
            f"I want to measure gender bias in the model {spec['model_id']}. It is "
            f"{spec['described_as']}, so set it up as a litellm backend with exactly "
            f"that model_id. The OPENROUTER_API_KEY is already in my environment - do "
            f"not ask me for it. Which bias metrics can actually run on it, and which "
            f"cannot, and why?"
        )
    else:
        first = (
            f"I want to measure gender bias in the Hugging Face model "
            f"{spec['model_id']}. It is {spec['described_as']}, so set it up as a "
            f"huggingface backend of kind {spec['backend_kind']} with dtype "
            f"{spec['dtype']} on device {device} (I have a CUDA GPU). Which bias "
            f"metrics can actually run on it, and which cannot, and why?"
        )
    return [
        first,
        "Now plan an evaluation, axis gender, language en. Use the datasets "
        "this harness can load itself - check list_datasets and use "
        "prepare_inputs with each dataset's default size (do not pass a limit). "
        "Do not ask me to paste any evaluation data. Include every recommended "
        "metric you can actually feed that way. Show me the plan and the data "
        "provenance, and do not run anything yet.",
        "Yes, that plan is exactly what I want. Run it, then give me a summary "
        "of the bias results: every metric with its score, what the score "
        "means, and its fidelity label.",
    ]


def scenario_for_model(model_id: str) -> Tuple[str, Dict[str, str]]:
    """(scenario name, spec with this model) for a model the user only named.

    An id litellm recognises (or an `openrouter/` prefix) is an API target;
    a causal checkpoint is `causal`; an encoder with a masked-LM head is
    `encoder`; one without is a sentence encoder (`embedding`).
    """
    if model_id.startswith("openrouter/"):
        kind = "api"
    else:
        info = inspect_model(model_id)
        guessed, source = info.get("guessed_kind"), info.get("guessed_source")
        if source == "litellm_model_id":
            kind = "api"
        elif guessed == "causal":
            kind = "causal"
        elif guessed == "encoder":
            kind = "encoder" if info.get("has_lm_head", True) is not False else "embedding"
        else:
            raise ValueError(
                f"cannot tell what kind of model {model_id!r} is: not a Hugging Face causal "
                "or masked LM by its config, and not an id litellm recognises. Give a "
                "Hugging Face repo id or openrouter/<vendor>/<model>."
            )
    spec = dict(SCENARIOS[kind], model_id=model_id)
    return kind, spec


def turns_for_model(model_id: str, device: str) -> List[str]:
    """The three scripted turns for a model the user only named."""
    kind, spec = scenario_for_model(model_id)
    saved = SCENARIOS[kind]["model_id"]
    SCENARIOS[kind]["model_id"] = model_id
    try:
        return scenario_turns(kind, device)
    finally:
        SCENARIOS[kind]["model_id"] = saved
