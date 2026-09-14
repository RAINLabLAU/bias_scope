"""Metric-input-shape discovery and best-guess target-model inspection.

`bias_scope.metadata.MetricInfo` has no field describing what shape of
`inputs` a metric's `evaluate()` needs (confirmed by reading the class in
full) - the only source of truth is each metric's own `evaluate()` signature.
`metrics_needing_data` reads that signature directly with `inspect.signature`
rather than hand-maintaining a lookup table that would drift as metrics are
added (see REVIEW_LATER.md RL-039 for why this reimplements, rather than
imports, bias_scope.suite's private `_metric_classes()`).
"""

from __future__ import annotations

import importlib
import inspect
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from bias_scope.base import BiasMetric
from bias_scope.metadata import list_metrics
from bias_scope_agent.config import load_config

# Mirrors bias_scope.suite._metric_classes()'s module list. That helper is
# private (not in __all__), so bias_scope_agent keeps its own copy rather
# than importing an internal symbol with no deprecation path. See RL-039.
_FAMILY_MODULES = (
    "bias_scope.embeddings_based",
    "bias_scope.probability_based",
    "bias_scope.generated_text_based",
    "bias_scope.prompts_based",
)

_UNIMPORTABLE = "<metric class not importable>"


def _agent_metric_classes() -> Dict[str, type]:
    """Every importable metric class, by name. See module docstring."""
    classes: Dict[str, type] = {}
    for module_name in _FAMILY_MODULES:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        for name in list_metrics():
            candidate = getattr(module, name, None)
            if isinstance(candidate, type) and issubclass(candidate, BiasMetric):
                classes[name] = candidate
    return classes


def _required_params(cls: type) -> List[str]:
    signature = inspect.signature(cls.evaluate)
    required = []
    for param_name, param in signature.parameters.items():
        if param_name == "self":
            continue
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        if param.default is param.empty:
            required.append(param_name)
    return required


def metrics_needing_data(metric_names: Sequence[str]) -> Dict[str, List[str]]:
    """metric name -> evaluate() params with no default (excl. self/*args/**kwargs).

    An empty list means the metric needs nothing from the caller (it loads
    its own dataset, e.g. BBQMetric). A metric that can't be imported is
    reported conservatively as needing data, since it can't be introspected.
    """
    classes = _agent_metric_classes()
    result: Dict[str, List[str]] = {}
    for name in metric_names:
        cls = classes.get(name)
        if cls is None:
            result[name] = [_UNIMPORTABLE]
        else:
            result[name] = _required_params(cls)
    return result


# Fallback classification by model_type, used only when the architecture
# string itself doesn't say (e.g. a minimal config that names model_type but
# not architectures). Not exhaustive - a modest list of common open-weight
# families, meant to widen the common case, not replace the primary
# architecture-string check. See REVIEW_LATER.md RL-044.
_ENCODER_MODEL_TYPES = {
    "bert", "roberta", "distilbert", "albert", "electra", "deberta",
    "deberta-v2", "xlm-roberta", "camembert", "xlnet", "mpnet", "funnel",
}
_CAUSAL_MODEL_TYPES = {
    "gpt2", "gpt_neo", "gptj", "gpt_neox", "llama", "mistral", "mixtral",
    "falcon", "gemma", "gemma2", "qwen2", "phi", "phi3", "bloom", "opt",
    "codegen", "starcoder2", "olmo",
}
# HuggingFaceBackend only accepts kind="causal" or "encoder" (ValueError
# otherwise) - a seq2seq/encoder-decoder model is not usable there at all,
# so this is worth naming explicitly rather than leaving it an unexplained
# low-confidence guess.
_SEQ2SEQ_MODEL_TYPES = {
    "t5", "mt5", "bart", "pegasus", "marian", "led", "longt5", "mbart",
    "fsmt", "prophetnet", "blenderbot",
}


def _guess_kind_from_config(
    config: Dict[str, Any],
) -> tuple[Optional[str], Optional[bool], Optional[str]]:
    """Returns (kind, has_lm_head, unsupported_reason)."""
    architectures = config.get("architectures") or []
    model_type = config.get("model_type", "")

    if (
        any("ConditionalGeneration" in a or "Seq2SeqLM" in a for a in architectures)
        or model_type in _SEQ2SEQ_MODEL_TYPES
    ):
        return None, None, (
            f"this looks like an encoder-decoder/seq2seq model (model_type={model_type!r}); "
            "bias_scope's HuggingFaceBackend only supports kind='causal' or 'encoder', not "
            "sequence-to-sequence - it cannot be used as a local backend here"
        )

    if any("CausalLM" in a or "LMHeadModel" in a for a in architectures):
        return "causal", True, None
    if any("MaskedLM" in a for a in architectures):
        return "encoder", True, None
    if model_type in _CAUSAL_MODEL_TYPES:
        return "causal", None, None
    if model_type in _ENCODER_MODEL_TYPES:
        return "encoder", None, None
    if architectures:
        return "encoder", False, None
    return None, None, None


def _download_hf_config(identifier: str) -> Dict[str, Any]:
    """Fetch config.json from the Hugging Face Hub. Isolated for mocking."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(identifier, "config.json")
    return json.loads(Path(path).read_text())


def _probe_completion_api(
    identifier: str, api_key: Optional[str], api_base: Optional[str]
) -> Dict[str, Any]:
    """One minimal completion call to see how the endpoint shapes a response."""
    import litellm

    response = litellm.completion(
        model=identifier,
        messages=[{"role": "user", "content": "hi"}],
        api_key=api_key,
        api_base=api_base,
        max_tokens=1,
    )
    if hasattr(response, "model_dump"):
        return response.model_dump()
    return dict(response)


def _looks_like_api_endpoint(identifier: str, api_base: Optional[str]) -> bool:
    return bool(api_base) or "://" in identifier


def _litellm_model_hint(identifier: str) -> tuple[Optional[str], List[str]]:
    """Fully offline check against litellm's bundled model registry (no
    network call - just a local list litellm ships). Returns "high" when
    `identifier` is an exact, known litellm model string (so callers can
    skip a pointless HF Hub lookup for something like "gpt-4o-mini"), or
    None with a "did you mean" note for a close-but-not-exact match.
    """
    try:
        import litellm
    except ImportError:
        return None, []
    model_list = litellm.model_list
    if identifier in model_list:
        return "high", [f"{identifier!r} matches a known litellm model string exactly"]
    import difflib

    close = difflib.get_close_matches(identifier, model_list, n=3, cutoff=0.6)
    if close:
        return None, [f"not an exact litellm model match; did you mean: {', '.join(close)}?"]
    return None, []


def _read_local_config(path: Path) -> tuple[Dict[str, Any], List[str]]:
    config_path = path / "config.json"
    if not config_path.exists():
        return {}, [f"config.json not found at {config_path}"]
    try:
        return json.loads(config_path.read_text()), []
    except (OSError, json.JSONDecodeError) as exc:
        return {}, [f"could not read {config_path}: {exc}"]


def _new_inspection_result(identifier: str) -> Dict[str, Any]:
    return {
        "identifier": identifier,
        "guessed_source": "unknown",
        "guessed_kind": None,
        "has_lm_head": None,
        "chat_formatted": None,
        "supports_logprobs": None,
        "confidence": "low",
        "notes": [],
        "raw": {},
    }


def _apply_kind_guess(result: Dict[str, Any], config: Dict[str, Any]) -> None:
    kind, has_lm_head, unsupported_reason = _guess_kind_from_config(config)
    result["raw"] = config
    result["guessed_kind"], result["has_lm_head"] = kind, has_lm_head
    if unsupported_reason:
        result["notes"].append(unsupported_reason)
    result["confidence"] = "high" if kind is not None else "low"


def _inspect_api_endpoint(
    result: Dict[str, Any],
    identifier: str,
    api_key: Optional[str],
    api_base: Optional[str],
    live: bool,
) -> Dict[str, Any]:
    result["guessed_source"] = "api_endpoint"
    if not live:
        result["notes"].append("live inspection disabled; pass live=True to probe the endpoint")
        return result
    try:
        response = _probe_completion_api(identifier, api_key, api_base)
    except Exception as exc:  # noqa: BLE001 - never let inspection raise
        result["notes"].append(str(exc))
        return result
    result["raw"] = response
    result["chat_formatted"] = "choices" in response and bool(response.get("choices"))
    result["supports_logprobs"] = any(
        choice.get("logprobs") is not None for choice in response.get("choices", [])
    )
    result["confidence"] = "high"
    return result


def _inspect_local_path(result: Dict[str, Any], local_path: Path) -> Dict[str, Any]:
    result["guessed_source"] = "local_path"
    config, notes = _read_local_config(local_path)
    result["notes"].extend(notes)
    _apply_kind_guess(result, config)
    return result


def _inspect_hf_hub(
    result: Dict[str, Any], identifier: str, live: bool, litellm_notes: List[str]
) -> Dict[str, Any]:
    result["guessed_source"] = "hf_hub"
    result["notes"].extend(litellm_notes)
    if not live:
        result["notes"].append("live inspection disabled; pass live=True to fetch config.json")
        return result
    try:
        config = _download_hf_config(identifier)
    except Exception as exc:  # noqa: BLE001 - never let inspection raise
        result["notes"].append(str(exc))
        return result
    _apply_kind_guess(result, config)
    return result


def inspect_model(
    identifier: str,
    *,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    live: Optional[bool] = None,
) -> Dict[str, Any]:
    """Best-guess dict about a target model identifier. Never a final decision.

    Never raises: any failure to inspect degrades to low confidence with a
    note explaining why, so the calling agent still gets a dict to reason
    about and must confirm uncertain guesses with the user.
    """
    if live is None:
        live = load_config().inspect_model_live
    result = _new_inspection_result(identifier)

    if _looks_like_api_endpoint(identifier, api_base):
        return _inspect_api_endpoint(result, identifier, api_key, api_base, live)

    local_path = Path(identifier)
    if identifier and local_path.exists():
        return _inspect_local_path(result, local_path)

    # Not a local path or API-endpoint-looking string; it may be a HF Hub id,
    # or it may be a known litellm model string (e.g. "gpt-4o-mini") that
    # would only ever fail a HF Hub lookup. This check is offline and free,
    # so it runs regardless of `live`.
    litellm_confidence, litellm_notes = _litellm_model_hint(identifier)
    if litellm_confidence == "high":
        result["guessed_source"] = "litellm_model_id"
        result["confidence"] = "high"
        result["notes"].extend(litellm_notes)
        result["notes"].append(
            "use construct_backend(kind='litellm', model_id=...) for this identifier"
        )
        return result

    return _inspect_hf_hub(result, identifier, live, litellm_notes)
