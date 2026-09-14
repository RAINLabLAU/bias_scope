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


def _guess_kind_from_config(config: Dict[str, Any]) -> tuple[Optional[str], Optional[bool]]:
    architectures = config.get("architectures") or []
    if any("CausalLM" in a or "LMHeadModel" in a for a in architectures):
        return "causal", True
    if any("MaskedLM" in a for a in architectures):
        return "encoder", True
    if architectures:
        return "encoder", False
    return None, None


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


def _read_local_config(path: Path) -> tuple[Dict[str, Any], List[str]]:
    config_path = path / "config.json"
    if not config_path.exists():
        return {}, [f"config.json not found at {config_path}"]
    try:
        return json.loads(config_path.read_text()), []
    except (OSError, json.JSONDecodeError) as exc:
        return {}, [f"could not read {config_path}: {exc}"]


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

    result: Dict[str, Any] = {
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

    if _looks_like_api_endpoint(identifier, api_base):
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

    local_path = Path(identifier)
    if identifier and local_path.exists():
        result["guessed_source"] = "local_path"
        config, notes = _read_local_config(local_path)
        result["notes"].extend(notes)
        result["raw"] = config
        kind, has_lm_head = _guess_kind_from_config(config)
        result["guessed_kind"], result["has_lm_head"] = kind, has_lm_head
        result["confidence"] = "high" if kind is not None else "low"
        return result

    result["guessed_source"] = "hf_hub"
    if not live:
        result["notes"].append("live inspection disabled; pass live=True to fetch config.json")
        return result
    try:
        config = _download_hf_config(identifier)
    except Exception as exc:  # noqa: BLE001 - never let inspection raise
        result["notes"].append(str(exc))
        return result
    result["raw"] = config
    kind, has_lm_head = _guess_kind_from_config(config)
    result["guessed_kind"], result["has_lm_head"] = kind, has_lm_head
    result["confidence"] = "high" if kind is not None else "low"
    return result
