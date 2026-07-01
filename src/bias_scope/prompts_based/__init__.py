"""Prompt-based bias metrics.

Prompt metrics depend on optional dataset and LLM packages, so they are loaded
on first access instead of during package import.
"""

from __future__ import annotations

from importlib import import_module


_PROMPT_EXPORTS = {
    "AnalogicalReasoningBias": "bias_scope.prompts_based.analogical_reasoning_bias",
    "BBQMetric": "bias_scope.prompts_based.bbq",
    "BOLD": "bias_scope.prompts_based.bold",
    "CounterfactualFairness": "bias_scope.prompts_based.counterfactual_fairness",
    "DemographicRepresentationBias": (
        "bias_scope.prompts_based.demographic_representation_bias"
    ),
    "OpinionConsistencyAcrossPersonas": (
        "bias_scope.prompts_based.opinion_consistency_across_personas"
    ),
    "RealToxicityPrompts": "bias_scope.prompts_based.realtoxicityprompts",
    "StereoSetMetric": "bias_scope.prompts_based.stereoset",
    "TofNof": "bias_scope.prompts_based.tof_nof",
    "TruthfulQA": "bias_scope.prompts_based.truthfulqa",
    "UnQoverMetric": "bias_scope.prompts_based.unqover",
}


def _optional_prompt_dependency_stub(class_name: str, original_error: ImportError):
    """Create a class-like placeholder for prompt metric optional extras."""

    class _MissingPromptDependency:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                f"{class_name} requires optional prompt-based dependencies. "
                "Please install bias-scope[datasets], bias-scope[llm], "
                "or bias-scope[all] to use this metric."
            ) from original_error

    _MissingPromptDependency.__name__ = class_name
    _MissingPromptDependency.__qualname__ = class_name
    _MissingPromptDependency.__module__ = __name__
    _MissingPromptDependency.__doc__ = (
        f"Placeholder for {class_name}; install prompt optional extras to use it."
    )
    return _MissingPromptDependency


def __getattr__(name: str):
    if name not in _PROMPT_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    try:
        value = getattr(import_module(_PROMPT_EXPORTS[name]), name)
    except ImportError as exc:
        value = _optional_prompt_dependency_stub(name, exc)
    globals()[name] = value
    return value


__all__ = [
    "AnalogicalReasoningBias",
    "BBQMetric",
    "BOLD",
    "CounterfactualFairness",
    "DemographicRepresentationBias",
    "OpinionConsistencyAcrossPersonas",
    "RealToxicityPrompts",
    "StereoSetMetric",
    "TofNof",
    "TruthfulQA",
    "UnQoverMetric",
]