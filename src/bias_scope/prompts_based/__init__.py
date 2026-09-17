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
    "FirstPersonFairness": "bias_scope.prompts_based.first_person_fairness",
    "IdentitySwapConsistency": "bias_scope.prompts_based.identity_swap_consistency",
    "OccupationPronounSkew": "bias_scope.prompts_based.occupation_pronoun_skew",
    "WinoBias": "bias_scope.prompts_based.winobias",
    "ImplicitAssociationTest": "bias_scope.prompts_based.implicit_association",
    "LLMDecisionBias": "bias_scope.prompts_based.implicit_association",
    "DiscrimEval": "bias_scope.prompts_based.discrim_eval",
    "DecodingTrustStereotype": "bias_scope.prompts_based.decodingtrust",
    "DecodingTrustFairness": "bias_scope.prompts_based.decodingtrust",
    "PoliticalEvenHandedness": "bias_scope.prompts_based.political_even_handedness",
    "OpinionConsistencyAcrossPersonas": (
        "bias_scope.prompts_based.opinion_consistency_across_personas"
    ),
    "RealToxicityPrompts": "bias_scope.prompts_based.realtoxicityprompts",
    "StereoSetMetric": "bias_scope.prompts_based.stereoset",
    "TofNof": "bias_scope.prompts_based.tof_nof",
    "TrustLLMStereotypeRecognition": "bias_scope.prompts_based.trustllm",
    "TrustLLMStereotypeAgreement": "bias_scope.prompts_based.trustllm",
    "TrustLLMDisparagement": "bias_scope.prompts_based.trustllm",
    "TrustLLMPreference": "bias_scope.prompts_based.trustllm",
    "TruthfulQA": "bias_scope.prompts_based.truthfulqa",
    "UnQoverMetric": "bias_scope.prompts_based.unqover",
}


#: Every prompt metric this package can provide, installed or not. A metric
#: whose optional dependency is missing never reaches `register()`, so it is
#: absent from `list_metrics()` and indistinguishable from a typo; this is the
#: source of truth for telling those two apart (REVIEW_LATER RL-047).
PROMPT_METRIC_NAMES = tuple(sorted(_PROMPT_EXPORTS))


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


#: Renamed in 0.2.0. The old name stays importable and raises a
#: DeprecationWarning (PLAN.md Section 1). This is a pure rename — the
#: statistic is unchanged — so the alias is safe; contrast LPBS, where the old
#: name now means a different (correct) metric and no alias exists.
_RENAMED = {
    "DemographicRepresentationBias": "OccupationPronounSkew",
    "CounterfactualFairness": "IdentitySwapConsistency",
}


def __getattr__(name: str):
    if name in _RENAMED:
        from bias_scope._deprecation import deprecated_alias

        value = deprecated_alias(__getattr__(_RENAMED[name]), name)
        globals()[name] = value
        return value

    if name not in _PROMPT_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    try:
        value = getattr(import_module(_PROMPT_EXPORTS[name]), name)
    except ImportError as exc:
        value = _optional_prompt_dependency_stub(name, exc)
    globals()[name] = value
    return value


__all__ = [
    "PROMPT_METRIC_NAMES",
    "AnalogicalReasoningBias",
    "BBQMetric",
    "BOLD",
    "FirstPersonFairness",
    "IdentitySwapConsistency",
    "OccupationPronounSkew",
    "OpinionConsistencyAcrossPersonas",
    "RealToxicityPrompts",
    "StereoSetMetric",
    "TofNof",
    "TrustLLMStereotypeRecognition",
    "TrustLLMStereotypeAgreement",
    "TrustLLMDisparagement",
    "TrustLLMPreference",
    "TruthfulQA",
    "UnQoverMetric",
    "WinoBias",
    "ImplicitAssociationTest",
    "LLMDecisionBias",
    "DiscrimEval",
    "DecodingTrustStereotype",
    "DecodingTrustFairness",
    "PoliticalEvenHandedness",
    # Deprecated alias, removed in 0.3.0.
    "DemographicRepresentationBias",
    "CounterfactualFairness",
]