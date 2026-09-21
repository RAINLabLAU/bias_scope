"""
bias-scope: Comprehensive bias detection for language models.

Public API for bias detection metrics organized by category:
- Embeddings: WEAT, SEAT, CEAT, SentenceBiasScore
- Probability: CrowSPairs, CAT, AUL, LPBS, CBS, DisCoMetric, ICAT, AULA, LMB
- Generated Text: ToxicityFraction, ToxicityProbability, RegardScore, ScoreParity,
                  SocialGroupSubstitution, CoOccurrenceBiasScore,
                  DemographicRepresentation, StereotypicalAssociations, MarkedPersons
- Prompt-Based: AnalogicalReasoningBias, BBQMetric, BOLD,
                CounterfactualFairness, OpinionConsistencyAcrossPersonas,
                RealToxicityPrompts, StereoSetMetric, TofNof,
                TruthfulQA, UnQoverMetric
- Utilities: to_numpy, cosine_similarity
"""

from importlib import import_module

__version__ = "0.1.0"


# Optional torch-backed metrics expose constructor stubs when torch is absent.
def _torch_dependency_stub(class_name: str, original_error: ImportError):
    """Create a class-like placeholder for optional torch-backed metrics."""

    class _MissingTorchDependency:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                f"{class_name} requires optional torch dependencies. "
                "Please install bias-scope[torch] to use this metric."
            ) from original_error

    _MissingTorchDependency.__name__ = class_name
    _MissingTorchDependency.__qualname__ = class_name
    _MissingTorchDependency.__module__ = __name__
    _MissingTorchDependency.__doc__ = (
        f"Placeholder for {class_name}; install bias-scope[torch] to use it."
    )
    return _MissingTorchDependency


try:
    from bias_scope.embeddings_based import CEAT, SEAT, WEAT, SentenceBiasScore
except ImportError as exc:
    CEAT = _torch_dependency_stub("CEAT", exc)
    SEAT = _torch_dependency_stub("SEAT", exc)
    WEAT = _torch_dependency_stub("WEAT", exc)
    SentenceBiasScore = _torch_dependency_stub("SentenceBiasScore", exc)

try:
    from bias_scope.embeddings_based import embed
except ImportError as exc:

    def embed(*args, _original_error=exc, **kwargs):
        raise ImportError(
            "embed requires optional embedding dependencies. "
            "Please install bias-scope[embeddings] to use this helper."
        ) from _original_error


# Attach MetricInfo to every metric class that imported successfully. Explicit,
# one call, in one place (PLAN.md Section 1 forbids import-side-effect
# registries). Prompt-family metrics are attached lazily by __getattr__ below,
# since they are themselves lazily imported.
from bias_scope._metric_info import METRIC_INFO
from bias_scope._metric_info import attach as _attach_metric_info
from bias_scope.backends import (
    HuggingFaceBackend,
    LiteLLMBackend,
    StubBackend,
    cached_generate,
    load_model,
)
from bias_scope.generated_text_based import (
    EMT,
    FGB,
    HONEST,
    PGB,
    CoOccurrenceBiasScore,
    CounterfactualSentimentBias,
    DemographicRepresentation,
    GenderPolarity,
    MarkedPersons,
    MeanScoreGap,
    PerspectiveAPIClient,
    PsycholinguisticNorms,
    RegardScore,
    SocialGroupSubstitution,
    StereotypeRuleHitRate,
    StereotypicalAssociations,
    ToxicityFraction,
    ToxicityProbability,
)
from bias_scope.metadata import (
    MetricInfo,
    fidelity_counts,
    list_metrics,
    normalized_deviation,
)
from bias_scope.probability_based import (
    AUL,
    AULA,
    CAT,
    CBS,
    ICAT,
    LMB,
    LPBS,
    BertPLLScorer,
    CrowSPairs,
    DisCoMetric,
    PairwiseLikelihoodPreference,
    TokenPredictionScorer,
    TopKFillDivergence,
)
from bias_scope.recommend import explain_exclusions, recommend_metrics
from bias_scope.report import (
    Report,
    compare,
    correlate,
    save_json,
    to_html,
    to_markdown,
)
from bias_scope.result import BiasResult, from_dict, make_protocol
from bias_scope.stats import bootstrap_ci, hedges_olkin_ci, permutation_p, wald_ci
from bias_scope.suite import BiasSuite
from bias_scope.utils import cosine_similarity, protocol_hash, seed_everything, to_numpy

_attach_metric_info()

_PROMPT_EXPORTS = {
    "AnalogicalReasoningBias",
    "CounterfactualAnalogyDiagnostic",
    "WinoBias",
    "ImplicitAssociationTest",
    "LLMDecisionBias",
    "DiscrimEval",
    "DecodingTrustStereotype",
    "DecodingTrustFairness",
    "PoliticalEvenHandedness",
    "FirstPersonFairness",
    "IdentitySwapConsistency",
    "BBQMetric",
    "BOLD",
    "FirstPersonFairness",
    "IdentitySwapConsistency",
    "CounterfactualFairness",
    "OccupationPronounSkew",
    "OccupationPronounSkew",
    "DemographicRepresentationBias",
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
}


_RENAMED_GENERATED = {"ScoreParity": "MeanScoreGap"}


def __getattr__(name: str):
    if name in _RENAMED_GENERATED:
        from bias_scope.generated_text_based import __getattr__ as _gt

        value = _gt(name)
        globals()[name] = value
        return value
    if name not in _PROMPT_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    value = getattr(import_module("bias_scope.prompts_based"), name)
    # The prompt family is imported lazily, so its MetricInfo is attached on
    # first access rather than at package import.
    info = METRIC_INFO.get(name)
    if info is not None and isinstance(value, type):
        from bias_scope.metadata import register

        value.info = info
        register(name, info)
    globals()[name] = value
    return value


__all__ = [
    # Framework layer (PLAN.md 5.1, 5.3)
    "MetricInfo",
    "normalized_deviation",
    "list_metrics",
    "fidelity_counts",
    "METRIC_INFO",
    "BiasResult",
    "make_protocol",
    "from_dict",
    "bootstrap_ci",
    "wald_ci",
    "hedges_olkin_ci",
    "permutation_p",
    "seed_everything",
    "protocol_hash",
    # Framework layer (PLAN.md 5.4)
    "load_model",
    "HuggingFaceBackend",
    "LiteLLMBackend",
    "StubBackend",
    "cached_generate",
    "BiasSuite",
    "recommend_metrics",
    "explain_exclusions",
    "Report",
    "save_json",
    "to_markdown",
    "to_html",
    "compare",
    "correlate",
    "WEAT",
    "SEAT",
    "CEAT",
    "SentenceBiasScore",
    "embed",
    "CrowSPairs",
    "CAT",
    "AUL",
    "LPBS",
    "PairwiseLikelihoodPreference",
    "CBS",
    "DisCoMetric",
    "TopKFillDivergence",
    "ICAT",
    "AULA",
    "LMB",
    "BertPLLScorer",
    "TokenPredictionScorer",
    "ToxicityFraction",
    "ToxicityProbability",
    "RegardScore",
    "MeanScoreGap",
    "ScoreParity",
    "SocialGroupSubstitution",
    "CoOccurrenceBiasScore",
    "CounterfactualSentimentBias",
    "DemographicRepresentation",
    "StereotypeRuleHitRate",
    "StereotypicalAssociations",
    "MarkedPersons",
    "EMT",
    "FGB",
    "GenderPolarity",
    "HONEST",
    "PGB",
    "PerspectiveAPIClient",
    "PsycholinguisticNorms",
    "AnalogicalReasoningBias",
    "CounterfactualAnalogyDiagnostic",
    "BBQMetric",
    "BOLD",
    "FirstPersonFairness",
    "IdentitySwapConsistency",
    "CounterfactualFairness",
    "OccupationPronounSkew",
    "OccupationPronounSkew",
    "DemographicRepresentationBias",
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
    "to_numpy",
    "cosine_similarity",
]
