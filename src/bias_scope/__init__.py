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


from bias_scope.probability_based import (
    AUL,
    AULA,
    BertPLLScorer,
    CAT,
    CBS,
    ICAT,
    LMB,
    LPBS,
    TokenPredictionScorer,
    CrowSPairs,
    DisCoMetric,
)

from bias_scope.generated_text_based import (
    HONEST,
    CoOccurrenceBiasScore,
    CounterfactualSentimentBias,
    DemographicRepresentation,
    EMT,
    FGB,
    GenderPolarity,
    MarkedPersons,
    PGB,
    PerspectiveAPIClient,
    PsycholinguisticNorms,
    RegardScore,
    ScoreParity,
    SocialGroupSubstitution,
    StereotypicalAssociations,
    ToxicityFraction,
    ToxicityProbability,
)

from bias_scope.utils import cosine_similarity, to_numpy


_PROMPT_EXPORTS = {
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
}


def __getattr__(name: str):
    if name not in _PROMPT_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    value = getattr(import_module("bias_scope.prompts_based"), name)
    globals()[name] = value
    return value


__all__ = [
    "WEAT",
    "SEAT",
    "CEAT",
    "SentenceBiasScore",
    "embed",
    "CrowSPairs",
    "CAT",
    "AUL",
    "LPBS",
    "CBS",
    "DisCoMetric",
    "ICAT",
    "AULA",
    "LMB",
    "BertPLLScorer",
    "TokenPredictionScorer",
    "ToxicityFraction",
    "ToxicityProbability",
    "RegardScore",
    "ScoreParity",
    "SocialGroupSubstitution",
    "CoOccurrenceBiasScore",
    "CounterfactualSentimentBias",
    "DemographicRepresentation",
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
    "to_numpy",
    "cosine_similarity",
]