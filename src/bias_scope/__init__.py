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

__version__ = "0.1.0"

# Public API: Import metric classes
# Embedding and probability metrics require torch - make them optional
try:
    from bias_scope.embeddings_based import CEAT, SEAT, WEAT, SentenceBiasScore
    from bias_scope.probability_based import (
        AUL,
        AULA,
        CAT,
        CBS,
        ICAT,
        LMB,
        LPBS,
        CrowSPairs,
        DisCoMetric,
    )

    _TORCH_AVAILABLE = True
except ImportError as e:
    # Torch not available - embedding and probability metrics won't work
    _TORCH_AVAILABLE = False
    CEAT = SEAT = WEAT = SentenceBiasScore = None
    AUL = AULA = CAT = CBS = CrowSPairs = DisCoMetric = ICAT = LMB = LPBS = None

# Import generated text metrics (all in generated_text_based)
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

try:
    from bias_scope.prompts_based.analogical_reasoning_bias import (
        AnalogicalReasoningBias,
    )
except ImportError:
    AnalogicalReasoningBias = None

try:
    from bias_scope.prompts_based.bbq import BBQMetric
except ImportError:
    BBQMetric = None

try:
    from bias_scope.prompts_based.bold import BOLD
except ImportError:
    BOLD = None

try:
    from bias_scope.prompts_based.counterfactual_fairness import (
        CounterfactualFairness,
    )
except ImportError:
    CounterfactualFairness = None

try:
    from bias_scope.prompts_based.demographic_representation_bias import (
        DemographicRepresentationBias,
    )
except ImportError:
    DemographicRepresentationBias = None

try:
    from bias_scope.prompts_based.opinion_consistency_across_personas import (
        OpinionConsistencyAcrossPersonas,
    )
except ImportError:
    OpinionConsistencyAcrossPersonas = None

try:
    from bias_scope.prompts_based.realtoxicityprompts import RealToxicityPrompts
except ImportError:
    RealToxicityPrompts = None

try:
    from bias_scope.prompts_based.stereoset import StereoSetMetric
except ImportError:
    StereoSetMetric = None

try:
    from bias_scope.prompts_based.tof_nof import TofNof
except ImportError:
    TofNof = None

try:
    from bias_scope.prompts_based.truthfulqa import TruthfulQA
except ImportError:
    TruthfulQA = None

try:
    from bias_scope.prompts_based.unqover import UnQoverMetric
except ImportError:
    UnQoverMetric = None

# Public utilities
from bias_scope.utils import cosine_similarity, to_numpy

__all__ = [
    # Embedding metrics (classes)
    "WEAT",
    "SEAT",
    "CEAT",
    "SentenceBiasScore",
    # Probability metrics
    "CrowSPairs",
    "CAT",
    "AUL",
    "LPBS",
    "CBS",
    "DisCoMetric",
    "ICAT",
    "AULA",
    "LMB",
    # Generated text metrics
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
    # Prompt-based metrics
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
    # Utilities
    "to_numpy",
    "cosine_similarity",
]
