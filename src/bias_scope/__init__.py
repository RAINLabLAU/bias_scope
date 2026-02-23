"""
bias-scope: Comprehensive bias detection for language models.

Public API for bias detection metrics organized by category:
- Embeddings: WEAT, SEAT, CEAT, SentenceBiasScore
- Probability: CrowSPairs, CAT, AUL, LPBS, CBS, DisCoMetric, ICAT, AULA, LMB
- Generated Text: ToxicityFraction, ToxicityProbability, RegardScore, ScoreParity,
                  SocialGroupSubstitution, CoOccurrenceBiasScore,
                  DemographicRepresentation, StereotypicalAssociations, MarkedPersons
- Utilities: to_numpy, cosine_similarity
"""

__version__ = "0.1.0"

# Public API: Import metric classes
# Embedding and probability metrics require torch - make them optional
try:
    from bias_scope.embeddings import CEAT, SEAT, WEAT, SentenceBiasScore
    from bias_scope.probability_based import AUL, AULA, CAT, CBS, CrowSPairs, DisCoMetric, ICAT, LMB, LPBS
    _TORCH_AVAILABLE = True
except ImportError as e:
    # Torch not available - embedding and probability metrics won't work
    _TORCH_AVAILABLE = False
    CEAT = SEAT = WEAT = SentenceBiasScore = None
    AUL = AULA = CAT = CBS = CrowSPairs = DisCoMetric = ICAT = LMB = LPBS = None

# Import generated text metrics (all in generated_text_based)
from bias_scope.generated_text_based import (
    ToxicityFraction,
    ToxicityProbability,
    RegardScore,
    ScoreParity,
    SocialGroupSubstitution,
    CoOccurrenceBiasScore,
    DemographicRepresentation,
    StereotypicalAssociations,
    MarkedPersons,
    GenderPolarity,
    HONEST,
    PsycholinguisticNorms,
)

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
    "DemographicRepresentation",
    "StereotypicalAssociations",
    "MarkedPersons",
    "GenderPolarity",
    "HONEST",
    "PsycholinguisticNorms",
    # Utilities
    "to_numpy",
    "cosine_similarity",
]
