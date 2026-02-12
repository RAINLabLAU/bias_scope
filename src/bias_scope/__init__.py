"""
bias-scope: Comprehensive bias detection for language models.

Public API for bias detection metrics organized by category:
- Embeddings: WEAT, SEAT, CEAT, SentenceBiasScore
- Probability: CrowSPairs, CAT, AUL
- Generated Text: SocialGroupSubstitution, CoOccurrenceBiasScore, 
                  DemographicRepresentation, StereotypicalAssociations, MarkedPersons
- Utilities: to_numpy, cosine_similarity
"""

__version__ = "0.1.0"

# Public API: Import metric classes
from bias_scope.embeddings import CEAT, SEAT, WEAT, SentenceBiasScore

# Import probability metrics
from bias_scope.probability_based import AUL, CAT, CrowSPairs

# Import generated text metrics
from bias_scope.generated_text_based import (
    SocialGroupSubstitution,
    CoOccurrenceBiasScore,
    DemographicRepresentation,
    StereotypicalAssociations,
    MarkedPersons
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
    # Generated text metrics
    "SocialGroupSubstitution",
    "CoOccurrenceBiasScore",
    "DemographicRepresentation",
    "StereotypicalAssociations",
    "MarkedPersons",
    # Utilities
    "to_numpy",
    "cosine_similarity",
]
