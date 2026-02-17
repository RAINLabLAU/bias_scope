"""
bias-scope: Comprehensive bias detection for language models.

Public API for bias detection metrics organized by category:
- Embeddings: WEAT, SEAT, CEAT, SentenceBiasScore
- Probability: CrowSPairs, CAT, AUL, ICAT, AULA, LMB
- Generated Text: SocialGroupSubstitution, CoOccurrenceBiasScore, 
                  DemographicRepresentation, StereotypicalAssociations, MarkedPersons
- Utilities: to_numpy, cosine_similarity
"""

__version__ = "0.1.0"

# Public API: Import metric classes
# Embedding and probability metrics require torch - make them optional
try:
    from bias_scope.embeddings import CEAT, SEAT, WEAT, SentenceBiasScore
    from bias_scope.probability_based import AUL, AULA, CAT, CrowSPairs, ICAT, LMB
    _TORCH_AVAILABLE = True
except ImportError as e:
    # Torch not available - embedding and probability metrics won't work
    _TORCH_AVAILABLE = False
    CEAT = SEAT = WEAT = SentenceBiasScore = None
    AUL = AULA = CAT = CrowSPairs = ICAT = LMB = None

# Import generated text metrics (no torch required)
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
    "ICAT",
    "AULA",
    "LMB",
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
