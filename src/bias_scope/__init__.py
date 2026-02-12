"""
bias-scope: Comprehensive bias detection for language models.

Public API for bias detection metrics organized by category:
- Embeddings: WEAT, SEAT, CEAT, SentenceBiasScore (requires torch)
- Probability: CrowSPairs, CAT, AUL (requires torch)
- Generated Text: SocialGroupSubstitution, CoOccurrenceBiasScore, 
                  DemographicRepresentation, StereotypicalAssociations, MarkedPersons
- Utilities: to_numpy, cosine_similarity
"""

__version__ = "0.1.0"

# Public API: Import metric classes
# Embedding and probability metrics require torch - make them optional
try:
    from bias_scope.embeddings import WEAT, SEAT, CEAT, SentenceBiasScore
    from bias_scope.probability_based import CrowSPairs, CAT, AUL
    _torch_available = True
except ImportError:
    _torch_available = False
    WEAT = None
    SEAT = None
    CEAT = None
    SentenceBiasScore = None
    CrowSPairs = None
    CAT = None
    AUL = None

# Import generated text metrics (no torch required)
from bias_scope.generated_text_based import (
    SocialGroupSubstitution,
    CoOccurrenceBiasScore,
    DemographicRepresentation,
    StereotypicalAssociations,
    MarkedPersons
)

# Public utilities
from bias_scope.utils import to_numpy, cosine_similarity

__all__ = [
    # Embedding metrics (classes) - require torch
    "WEAT",
    "SEAT",
    "CEAT",
    "SentenceBiasScore",
    
    # Probability metrics - require torch
    "CrowSPairs",
    "CAT",
    "AUL",
    
    # Generated text metrics - work without torch
    "SocialGroupSubstitution",
    "CoOccurrenceBiasScore",
    "DemographicRepresentation",
    "StereotypicalAssociations",
    "MarkedPersons",
    
    # Utilities
    "to_numpy",
    "cosine_similarity",
]
