"""
bias-scope: Comprehensive bias detection for language models.

Public API for bias detection metrics organized by category:
- Embeddings: WEAT, SEAT, CEAT, SentenceBiasScore
- Probability: CrowSPairs, CAT, AUL, LPBS, CBS, DisCoMetric
- Generated Text: ToxicityFraction, ToxicityProbability, RegardScore, ScoreParity
- Utilities: to_numpy, cosine_similarity
"""

__version__ = "0.1.0"

# Public API: Import metric classes
from bias_scope.embeddings import CEAT, SEAT, WEAT, SentenceBiasScore

# Import probability metrics
from bias_scope.probability_based import AUL, CAT, CBS, CrowSPairs, DisCoMetric, LPBS

# Import generated text metrics
from bias_scope.generated_text import (
    ToxicityFraction,
    ToxicityProbability,
    RegardScore,
    ScoreParity
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
    # Generated text metrics
    "ToxicityFraction",
    "ToxicityProbability",
    "RegardScore",
    "ScoreParity",
    # Utilities
    "to_numpy",
    "cosine_similarity",
]
