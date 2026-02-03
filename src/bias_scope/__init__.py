"""
bias-scope: Comprehensive bias detection for language models.

Public API for bias detection metrics organized by category:
- Embeddings: WEAT, SEAT, SentenceBiasScore
- Utilities: to_numpy, cosine_similarity
"""

__version__ = "0.1.0"

# Public API: Import metric classes
from bias_scope.embeddings import WEAT, SEAT, SentenceBiasScore

# Public utilities
from bias_scope.utils import to_numpy, cosine_similarity

__all__ = [
    # Embedding metrics (classes)
    "WEAT",
    "SEAT",
    "SentenceBiasScore",
    
    # Utilities
    "to_numpy",
    "cosine_similarity",
]
