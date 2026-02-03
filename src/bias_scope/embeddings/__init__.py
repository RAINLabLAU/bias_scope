"""Embedding-based bias metrics."""

from bias_scope.embeddings.word_embeddings import WEAT, weat
from bias_scope.embeddings.sentence_embeddings import (
    SEAT, 
    SentenceBiasScore,
    seat,
    sentence_bias
)

# Public API (classes)
__all__ = ["WEAT", "SEAT", "SentenceBiasScore"]

# Note: Functional API (weat, seat, sentence_bias) is available
# but not exported in __all__ - users can still import them if needed
