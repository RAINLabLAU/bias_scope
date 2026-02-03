"""Sentence embedding bias metrics."""

from typing import Tuple, Optional
import numpy as np
import torch

from bias_scope.base import EmbeddingMetric
from bias_scope.utils import to_numpy, cosine_similarity
from bias_scope.embeddings.word_embeddings import WEAT


class SEAT(EmbeddingMetric):
    """
    Sentence Encoder Association Test.
    
    Adapts WEAT to contextualized sentence embeddings. Uses the same
    effect size calculation as WEAT.
    
    Reference
    ---------
    May, C., Wang, A., Bordia, S., Bowman, S. R., & Rudinger, R. (2019).
    On measuring social biases in sentence encoders. NAACL-HLT 2019.
    
    Examples
    --------
    >>> from bias_scope.embeddings import SEAT
    >>> import numpy as np
    >>> 
    >>> seat = SEAT()
    >>> # Sentence embeddings from BERT (768-dim)
    >>> male_sentences = np.random.randn(10, 768)
    >>> female_sentences = np.random.randn(10, 768)
    >>> career_sentences = np.random.randn(8, 768)
    >>> family_sentences = np.random.randn(8, 768)
    >>> 
    >>> score = seat.compute(
    ...     (male_sentences, female_sentences),
    ...     (career_sentences, family_sentences)
    ... )
    """
    
    @property
    def name(self) -> str:
        return "SEAT"
    
    @property
    def reference(self) -> str:
        return (
            "May, C., Wang, A., Bordia, S., Bowman, S. R., & Rudinger, R. (2019). "
            "On measuring social biases in sentence encoders. NAACL-HLT 2019."
        )
    
    @property
    def complexity(self) -> str:
        return "easy"
    
    def compute(
        self,
        target_embeddings: Tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor],
        attribute_embeddings: Tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor],
    ) -> float:
        """
        Compute SEAT score.
        
        SEAT uses the same calculation as WEAT, just with sentence embeddings
        instead of word embeddings.
        
        Parameters
        ----------
        target_embeddings : Tuple of 2 arrays
            (target_group1, target_group2) sentence embeddings
            Each shape: (n_sentences, embedding_dim)
        attribute_embeddings : Tuple of 2 arrays
            (attribute_group1, attribute_group2) sentence embeddings
            Each shape: (n_sentences, embedding_dim)
        
        Returns
        -------
        float
            SEAT effect size (same as WEAT)
        """
        # SEAT is just WEAT applied to sentence embeddings
        weat_instance = WEAT()
        return weat_instance.compute(target_embeddings, attribute_embeddings)


class SentenceBiasScore(EmbeddingMetric):
    """
    Sentence-level bias score using word embeddings and importance weights.
    
    Computes weighted gender bias by measuring cosine similarity between
    word embeddings and a gender direction vector, weighted by semantic
    importance.
    
    Reference
    ---------
    Dolci, M., Azzalini, D., & Tanelli, M. (2023). Sentence Bias Score.
    
    Examples
    --------
    >>> from bias_scope.embeddings import SentenceBiasScore
    >>> import numpy as np
    >>> 
    >>> sbs = SentenceBiasScore()
    >>> # 5 words, 300-dimensional embeddings
    >>> words = np.random.randn(5, 300)
    >>> gender_dir = np.random.randn(300)
    >>> importance = np.array([0.15, 0.25, 0.20, 0.30, 0.10])
    >>> mask = np.array([True, False, False, False, False])  # Mask gendered words
    >>> 
    >>> female_bias, male_bias = sbs.compute(
    ...     words, gender_dir, importance, mask
    ... )
    """
    
    @property
    def name(self) -> str:
        return "Sentence Bias Score"
    
    @property
    def reference(self) -> str:
        return (
            "Dolci, M., Azzalini, D., & Tanelli, M. (2023). "
            "Sentence-level bias detection in transformer models."
        )
    
    @property
    def complexity(self) -> str:
        return "easy"
    
    def compute(
        self,
        word_embeddings: np.ndarray | torch.Tensor,
        gender_direction: np.ndarray | torch.Tensor,
        word_importance: np.ndarray | torch.Tensor,
        gender_words_mask: Optional[np.ndarray | torch.Tensor] = None,
    ) -> Tuple[float, float]:
        """
        Compute gender bias score for a sentence.
        
        Parameters
        ----------
        word_embeddings : array-like
            Word embedding vectors, shape (num_words, embedding_dim)
        gender_direction : array-like
            Gender direction vector from PCA, shape (embedding_dim,)
            Positive = feminine, negative = masculine
        word_importance : array-like
            Semantic importance weight for each word, shape (num_words,)
        gender_words_mask : array-like, optional
            Boolean mask for gendered words to exclude, shape (num_words,)
            True = exclude this word
        
        Returns
        -------
        Tuple[float, float]
            (female_bias, male_bias) where:
            - female_bias: Sum of positive weighted similarities
            - male_bias: Sum of negative weighted similarities
        
        Raises
        ------
        ValueError
            If array shapes are incompatible or values are invalid
        TypeError
            If mask is not boolean type
        """
        # Convert to numpy
        word_embeddings = to_numpy(word_embeddings)
        gender_direction = to_numpy(gender_direction)
        word_importance = to_numpy(word_importance)
        
        # Validate using inherited and private methods
        self._validate_embeddings(word_embeddings, "word_embeddings")
        self._validate_gender_direction(gender_direction, word_embeddings.shape[1])
        self._validate_importance(word_importance, len(word_embeddings))
        
        # Normalize gender direction
        gender_direction = self._normalize_gender_direction(gender_direction)
        
        # Compute word-level biases
        word_biases = self._compute_word_biases(word_embeddings, gender_direction)
        
        # Apply mask if provided
        if gender_words_mask is not None:
            word_biases = self._apply_mask(
                word_biases, 
                gender_words_mask, 
                len(word_embeddings)
            )
        
        # Weight by importance and separate
        return self._compute_bias_scores(word_biases, word_importance)
    
    def _validate_gender_direction(
        self, 
        gender_direction: np.ndarray, 
        expected_dim: int
    ) -> None:
        """Validate gender direction vector (PRIVATE)."""
        if np.isnan(gender_direction).any():
            raise ValueError("gender_direction contains NaN values")
        
        if np.isinf(gender_direction).any():
            raise ValueError("gender_direction contains Inf values")
        
        if gender_direction.shape[0] != expected_dim:
            raise ValueError(
                f"Gender direction dimension {gender_direction.shape[0]} "
                f"does not match embedding dimension {expected_dim}"
            )
    
    def _validate_importance(
        self, 
        importance: np.ndarray, 
        expected_len: int
    ) -> None:
        """Validate importance weights (PRIVATE)."""
        if np.isnan(importance).any():
            raise ValueError("word_importance contains NaN values")
        
        if np.isinf(importance).any():
            raise ValueError("word_importance contains Inf values")
        
        if importance.shape[0] != expected_len:
            raise ValueError(
                f"Importance array length {importance.shape[0]} "
                f"does not match number of words {expected_len}"
            )
        
        if (importance < 0).any():
            raise ValueError(
                "Word importance values must be non-negative. "
                f"Found {np.sum(importance < 0)} negative values."
            )
    
    def _normalize_gender_direction(
        self, 
        gender_direction: np.ndarray
    ) -> np.ndarray:
        """Normalize gender direction to unit vector (PRIVATE)."""
        norm = np.linalg.norm(gender_direction)
        
        if norm < 1e-10:
            raise ValueError(
                "Gender direction vector has zero or near-zero magnitude. "
                "Cannot normalize."
            )
        
        return gender_direction / norm
    
    def _compute_word_biases(
        self, 
        embeddings: np.ndarray, 
        gender_direction: np.ndarray
    ) -> np.ndarray:
        """Compute cosine similarity for each word (PRIVATE)."""
        return np.array([
            cosine_similarity(word_emb, gender_direction) 
            for word_emb in embeddings
        ])
    
    def _apply_mask(
        self, 
        word_biases: np.ndarray, 
        mask: np.ndarray | torch.Tensor, 
        expected_len: int
    ) -> np.ndarray:
        """Apply gender words mask (PRIVATE)."""
        mask = to_numpy(mask)
        
        # Validate mask
        if mask.shape[0] != expected_len:
            raise ValueError(
                f"Mask length {mask.shape[0]} "
                f"does not match number of words {expected_len}"
            )
        
        if mask.dtype != bool:
            raise TypeError(
                f"gender_words_mask must be boolean array. "
                f"Got dtype: {mask.dtype}. "
                f"Convert using .astype(bool) if needed."
            )
        
        # Return zero if all masked
        if mask.all():
            return np.zeros_like(word_biases)
        
        # Apply mask
        return word_biases * (~mask)
    
    def _compute_bias_scores(
        self, 
        word_biases: np.ndarray, 
        importance: np.ndarray
    ) -> Tuple[float, float]:
        """Compute final bias scores (PRIVATE)."""
        weighted_biases = word_biases * importance
        
        female_bias = float(np.sum(weighted_biases[weighted_biases > 0]))
        male_bias = float(np.sum(weighted_biases[weighted_biases < 0]))
        
        return female_bias, male_bias


# Convenience functions for backward compatibility
def seat(
    target_embeddings: Tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor],
    attribute_embeddings: Tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor],
) -> float:
    """Convenience function for SEAT."""
    return SEAT().compute(target_embeddings, attribute_embeddings)


def sentence_bias(
    word_embeddings: np.ndarray | torch.Tensor,
    gender_direction: np.ndarray | torch.Tensor,
    word_importance: np.ndarray | torch.Tensor,
    gender_words_mask: Optional[np.ndarray | torch.Tensor] = None,
) -> Tuple[float, float]:
    """Convenience function for Sentence Bias Score."""
    return SentenceBiasScore().compute(
        word_embeddings, 
        gender_direction, 
        word_importance, 
        gender_words_mask
    )
