"""Sentence-level bias score with word importance weighting."""

from typing import Tuple, Optional
import numpy as np
import torch

from bias_scope.base import EmbeddingMetric
from bias_scope.utils import to_numpy, cosine_similarity


class SentenceBiasScore(EmbeddingMetric):
    """
    Sentence-level bias score using word embeddings and importance weights.
    
    Computes weighted gender bias by measuring cosine similarity between
    word embeddings and a gender direction vector, weighted by each word's
    semantic importance to the sentence.
    
    Unlike WEAT/SEAT which compare groups, this metric scores individual
    sentences by aggregating word-level biases weighted by importance.
    
    Reference
    ---------
    Dolci, M., Azzalini, D., & Tanelli, M. (2023). Sentence-level bias
    detection in transformer models.
    
    Examples
    --------
    >>> from bias_scope.embeddings import SentenceBiasScore
    >>> import numpy as np
    >>> 
    >>> sbs = SentenceBiasScore()
    >>> 
    >>> # 5 words, 300-dimensional embeddings
    >>> words = np.random.randn(5, 300)
    >>> 
    >>> # Gender direction from PCA of gendered word pairs
    >>> gender_dir = np.random.randn(300)
    >>> 
    >>> # Importance from sentence encoder's max-pooling
    >>> importance = np.array([0.15, 0.25, 0.20, 0.30, 0.10])
    >>> 
    >>> # Mark first word as gendered (e.g., "she")
    >>> mask = np.array([True, False, False, False, False])
    >>> 
    >>> female_bias, male_bias = sbs.compute(
    ...     words, gender_dir, importance, mask
    ... )
    >>> print(f"Female bias: {female_bias:.4f}")
    >>> print(f"Male bias: {male_bias:.4f}")
    """
    
    @property
    def name(self) -> str:
        """Return metric name."""
        return "Sentence Bias Score"
    
    @property
    def reference(self) -> str:
        """Return paper citation."""
        return (
            "Dolci, M., Azzalini, D., & Tanelli, M. (2023). "
            "Sentence-level bias detection in transformer models."
        )
    
    @property
    def complexity(self) -> str:
        """Return complexity rating."""
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
        
        Measures stereotypical gender associations by computing weighted
        cosine similarities between word embeddings and a gender direction,
        excluding explicitly gendered words.
        
        Parameters
        ----------
        word_embeddings : array-like
            Word embedding vectors for each word in sentence.
            Shape: (num_words, embedding_dim)
        gender_direction : array-like
            Gender direction vector from PCA of gendered word pairs.
            Shape: (embedding_dim,)
            Convention: positive values = feminine, negative = masculine
        word_importance : array-like
            Semantic importance weight for each word.
            Shape: (num_words,)
            Typically derived from sentence encoder's max-pooling layer.
            Higher values = word contributes more to sentence meaning.
        gender_words_mask : array-like, optional
            Boolean mask indicating which words are explicitly gendered.
            Shape: (num_words,)
            True = exclude this word (e.g., "she", "he", "mother", "father")
            False = include this word in bias calculation
        
        Returns
        -------
        Tuple[float, float]
            (female_bias, male_bias) where:
            - female_bias: Sum of positive weighted similarities (≥ 0)
            - male_bias: Sum of negative weighted similarities (≤ 0)
            
            Larger absolute values indicate stronger bias.
        
        Raises
        ------
        ValueError
            If inputs are invalid:
            - Dimension mismatch between embeddings and gender direction
            - Length mismatch between embeddings and importance/mask
            - Negative importance values
            - Zero-magnitude gender direction
            - Arrays contain NaN or Inf values
        TypeError
            If mask is not boolean type
        
        Notes
        -----
        Formula:
            female_bias = Σ (cos(wᵢ, g) × αᵢ) for cos > 0, wᵢ not gendered
            male_bias = Σ (cos(wᵢ, g) × αᵢ) for cos < 0, wᵢ not gendered
        
        Where:
            - wᵢ = word embedding for word i
            - g = normalized gender direction
            - αᵢ = importance weight for word i
            - cos = cosine similarity
        
        Gender direction derivation:
            1. Take gendered word pairs: (she, he), (woman, man), etc.
            2. Compute difference vectors: she - he, woman - man, ...
            3. Apply PCA to find principal component
            4. This component is the gender direction
        
        Examples
        --------
        >>> import numpy as np
        >>> from bias_scope.embeddings import SentenceBiasScore
        >>> 
        >>> sbs = SentenceBiasScore()
        >>> 
        >>> # Example: "She likes beautiful dresses"
        >>> # 4 words (excluding "she")
        >>> words = np.random.randn(4, 300)  # "likes", "beautiful", "dresses"
        >>> gender_dir = np.array([1.0] + [0.0]*299)  # Simplified
        >>> importance = np.array([0.2, 0.3, 0.3, 0.2])
        >>> 
        >>> # No mask (or mask out "she" separately)
        >>> female, male = sbs.compute(words, gender_dir, importance)
        >>> 
        >>> if female > abs(male):
        ...     print("Sentence has feminine associations")
        ... else:
        ...     print("Sentence has masculine associations")
        """
        # Convert to numpy
        word_embeddings = to_numpy(word_embeddings)
        gender_direction = to_numpy(gender_direction)
        word_importance = to_numpy(word_importance)
        
        # Validate using inherited and private methods
        self._validate_embeddings(word_embeddings, "word_embeddings")
        self._validate_gender_direction(gender_direction, word_embeddings.shape[1])
        self._validate_importance(word_importance, len(word_embeddings))
        
        # Normalize gender direction to unit vector
        gender_direction = self._normalize_gender_direction(gender_direction)
        
        # Compute cosine similarity for each word
        word_biases = self._compute_word_biases(word_embeddings, gender_direction)
        
        # Apply mask if provided
        if gender_words_mask is not None:
            word_biases = self._apply_mask(
                word_biases, 
                gender_words_mask, 
                len(word_embeddings)
            )
        
        # Weight by importance and separate into female/male components
        return self._compute_bias_scores(word_biases, word_importance)
    
    def _validate_gender_direction(
        self, 
        gender_direction: np.ndarray, 
        expected_dim: int
    ) -> None:
        """
        Validate gender direction vector (PRIVATE).
        
        Parameters
        ----------
        gender_direction : np.ndarray
            Gender direction to validate
        expected_dim : int
            Expected dimension (should match embedding dimension)
            
        Raises
        ------
        ValueError
            If validation fails
        """
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
        """
        Validate importance weights (PRIVATE).
        
        Parameters
        ----------
        importance : np.ndarray
            Importance weights to validate
        expected_len : int
            Expected length (should match number of words)
            
        Raises
        ------
        ValueError
            If validation fails
        """
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
                f"Found {np.sum(importance < 0)} negative values. "
                f"Minimum value: {np.min(importance)}"
            )
    
    def _normalize_gender_direction(
        self, 
        gender_direction: np.ndarray
    ) -> np.ndarray:
        """
        Normalize gender direction to unit vector (PRIVATE).
        
        Parameters
        ----------
        gender_direction : np.ndarray
            Gender direction vector
        
        Returns
        -------
        np.ndarray
            Normalized unit vector
            
        Raises
        ------
        ValueError
            If vector has zero magnitude
        """
        norm = np.linalg.norm(gender_direction)
        
        if norm < 1e-10:
            raise ValueError(
                "Gender direction vector has zero or near-zero magnitude. "
                "Cannot normalize. Please provide a non-zero gender direction vector."
            )
        
        return gender_direction / norm
    
    def _compute_word_biases(
        self, 
        embeddings: np.ndarray, 
        gender_direction: np.ndarray
    ) -> np.ndarray:
        """
        Compute cosine similarity for each word (PRIVATE).
        
        Parameters
        ----------
        embeddings : np.ndarray
            Word embeddings
        gender_direction : np.ndarray
            Normalized gender direction
        
        Returns
        -------
        np.ndarray
            Cosine similarities (bias scores per word)
        """
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
        """
        Apply gender words mask to exclude gendered words (PRIVATE).
        
        Parameters
        ----------
        word_biases : np.ndarray
            Computed word biases
        mask : array-like
            Boolean mask (True = exclude)
        expected_len : int
            Expected length for validation
        
        Returns
        -------
        np.ndarray
            Masked word biases (gendered words set to 0)
            
        Raises
        ------
        ValueError
            If mask length doesn't match
        TypeError
            If mask is not boolean
        """
        mask = to_numpy(mask)
        
        # Validate mask length
        if mask.shape[0] != expected_len:
            raise ValueError(
                f"Mask length {mask.shape[0]} "
                f"does not match number of words {expected_len}"
            )
        
        # Validate mask is boolean
        if mask.dtype != bool:
            raise TypeError(
                f"gender_words_mask must be boolean array. "
                f"Got dtype: {mask.dtype}. "
                f"Convert using .astype(bool) if needed."
            )
        
        # Return zero if all masked (no non-gendered words)
        if mask.all():
            return np.zeros_like(word_biases)
        
        # Apply mask: True = exclude (multiply by False = 0)
        return word_biases * (~mask)
    
    def _compute_bias_scores(
        self, 
        word_biases: np.ndarray, 
        importance: np.ndarray
    ) -> Tuple[float, float]:
        """
        Compute final bias scores by weighting and separating (PRIVATE).
        
        Parameters
        ----------
        word_biases : np.ndarray
            Cosine similarities for each word
        importance : np.ndarray
            Importance weights for each word
        
        Returns
        -------
        Tuple[float, float]
            (female_bias, male_bias)
        """
        # Weight by importance
        weighted_biases = word_biases * importance
        
        # Separate into female (positive) and male (negative) components
        female_bias = float(np.sum(weighted_biases[weighted_biases > 0]))
        male_bias = float(np.sum(weighted_biases[weighted_biases < 0]))
        
        return female_bias, male_bias
