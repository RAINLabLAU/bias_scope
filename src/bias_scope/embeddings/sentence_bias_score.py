"""Sentence-level bias score with word importance weighting."""

from typing import Optional, Tuple

import numpy as np
import torch

from bias_scope.base import EmbeddingMetric
from bias_scope.utils import cosine_similarity, to_numpy


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
    >>> female_bias, male_bias = sbs.evaluate(
    ...     words, gender_dir, importance, mask
    ... )
    >>> print(f"Female bias: {female_bias:.4f}")
    >>> print(f"Male bias: {male_bias:.4f}")
    """

    @property
    def name(self) -> str:
        """Return metric name."""
        return "Sentence Bias Score"

    def evaluate(
        self,
        word_embeddings: np.ndarray | torch.Tensor,
        gender_direction: np.ndarray | torch.Tensor,
        word_importance: np.ndarray | torch.Tensor,
        gender_words_mask: Optional[np.ndarray | torch.Tensor] = None,
    ) -> Tuple[float, float]:
        """
        Evaluate gender bias score for a sentence.

        Measures stereotypical gender associations by computing weighted
        cosine similarities between word embeddings and a gender direction,
        excluding explicitly gendered words.

        Args:
            word_embeddings (np.ndarray | torch.Tensor): word embedding vectors
            gender_direction (np.ndarray | torch.Tensor): gender direction vector
            word_importance (np.ndarray | torch.Tensor): word importance weights
            gender_words_mask (np.ndarray | torch.Tensor, optional): gendered words mask

        Returns:
            Tuple[float, float]: female and male bias scores

        Raises:
            ValueError: If inputs are invalid
            TypeError: If mask is not boolean

        Notes:
            **Input Structure:**
            - word_embeddings: Shape (num_words, embedding_dim)
            - gender_direction: Shape (embedding_dim,)
              - Convention: positive = feminine, negative = masculine
              - Derived from PCA of gendered word pairs
            - word_importance: Shape (num_words,)
              - Typically from sentence encoder's max-pooling layer
              - Higher values = word contributes more to meaning
            - gender_words_mask: Shape (num_words,), boolean
              - True = exclude word (e.g., "she", "he", "mother")
              - False = include in bias calculation

            **Returns:**
            - female_bias: Sum of positive weighted similarities (≥ 0)
            - male_bias: Sum of negative weighted similarities (≤ 0)
            - Larger absolute values indicate stronger bias

            **Formula:**
                female_bias = Σ (cos(wᵢ, g) × αᵢ) for cos > 0, wᵢ not gendered
                male_bias = Σ (cos(wᵢ, g) × αᵢ) for cos < 0, wᵢ not gendered

            Where:
                - wᵢ = word embedding for word i
                - g = normalized gender direction
                - αᵢ = importance weight for word i
                - cos = cosine similarity

            **Gender Direction Derivation:**
            1. Take gendered word pairs: (she, he), (woman, man), etc.
            2. Compute difference vectors: she - he, woman - man, ...
            3. Apply PCA to find principal component
            4. This component is the gender direction

        Examples:
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
            >>> female, male = sbs.evaluate(words, gender_dir, importance)
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
                word_biases, gender_words_mask, len(word_embeddings)
            )

        # Weight by importance and separate into female/male components
        return self._compute_bias_scores(word_biases, word_importance)

    def _validate_gender_direction(
        self, gender_direction: np.ndarray, expected_dim: int
    ) -> None:
        """
        Validate gender direction vector (PRIVATE).

        Args:
            gender_direction (np.ndarray): Gender direction to validate
            expected_dim (int): Expected dimension (should match embedding dimension)

        Raises:
            ValueError: If validation fails
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

    def _validate_importance(self, importance: np.ndarray, expected_len: int) -> None:
        """
        Validate importance weights (PRIVATE).

        Args:
            importance (np.ndarray): Importance weights to validate
            expected_len (int): Expected length (should match number of words)

        Raises:
            ValueError: If validation fails
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

    def _normalize_gender_direction(self, gender_direction: np.ndarray) -> np.ndarray:
        """
        Normalize gender direction to unit vector (PRIVATE).

        Args:
            gender_direction (np.ndarray): Gender direction vector

        Returns:
            np.ndarray: Normalized unit vector

        Raises:
            ValueError: If vector has zero magnitude
        """
        norm = np.linalg.norm(gender_direction)

        if norm < 1e-10:
            raise ValueError(
                "Gender direction vector has zero or near-zero magnitude. "
                "Cannot normalize. Please provide a non-zero gender direction vector."
            )

        return gender_direction / norm

    def _compute_word_biases(
        self, embeddings: np.ndarray, gender_direction: np.ndarray
    ) -> np.ndarray:
        """
        Compute cosine similarity for each word (PRIVATE).

        Args:
            embeddings (np.ndarray): Word embeddings
            gender_direction (np.ndarray): Normalized gender direction

        Returns:
            np.ndarray: Cosine similarities (bias scores per word)
        """
        return np.array(
            [cosine_similarity(word_emb, gender_direction) for word_emb in embeddings]
        )

    def _apply_mask(
        self,
        word_biases: np.ndarray,
        mask: np.ndarray | torch.Tensor,
        expected_len: int,
    ) -> np.ndarray:
        """
        Apply gender words mask to exclude gendered words (PRIVATE).

        Args:
            word_biases (np.ndarray): Computed word biases
            mask (np.ndarray | torch.Tensor): Boolean mask (True = exclude)
            expected_len (int): Expected length for validation

        Returns:
            np.ndarray: Masked word biases (gendered words set to 0)

        Raises:
            ValueError: If mask length doesn't match
            TypeError: If mask is not boolean
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
        self, word_biases: np.ndarray, importance: np.ndarray
    ) -> Tuple[float, float]:
        """
        Compute final bias scores by weighting and separating (PRIVATE).

        Args:
            word_biases (np.ndarray): Cosine similarities for each word
            importance (np.ndarray): Importance weights for each word

        Returns:
            Tuple[float, float]: (female_bias, male_bias)
        """
        # Weight by importance
        weighted_biases = word_biases * importance

        # Separate into female (positive) and male (negative) components
        female_bias = float(np.sum(weighted_biases[weighted_biases > 0]))
        male_bias = float(np.sum(weighted_biases[weighted_biases < 0]))

        return female_bias, male_bias
