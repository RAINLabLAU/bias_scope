"""
Abstract base classes for bias detection metrics.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import numpy as np


class BiasMetric(ABC):
    """
    Abstract base class for all bias detection metrics.

    All bias metrics must implement the `evaluate` method and provide
    metadata about the metric through properties.

    Examples
    --------
    >>> class MyMetric(BiasMetric):
    ...     @property
    ...     def name(self):
    ...         return "My Custom Metric"
    ...
    ...     @property
    ...     def category(self):
    ...         return "embedding"
    ...
    ...     def evaluate(self, inputs):
    ...         return 0.5
    """

    @abstractmethod
    def evaluate(self, *args, **kwargs) -> float | Dict[str, float]:
        """
        Evaluate the bias metric.

        Args:
            *args: metric-specific input data
            **kwargs: additional metric parameters

        Returns:
            float | Dict[str, float]: bias score(s)

        Raises:
            ValueError: If inputs are invalid

        Notes:
            - Simple metrics return a single float score
            - Complex metrics return a dictionary with multiple scores
            - Subclasses must implement with their specific signature and validation
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def category(self) -> str:
        """
        Metric category.

        Returns:
            str: One of: 'embedding', 'probability', 'generated_text'
        """
        raise NotImplementedError


class EmbeddingMetric(BiasMetric):
    """
    Base class for embedding-based bias metrics.

    Provides common validation methods for embeddings.
    """

    @property
    def category(self) -> str:
        """Category is automatically set to 'embedding'."""
        return "embedding"

    def _validate_embeddings(self, embeddings: Tuple[np.ndarray, np.ndarray], name: str) -> None:
        """
        Validate embedding tuple structure (PRIVATE).

        Args:
            embeddings (Tuple[np.ndarray, np.ndarray]): embedding array tuple
            name (str): name for error messages

        Raises:
            ValueError: If validation fails
        """
        if len(embeddings) == 0:
            raise ValueError(f"{name} cannot be empty")

        if np.isnan(embeddings).any():
            raise ValueError(f"{name} contains NaN values")

        if np.isinf(embeddings).any():
            raise ValueError(f"{name} contains Inf values")


class ProbabilityMetric(BiasMetric):
    """
    Base class for probability-based bias metrics.

    Provides common validation methods for probabilities and sentence pairs.
    All probability-based metrics (CrowS-Pairs, CAT, AUL, iCAT, AULA, LMB)
    should inherit from this class.
    """

    @property
    def category(self) -> str:
        """Category is automatically set to 'probability'."""
        return "probability"

    def _validate_probabilities(self, probabilities: np.ndarray, name: str = "probabilities") -> None:
        """
        Validate probability array (PRIVATE helper).

        Checks that probabilities are valid: in [0,1], no NaN/Inf.

        Args:
            probabilities (np.ndarray): Probability array to validate
            name (str): Name for error messages (default: "probabilities")

        Raises:
            ValueError: If probabilities are invalid
        """
        if len(probabilities) == 0:
            raise ValueError(f"{name} cannot be empty")

        if np.isnan(probabilities).any():
            raise ValueError(f"{name} contains NaN values")

        if np.isinf(probabilities).any():
            raise ValueError(f"{name} contains Inf values")

        if (probabilities < 0).any() or (probabilities > 1).any():
            raise ValueError(
                f"{name} must be in range [0, 1]. "
                f"Got min={np.min(probabilities)}, max={np.max(probabilities)}"
            )

    def _validate_sentence_pair(self, sentence1: List[str], sentence2: List[str]) -> None:
        """
        Validate sentence pair has same length (PRIVATE).

        Args:
            sentence1 (List[str]): first tokenized sentence
            sentence2 (List[str]): second tokenized sentence

        Raises:
            ValueError: If validation fails
        """
        if len(sentence1) == 0 or len(sentence2) == 0:
            raise ValueError("Sentences cannot be empty")

        if len(sentence1) != len(sentence2):
            raise ValueError(
                "Sentence pairs must have same length. "
                f"Got {len(sentence1)} and {len(sentence2)} tokens."
            )
