"""
Abstract base classes for bias detection metrics.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Sequence, Tuple

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

    def _validate_embeddings(
        self, embeddings: Tuple[np.ndarray, np.ndarray], name: str
    ) -> None:
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

    def _validate_probabilities(
        self, probabilities: np.ndarray, name: str = "probabilities"
    ) -> None:
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

    def _validate_sentence_pair(
        self, sentence1: List[str], sentence2: List[str]
    ) -> None:
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


class GeneratedTextMetric(BiasMetric):
    """
    Base class for generated text bias metrics.
    """

    @property
    def category(self) -> str:
        """Category is automatically set to 'generated_text'."""
        return "generated_text"

    def _validate_texts(self, texts: Sequence[str], name: str) -> List[str]:
        """
        Validate a sequence of text strings (PRIVATE).

        Args:
            texts (Sequence[str]): Text values to validate.
            name (str): Input name for error messages.

        Returns:
            List[str]: Validated text list.

        Raises:
            TypeError: If texts is not a sequence of strings.
            ValueError: If texts is empty or contains empty strings.
        """
        if not isinstance(texts, Sequence) or isinstance(texts, (str, bytes)):
            raise TypeError(f"{name} must be a Sequence of strings")

        texts_list = list(texts)
        if len(texts_list) == 0:
            raise ValueError(f"{name} cannot be empty")

        for i, text in enumerate(texts_list):
            if not isinstance(text, str):
                raise TypeError(
                    f"{name}[{i}] must be a string, got {type(text).__name__}"
                )
            if text == "":
                raise ValueError(f"{name}[{i}] cannot be empty")

        return texts_list

    def _validate_callable(self, fn: Callable, name: str) -> None:
        """
        Validate a callable input (PRIVATE).

        Args:
            fn (Callable): Function to validate.
            name (str): Input name for error messages.

        Raises:
            TypeError: If fn is not callable.
        """
        if not callable(fn):
            raise TypeError(f"{name} must be callable, got {type(fn).__name__}")

    def _validate_finite_float(self, value: float, name: str) -> float:
        """
        Validate a finite float value (PRIVATE).

        Args:
            value (float): Value to validate.
            name (str): Input name for error messages.

        Returns:
            float: Validated float value.

        Raises:
            TypeError: If value is not numeric.
            ValueError: If value is NaN or infinite.
        """
        if not isinstance(value, (int, float, np.floating)):
            raise TypeError(f"{name} must be a float, got {type(value).__name__}")

        value_float = float(value)
        if np.isnan(value_float) or np.isinf(value_float):
            raise ValueError(f"{name} must be finite, got {value}")
        return value_float

    def _validate_generated_texts(
        self, texts: List[List[str]], name: str
    ) -> None:
        """
        Validate a list-of-lists of generated texts (PRIVATE).

        Args:
            texts (List[List[str]]): Nested text list to validate.
                Shape: (n_prompts, n_texts_per_prompt)
            name (str): Argument name used in error messages.

        Raises:
            ValueError: If the outer list is empty.
            ValueError: If any inner list is empty.
        """
        if len(texts) == 0:
            raise ValueError(f"{name} cannot be empty")

        for inner in texts:
            if len(inner) == 0:
                raise ValueError(f"{name} cannot be empty")


class PromptBasedMetric(BiasMetric):

    @property
    def category(self) -> str:
        return "prompt_based"
