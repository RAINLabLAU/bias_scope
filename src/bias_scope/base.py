"""
Abstract base classes for bias detection metrics.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Sequence

import numpy as np


class BiasMetric(ABC):
    """
    Abstract base class for all bias detection metrics.

    All bias metrics must implement the `evaluate` method and inherit
    their category from the appropriate intermediate base class.

    Examples
    --------
    >>> class MyMetric(BiasMetric):
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
            str: One of: 'embedding', 'probability', 'generated_text',
                or 'prompt_based'
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
        self, embeddings: np.ndarray, name: str
    ) -> None:
        """
        Validate embedding array structure (PRIVATE).

        Args:
            embeddings (np.ndarray): Embedding array to validate.
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
        self, texts: List[List[str]], name: str = "texts"
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

    def _validate_completions(
        self, completions: List[List[str]], name: str = "completions"
    ) -> None:
        """
        Validate nested completion lists (PRIVATE).

        Args:
            completions (List[List[str]]): Generated completions grouped by prompt.
            name (str): Argument name used in error messages.

        Raises:
            TypeError: If completions is not a nested list of strings.
            ValueError: If completions is empty or contains empty groups/strings.
        """
        self._validate_generated_texts(completions, name)
        expected_length = len(completions[0])

        for i, group in enumerate(completions):
            if len(group) != expected_length:
                raise ValueError(
                    f"{name} must have the same number of completions per group. "
                    f"Expected {expected_length}, got {len(group)} at index {i}."
                )
            for j, completion in enumerate(group):
                if not isinstance(completion, str):
                    raise TypeError(
                        f"{name}[{i}][{j}] must be a string, got {type(completion).__name__}"
                    )
                if completion == "":
                    raise ValueError(f"{name}[{i}][{j}] cannot be empty")

    def _validate_threshold(self, threshold: float, name: str = "threshold") -> float:
        """
        Validate a threshold in [0, 1] (PRIVATE).

        Args:
            threshold (float): Threshold value to validate.
            name (str): Argument name used in error messages.

        Returns:
            float: Validated threshold as float.

        Raises:
            TypeError: If threshold is not numeric.
            ValueError: If threshold is outside [0, 1].
        """
        if not isinstance(threshold, (int, float, np.floating)):
            raise TypeError(f"{name} must be numeric, got {type(threshold).__name__}")

        threshold_value = float(threshold)
        if not 0.0 <= threshold_value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]. Got {threshold}")

        return threshold_value

    def _validate_classifier_scores(
        self, scores: Sequence[float], name: str = "scores"
    ) -> None:
        """
        Validate classifier scores in [0, 1] (PRIVATE).

        Args:
            scores (Sequence[float]): Scores to validate.
            name (str): Argument name used in error messages.

        Raises:
            ValueError: If scores is empty or contains invalid values.
        """
        if len(scores) == 0:
            raise ValueError(f"{name} cannot be empty")

        for i, score in enumerate(scores):
            if not isinstance(score, (int, float, np.floating)):
                raise ValueError(
                    f"{name}[{i}] must be numeric, got {type(score).__name__}"
                )
            value = float(score)
            if np.isnan(value) or np.isinf(value):
                raise ValueError(f"{name}[{i}] must be finite. Got {score}")
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name}[{i}] must be in [0, 1]. Got {score}")

    def _validate_paired_completions(
        self,
        group_a_completions: List[List[str]],
        group_b_completions: List[List[str]],
    ) -> None:
        """
        Validate paired completion matrices are shape-aligned (PRIVATE).

        Args:
            group_a_completions (List[List[str]]): First completion matrix.
            group_b_completions (List[List[str]]): Second completion matrix.

        Raises:
            ValueError: If the two matrices are not shape-aligned.
        """
        if len(group_a_completions) != len(group_b_completions):
            raise ValueError(
                "group_a_completions and group_b_completions must have the same number of templates. "
                f"Got {len(group_a_completions)} and {len(group_b_completions)}."
            )
        for i, (a_template, b_template) in enumerate(
            zip(group_a_completions, group_b_completions)
        ):
            if len(a_template) != len(b_template):
                raise ValueError(
                    "group_a_completions and group_b_completions must be shape-aligned. "
                    f"Template index {i} has K={len(a_template)} vs K={len(b_template)}."
                )

    def _validate_and_cast_scores(
        self,
        completions: List[List[str]],
        scores: List[List[float]] | None = None,
        name: str = "toxicity_scores",
        sentiment_scores: List[List[float]] | None = None,
    ) -> np.ndarray:
        """
        Validate nested score matrices against completion shape (PRIVATE).

        Supports both generic paired scores in [-1, 1] and toxicity scores
        in [0, 1] while preserving existing metric call signatures.

        Args:
            completions (List[List[str]]): Completion matrix to align against.
            scores (List[List[float]] | None): Generic score matrix.
            name (str): Score matrix name for error messages.
            sentiment_scores (List[List[float]] | None): Sentiment score matrix.

        Returns:
            np.ndarray: Validated score matrix as float array.

        Raises:
            ValueError: If score inputs are missing, misaligned, or invalid.
        """
        if scores is not None and sentiment_scores is not None:
            raise ValueError(
                "Provide either scores or sentiment_scores, not both."
            )

        score_matrix = sentiment_scores if sentiment_scores is not None else scores
        if score_matrix is None:
            raise ValueError("scores cannot be None")

        if sentiment_scores is not None:
            min_value = -1.0
            max_value = 1.0
            item_prefix = name
        elif name == "toxicity_scores":
            min_value = 0.0
            max_value = 1.0
            item_prefix = "toxicity score"
        else:
            min_value = -1.0
            max_value = 1.0
            item_prefix = name

        if len(score_matrix) != len(completions):
            raise ValueError(
                f"{name} must have the same number of templates as its completions. "
                f"Got {len(score_matrix)} and {len(completions)}."
            )

        rows: List[List[float]] = []
        for i, (template_completions, template_scores) in enumerate(
            zip(completions, score_matrix)
        ):
            if len(template_scores) != len(template_completions):
                raise ValueError(
                    f"{name} must match completions shape. "
                    f"Template index {i} has K={len(template_completions)} completions "
                    f"but {len(template_scores)} "
                    f"{'toxicity scores' if name == 'toxicity_scores' else 'scores'}."
                )

            casted_template: List[float] = []
            for j, score in enumerate(template_scores):
                if not isinstance(score, (int, float, np.floating)):
                    raise ValueError(
                        f"{item_prefix} at [{i}][{j}] must be numeric, got {type(score)}"
                    )
                value = float(score)
                if np.isnan(value) or np.isinf(value):
                    raise ValueError(f"{item_prefix} at [{i}][{j}] is invalid: {value}")
                if value < min_value or value > max_value:
                    raise ValueError(
                        f"{item_prefix} at [{i}][{j}] must be in "
                        f"[{min_value:g}, {max_value:g}], got {value}"
                    )
                casted_template.append(value)
            rows.append(casted_template)

        return np.array(rows, dtype=float)


class PromptBasedMetric(BiasMetric):

    @property
    def category(self) -> str:
        return "prompt_based"

    def _validate_positive_int(self, value: int, name: str) -> int:
        """
        Validate a positive integer (PRIVATE).

        Args:
            value (int): Value to validate.
            name (str): Argument name used in error messages.

        Returns:
            int: Validated integer.

        Raises:
            ValueError: If value is not a positive integer.
        """
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer. Got {value}")
        return value
