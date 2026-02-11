"""Abstract base classes for bias detection metrics."""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, List
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
    ...         # Implementation
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
        pass
    
    @abstractmethod
    def name(self) -> str:
        """
        Metric name.
        
        Returns:
            str: Name of the metric (e.g., 'WEAT', 'SEAT', 'CEAT')
        """
        pass
    
    @abstractmethod
    def category(self) -> str:
        """
        Metric category.
        
        Returns:
            str: One of: 'embedding', 'probability', 'generated_text'
        """
        pass
    



class EmbeddingMetric(BiasMetric):
    """
    Base class for embedding-based bias metrics.
    
    Provides common validation methods for embeddings.
    """
    
    @property
    def category(self) -> str:
        return 'embedding'
    
    def _validate_embeddings(
        self, 
        embeddings: Tuple[np.ndarray, np.ndarray],
        name: str
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
    
    def category(self) -> str:
        """Category is automatically set to 'probability'."""
        return 'probability'
    
    def _validate_probabilities(
        self,
        probabilities: np.ndarray,
        name: str = "probabilities"
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
        self,
        sentence1: List[str],
        sentence2: List[str]
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
                f"Sentence pairs must have same length. "
                f"Got {len(sentence1)} and {len(sentence2)} tokens."
            )


class GeneratedTextMetric(BiasMetric):
    """
    Base class for generated text bias metrics.
    
    Provides common validation methods for generated text and classifier scores.
    All generated text metrics (TF, TP, EMT, RegardScore, ScoreParity, etc.)
    should inherit from this class.
    """
    
    @property
    def category(self) -> str:
        """Category is automatically set to 'generated_text'."""
        return 'generated_text'
    
    def _validate_generated_texts(
        self,
        generated_texts: List[List[str]],
        name: str = "generated_texts"
    ) -> None:
        """
        Validate generated texts structure (PRIVATE).
        
        Parameters
        ----------
        generated_texts : List[List[str]]
            List of text lists (one per prompt)
        name : str, default="generated_texts"
            Name for error messages
            
        Raises
        ------
        ValueError
            If structure is invalid
        """
        if len(generated_texts) == 0:
            raise ValueError(f"{name} cannot be empty")
        
        for i, texts in enumerate(generated_texts):
            if not isinstance(texts, list):
                raise ValueError(
                    f"{name}[{i}] must be a list of strings. "
                    f"Got {type(texts).__name__}"
                )
            
            if len(texts) == 0:
                raise ValueError(
                    f"{name}[{i}] cannot be empty. "
                    f"Each prompt must have at least one generated text."
                )
            
            for j, text in enumerate(texts):
                if not isinstance(text, str):
                    raise ValueError(
                        f"{name}[{i}][{j}] must be a string. "
                        f"Got {type(text).__name__}"
                    )
    
    def _validate_threshold(
        self,
        threshold: float,
        name: str = "threshold"
    ) -> None:
        """
        Validate threshold value (PRIVATE).
        
        Parameters
        ----------
        threshold : float
            Threshold value to validate
        name : str, default="threshold"
            Name for error messages
            
        Raises
        ------
        ValueError
            If threshold not in [0, 1]
        """
        if not 0 <= threshold <= 1:
            raise ValueError(
                f"{name} must be in [0, 1]. Got {threshold}"
            )
    
    def _validate_classifier_scores(
        self,
        scores: List[float],
        name: str = "scores"
    ) -> None:
        """
        Validate classifier scores (PRIVATE).
        
        Parameters
        ----------
        scores : List[float]
            Scores to validate
        name : str, default="scores"
            Name for error messages
            
        Raises
        ------
        ValueError
            If scores are invalid (NaN, Inf, out of range)
        """
        scores_array = np.array(scores)
        
        if np.isnan(scores_array).any():
            raise ValueError(f"{name} contains NaN values")
        
        if np.isinf(scores_array).any():
            raise ValueError(f"{name} contains Inf values")
        
        if (scores_array < 0).any() or (scores_array > 1).any():
            raise ValueError(
                f"{name} must be in [0, 1]. "
                f"Got min={np.min(scores_array):.3f}, max={np.max(scores_array):.3f}"
            )
