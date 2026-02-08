"""Abstract base classes for bias detection metrics."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import numpy as np


class BiasMetric(ABC):
    """
    Abstract base class for all bias detection metrics.
    
    All bias metrics must implement the `compute` method and provide
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
    ...     def compute(self, inputs):
    ...         # Implementation
    ...         return 0.5
    """
    
    @abstractmethod
    def compute(self, *args, **kwargs) -> float | Dict[str, float]:
        """
        Compute the bias metric.
        
        Returns
        -------
        float or Dict[str, float]
            Bias score(s). Simple metrics return float, complex metrics
            return dictionary with multiple scores.
        
        Raises
        ------
        ValueError
            If inputs are invalid
        """
        pass
    
    @abstractmethod
    def name(self) -> str:
        """
        Metric name.
        
        Returns
        -------
        str
            Name of the metric (e.g., 'WEAT', 'SEAT', 'CEAT')
        """
        pass
    
    @abstractmethod
    def category(self) -> str:
        """
        Metric category.
        
        Returns
        -------
        str
            One of: 'embedding', 'probability', 'generated_text'
        """
        pass
    
    def reference(self) -> Optional[str]:
        """
        Citation for the original paper.
        
        Returns
        -------
        str or None
            Full citation with authors, year, and title
        """
        return None
    
    def complexity(self) -> str:
        """
        Implementation complexity rating.
        
        Returns
        -------
        str
            One of: 'easy', 'medium', 'hard'
        """
        return 'medium'


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
        embeddings: np.ndarray,
        name: str = "embeddings"
    ) -> None:
        """
        Validate embedding array (PRIVATE helper).
        
        Parameters
        ----------
        embeddings : np.ndarray
            Embedding array to validate
        name : str
            Name for error messages
            
        Raises
        ------
        ValueError
            If embeddings are invalid (empty, NaN, Inf)
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
        
        Parameters
        ----------
        probabilities : np.ndarray
            Probability array to validate
        name : str, default="probabilities"
            Name for error messages
            
        Raises
        ------
        ValueError
            If probabilities are invalid
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
        sentence1: list,
        sentence2: list
    ) -> None:
        """
        Validate sentence pair for pseudo-log-likelihood metrics (PRIVATE).
        
        Parameters
        ----------
        sentence1, sentence2 : list
            Tokenized sentences to validate
            
        Raises
        ------
        ValueError
            If sentences are invalid (empty, different lengths)
        """
        if len(sentence1) == 0 or len(sentence2) == 0:
            raise ValueError("Sentences cannot be empty")
        
        if len(sentence1) != len(sentence2):
            raise ValueError(
                f"Sentence pairs must have same length. "
                f"Got {len(sentence1)} and {len(sentence2)} tokens."
            )
