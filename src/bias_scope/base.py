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
    
    @property
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
    
    @property
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
    
    @property
    def reference(self) -> Optional[str]:
        """
        Citation for the original paper.
        
        Returns
        -------
        str or None
            Full citation with authors, year, and title
        """
        return None
    
    @property
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
