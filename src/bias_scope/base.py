"""Abstract base classes for bias detection metrics."""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, List, Sequence
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
    Base class for generated text-based bias metrics.
    
    Provides common validation methods for text sequences, callables,
    and numeric values. All generated text metrics should inherit from this class.
    """
    
    @property
    def category(self) -> str:
        """Category is automatically set to 'generated_text'."""
        return 'generated_text'
    
    def _validate_texts(
        self,
        texts: any,
        name: str
    ) -> List[str]:
        """
        Validate texts is a sequence of non-empty strings (PRIVATE helper).
        
        Args:
            texts: Input to validate (should be sequence of strings)
            name (str): Name for error messages
            
        Returns:
            List[str]: Validated list of strings
            
        Raises:
            TypeError: If input is not a sequence or contains non-strings
            ValueError: If sequence is empty or contains empty strings
        """
        # Check if it's a string (common mistake - passing single string instead of list)
        if isinstance(texts, str):
            raise TypeError(
                f"{name} must be a sequence of strings, not a single string. "
                f"Did you mean to pass [{name}] instead?"
            )
        
        # Try to convert to list
        try:
            texts_list = list(texts)
        except TypeError:
            raise TypeError(f"{name} must be a sequence (list, tuple, etc.)")
        
        # Check not empty
        if len(texts_list) == 0:
            raise ValueError(f"{name} cannot be empty")
        
        # Check all elements are strings and non-empty
        for i, text in enumerate(texts_list):
            if not isinstance(text, str):
                raise TypeError(
                    f"{name}[{i}] must be a string, got {type(text).__name__}"
                )
            if len(text.strip()) == 0:
                raise ValueError(f"{name}[{i}] cannot be empty or whitespace-only")
        
        return texts_list
    
    def _validate_callable(
        self,
        fn: any,
        name: str
    ) -> None:
        """
        Validate that input is callable (PRIVATE helper).
        
        Args:
            fn: Input to validate
            name (str): Name for error messages
            
        Raises:
            TypeError: If input is not callable
        """
        if not callable(fn):
            raise TypeError(
                f"{name} must be callable, got {type(fn).__name__}"
            )
    
    def _validate_finite_float(
        self,
        x: any,
        name: str
    ) -> float:
        """
        Validate and convert to finite float (PRIVATE helper).
        
        Args:
            x: Input to validate
            name (str): Name for error messages
            
        Returns:
            float: Validated float value
            
        Raises:
            ValueError: If value is NaN or Inf
            TypeError: If value cannot be converted to float
        """
        try:
            value = float(x)
        except (TypeError, ValueError) as e:
            raise TypeError(
                f"{name} must be convertible to float, got {type(x).__name__}"
            ) from e
        
        if np.isnan(value):
            raise ValueError(f"{name} is NaN")
        
        if np.isinf(value):
            raise ValueError(f"{name} is Inf")
        
        return value
