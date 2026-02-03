"""Word Embedding Association Test (WEAT)."""

from typing import Tuple
import numpy as np
import torch

from bias_scope.base import EmbeddingMetric
from bias_scope.utils import to_numpy
from bias_scope.embeddings._helpers import (
    _compute_similarity_measure,
    _validate_embedding_dimensions,
    _validate_tuple_length
)


class WEAT(EmbeddingMetric):
    """
    Word Embedding Association Test.
    
    Measures bias by computing the effect size of differential associations
    between target concepts and attribute concepts in word embeddings.
    
    The test quantifies how much more strongly one target group associates
    with one attribute group compared to another, using cosine similarity
    and an effect size calculation.
    
    Reference
    ---------
    Caliskan, A., Bryson, J. J., & Narayanan, A. (2017). Semantics derived
    automatically from language corpora contain human-like biases. Science,
    356(6334), 183-186.
    
    Examples
    --------
    >>> from bias_scope.embeddings import WEAT
    >>> import numpy as np
    >>> 
    >>> # Test gender-career bias with Word2Vec embeddings
    >>> weat = WEAT()
    >>> 
    >>> # Load or create embeddings (300-dimensional)
    >>> male_names = np.random.randn(10, 300)
    >>> female_names = np.random.randn(10, 300)
    >>> career_words = np.random.randn(8, 300)
    >>> family_words = np.random.randn(8, 300)
    >>> 
    >>> # Compute bias
    >>> score = weat.compute(
    ...     (male_names, female_names),
    ...     (career_words, family_words)
    ... )
    >>> print(f"Gender-career bias effect size: {score:.3f}")
    >>> 
    >>> # Access metadata
    >>> print(weat.name)        # "WEAT"
    >>> print(weat.category)    # "embedding"
    >>> print(weat.complexity)  # "medium"
    """
    
    @property
    def name(self) -> str:
        """Return metric name."""
        return "WEAT"
    
    @property
    def reference(self) -> str:
        """Return paper citation."""
        return (
            "Caliskan, A., Bryson, J. J., & Narayanan, A. (2017). "
            "Semantics derived automatically from language corpora contain "
            "human-like biases. Science, 356(6334), 183-186."
        )
    
    @property
    def complexity(self) -> str:
        """Return complexity rating."""
        return "medium"
    
    def compute(
        self,
        target_embeddings: Tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor],
        attribute_embeddings: Tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor],
    ) -> float:
        """
        Compute WEAT effect size.
        
        Parameters
        ----------
        target_embeddings : Tuple of 2 arrays
            (target_group1, target_group2) word embeddings.
            Each array shape: (n_words, embedding_dim)
            Example: (male_names, female_names)
        attribute_embeddings : Tuple of 2 arrays
            (attribute_group1, attribute_group2) word embeddings.
            Each array shape: (n_words, embedding_dim)
            Example: (career_words, family_words)
        
        Returns
        -------
        float
            WEAT effect size. Positive values indicate target_group1 associates
            more strongly with attribute_group1. Negative values indicate
            target_group1 associates more strongly with attribute_group2.
            Larger absolute values indicate stronger bias.
        
        Raises
        ------
        ValueError
            If inputs are invalid:
            - Tuples don't have exactly 2 elements
            - Any array is empty
            - Arrays have mismatched dimensions
            - Arrays contain NaN or Inf values
            - Standard deviation is zero (all targets have identical associations)
            - Insufficient data (< 2 total target embeddings)
        
        Notes
        -----
        Effect size formula:
            d = (mean(s(X, A, B)) - mean(s(Y, A, B))) / std(s(X ∪ Y, A, B))
        
        Where:
            s(w, A, B) = mean(cos(w, a) for a in A) - mean(cos(w, b) for b in B)
            X, Y = target groups
            A, B = attribute groups
            cos = cosine similarity
        
        Examples
        --------
        >>> import numpy as np
        >>> weat = WEAT()
        >>> 
        >>> # Simple 2D example
        >>> targets1 = np.array([[1.0, 0.0], [0.9, 0.1]])
        >>> targets2 = np.array([[0.0, 1.0], [0.1, 0.9]])
        >>> attrs1 = np.array([[1.0, 0.0], [0.95, 0.05]])
        >>> attrs2 = np.array([[0.0, 1.0], [0.05, 0.95]])
        >>> 
        >>> score = weat.compute((targets1, targets2), (attrs1, attrs2))
        >>> print(f"Effect size: {score:.3f}")  # Positive score expected
        """
        # Validate tuple structure
        _validate_tuple_length(target_embeddings, "target_embeddings")
        _validate_tuple_length(attribute_embeddings, "attribute_embeddings")
        
        # Unpack and convert to numpy
        target1, target2 = target_embeddings
        attr1, attr2 = attribute_embeddings
        
        target1 = to_numpy(target1)
        target2 = to_numpy(target2)
        attr1 = to_numpy(attr1)
        attr2 = to_numpy(attr2)
        
        # Validate embeddings using inherited method
        self._validate_embeddings(target1, "target_embeddings[0]")
        self._validate_embeddings(target2, "target_embeddings[1]")
        self._validate_embeddings(attr1, "attribute_embeddings[0]")
        self._validate_embeddings(attr2, "attribute_embeddings[1]")
        
        # Validate all have same dimensions
        _validate_embedding_dimensions([target1, target2, attr1, attr2])
        
        # Compute similarity scores (using private helper)
        cos_target1 = [
            _compute_similarity_measure(w, attr1, attr2)
            for w in target1
        ]
        cos_target2 = [
            _compute_similarity_measure(w, attr1, attr2)
            for w in target2
        ]
        
        # Union for standard deviation
        union_targets = np.concatenate([target1, target2])
        cos_union = [
            _compute_similarity_measure(w, attr1, attr2)
            for w in union_targets
        ]
        
        # Compute and return effect size (private method)
        return self._compute_effect_size(cos_target1, cos_target2, cos_union)
    
    def _compute_effect_size(
        self,
        scores1: list,
        scores2: list,
        scores_union: list
    ) -> float:
        """
        Compute WEAT effect size (PRIVATE method).
        
        Calculates: (mean1 - mean2) / std(union)
        
        Parameters
        ----------
        scores1 : list of float
            Similarity scores for target group 1
        scores2 : list of float
            Similarity scores for target group 2
        scores_union : list of float
            Combined similarity scores for standard deviation calculation
        
        Returns
        -------
        float
            Effect size
            
        Raises
        ------
        ValueError
            If standard deviation is zero or insufficient data
        """
        # Check sufficient data
        if len(scores_union) < 2:
            raise ValueError(
                f"Need at least 2 total embeddings to compute effect size. "
                f"Got {len(scores_union)}."
            )
        
        # Compute standard deviation
        std_union = np.std(scores_union, ddof=1)
        
        # Check for zero std
        if std_union < 1e-10:
            raise ValueError(
                "Standard deviation of association scores is zero or near-zero. "
                "This occurs when all target embeddings have identical associations "
                "with the attribute embeddings. Cannot compute effect size."
            )
        
        # Return effect size
        return float((np.mean(scores1) - np.mean(scores2)) / std_union)
