"""Word embedding bias metrics."""

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
    
    Measures bias by computing effect size of differential associations
    between target concepts and attribute concepts.
    
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
    >>> # Gender-career bias
    >>> weat = WEAT()
    >>> male_names = np.random.randn(10, 300)
    >>> female_names = np.random.randn(10, 300)
    >>> career_words = np.random.randn(8, 300)
    >>> family_words = np.random.randn(8, 300)
    >>> 
    >>> score = weat.compute(
    ...     (male_names, female_names),
    ...     (career_words, family_words)
    ... )
    >>> print(f"Effect size: {score:.3f}")
    """
    
    @property
    def name(self) -> str:
        return "WEAT"
    
    @property
    def reference(self) -> str:
        return (
            "Caliskan, A., Bryson, J. J., & Narayanan, A. (2017). "
            "Semantics derived automatically from language corpora contain "
            "human-like biases. Science, 356(6334), 183-186."
        )
    
    @property
    def complexity(self) -> str:
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
            (target_group1, target_group2) embeddings
            Each shape: (n_words, embedding_dim)
        attribute_embeddings : Tuple of 2 arrays
            (attribute_group1, attribute_group2) embeddings
            Each shape: (n_words, embedding_dim)
        
        Returns
        -------
        float
            WEAT effect size. Positive indicates target1 associates more 
            strongly with attribute1.
        
        Raises
        ------
        ValueError
            If inputs are invalid (wrong length, empty, mismatched dimensions,
            NaN/Inf values)
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
        
        Formula: (mean1 - mean2) / std(union)
        
        Parameters
        ----------
        scores1, scores2 : list
            Similarity scores for each target group
        scores_union : list
            Combined similarity scores for std calculation
        
        Returns
        -------
        float
            Effect size
            
        Raises
        ------
        ValueError
            If std is zero or insufficient data
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


# Convenience function for backward compatibility
def weat(
    target_embeddings: Tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor],
    attribute_embeddings: Tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor],
) -> float:
    """
    Convenience function for WEAT (backward compatibility).
    
    Equivalent to:
        weat_instance = WEAT()
        weat_instance.compute(target_embeddings, attribute_embeddings)
    
    Parameters
    ----------
    target_embeddings : Tuple of 2 arrays
        Target concept embeddings
    attribute_embeddings : Tuple of 2 arrays
        Attribute concept embeddings
    
    Returns
    -------
    float
        WEAT effect size
    """
    return WEAT().compute(target_embeddings, attribute_embeddings)
