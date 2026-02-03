"""
Private helper functions for embedding metrics.

These are internal implementation details and should NOT be imported by users.
Notice the underscore prefix in filename - signals this is a private module.
"""

import numpy as np
from bias_scope.utils import cosine_similarity


def _validate_tuple_length(tup: tuple, name: str) -> None:
    """
    Validate tuple has exactly 2 elements (PRIVATE).
    
    Parameters
    ----------
    tup : tuple
        Tuple to validate
    name : str
        Name for error message
        
    Raises
    ------
    ValueError
        If tuple doesn't have exactly 2 elements
    """
    if len(tup) != 2:
        raise ValueError(
            f"{name} must have exactly 2 elements. Got {len(tup)}."
        )


def _validate_embedding_dimensions(arrays: list) -> None:
    """
    Validate all arrays have same embedding dimension (PRIVATE).
    
    Parameters
    ----------
    arrays : list of np.ndarray
        Embedding arrays to check
        
    Raises
    ------
    ValueError
        If arrays have different dimensions
    """
    dims = [arr.shape[1] for arr in arrays]
    
    if len(set(dims)) != 1:
        raise ValueError(
            f"All embeddings must have same dimension. "
            f"Got dimensions: {dims}"
        )


def _compute_similarity_measure(
    target_word: np.ndarray,
    attr_embeddings1: np.ndarray,
    attr_embeddings2: np.ndarray
) -> float:
    """
    Compute similarity measure for WEAT (PRIVATE).
    
    Calculates: mean(cos(target, attr1)) - mean(cos(target, attr2))
    
    Parameters
    ----------
    target_word : np.ndarray
        Single word embedding vector
    attr_embeddings1, attr_embeddings2 : np.ndarray
        Attribute group embeddings
    
    Returns
    -------
    float
        Difference in mean cosine similarities
    """
    cos_attr1 = [cosine_similarity(target_word, a) for a in attr_embeddings1]
    cos_attr2 = [cosine_similarity(target_word, a) for a in attr_embeddings2]
    
    return np.mean(cos_attr1) - np.mean(cos_attr2)
