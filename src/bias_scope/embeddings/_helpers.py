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
        raise ValueError(f"{name} must have exactly 2 elements. Got {len(tup)}.")


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
            f"All embeddings must have same dimension. Got dimensions: {dims}"
        )


def _compute_similarity_measure(
    target_word: np.ndarray, attr_embeddings1: np.ndarray, attr_embeddings2: np.ndarray
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


def _compute_random_effects_weights(weat_scores: list) -> np.ndarray:
    """
    Compute random-effects model weights for CEAT (PRIVATE).

    Uses a simplified DerSimonian-Laird estimator to compute inverse-variance
    weights that account for between-sample heterogeneity.

    Parameters
    ----------
    weat_scores : list of float
        WEAT scores from random samples

    Returns
    -------
    np.ndarray
        Normalized weights summing to 1.0

    Notes
    -----
    Formula:
        w_i = 1 / (var + tau^2 + epsilon)

    Where:
        - var: sample variance of WEAT scores
        - tau^2: between-sample variance (max(0, var - 0))
        - epsilon: small constant for numerical stability
    """
    # Calculate variance
    variance = np.var(weat_scores, ddof=1)

    # Estimate between-sample variance (tau-squared)
    # Simplified version: assumes within-sample variance is negligible
    tau_squared = max(0, variance)

    # Calculate inverse-variance weights
    weights = np.array(
        [
            1.0 / (variance + tau_squared + 1e-10)  # Add epsilon for stability
            for _ in weat_scores
        ]
    )

    # Normalize to sum to 1
    return weights / weights.sum()
