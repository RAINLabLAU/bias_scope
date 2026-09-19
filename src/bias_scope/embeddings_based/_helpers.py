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


def _weat_effect_components(
    target1: np.ndarray,
    target2: np.ndarray,
    attr1: np.ndarray,
    attr2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Return WEAT target associations, pooled SD, and effect size (PRIVATE).

    This is deliberately only the shared WEAT mathematics: cosine association
    values and the ``ddof=1`` standardized mean difference.  It has no public
    result formatting or permutation-test behaviour.
    """
    scores1 = np.asarray(
        [_compute_similarity_measure(word, attr1, attr2) for word in target1],
        dtype=float,
    )
    scores2 = np.asarray(
        [_compute_similarity_measure(word, attr1, attr2) for word in target2],
        dtype=float,
    )
    pooled = np.concatenate((scores1, scores2))
    if pooled.size < 2:
        raise ValueError("Need at least 2 target embeddings to compute a WEAT effect size.")
    pooled_sd = float(np.std(pooled, ddof=1))
    if not np.isfinite(pooled_sd) or pooled_sd < 1e-10:
        raise ValueError(
            "Standard deviation of association scores is zero or near-zero; "
            "the WEAT effect size is undefined."
        )
    effect_size = float((np.mean(scores1) - np.mean(scores2)) / pooled_sd)
    return scores1, scores2, pooled_sd, effect_size


def _ceat_random_effects(effect_sizes: np.ndarray, variances: np.ndarray) -> dict:
    """Fit CEAT's DerSimonian--Laird random-effects model (PRIVATE)."""
    effect_sizes = np.asarray(effect_sizes, dtype=float)
    variances = np.asarray(variances, dtype=float)
    if effect_sizes.ndim != 1 or variances.ndim != 1 or effect_sizes.size != variances.size:
        raise ValueError("effect_sizes and variances must be one-dimensional arrays of equal length.")
    if effect_sizes.size == 0 or not np.isfinite(effect_sizes).all():
        raise ValueError("effect_sizes must be a non-empty finite array.")
    if not np.isfinite(variances).all() or np.any(variances <= 0):
        raise ValueError("CEAT sample variances must be finite and positive.")

    fixed_weights = 1.0 / variances
    fixed_weight_sum = float(np.sum(fixed_weights))
    fixed_mean = float(np.sum(fixed_weights * effect_sizes) / fixed_weight_sum)
    q_stat = float(np.sum(fixed_weights * (effect_sizes - fixed_mean) ** 2))
    c_term = float(fixed_weight_sum - np.sum(fixed_weights**2) / fixed_weight_sum)
    if effect_sizes.size == 1:
        tau_squared = 0.0
    elif not np.isfinite(c_term) or c_term <= 0:
        raise ValueError("CEAT random-effects denominator is non-positive.")
    else:
        tau_squared = max(0.0, (q_stat - (effect_sizes.size - 1)) / c_term)

    random_weights = 1.0 / (variances + tau_squared)
    weight_sum = float(np.sum(random_weights))
    if not np.isfinite(weight_sum) or weight_sum <= 0:
        raise ValueError("CEAT random-effects weights are invalid.")
    effect_size = float(np.sum(random_weights * effect_sizes) / weight_sum)
    standard_error = float(np.sqrt(1.0 / weight_sum))
    return {
        "effect_size": effect_size,
        "standard_error": standard_error,
        "between_context_variance": float(tau_squared),
        "random_effect_weights": random_weights,
        "fixed_effect_mean": fixed_mean,
        "fixed_effect_weights": fixed_weights,
        "Q": q_stat,
        "c": c_term,
    }
