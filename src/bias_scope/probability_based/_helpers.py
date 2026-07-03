"""
Private helper functions for probability-based metrics.

These are internal implementation details and should NOT be imported by users.
The underscore prefix in the filename signals this is a private module.
"""

from typing import List, Set, Tuple

import numpy as np


def _categorize_tokens(
    sentence1: List[str], sentence2: List[str]
) -> Tuple[Set[int], Set[int]]:
    """
    Categorize tokens as modified or unmodified (PRIVATE).

    Used by CrowS-Pairs to identify which tokens differ between
    stereotype and anti-stereotype sentences.

    Parameters
    ----------
    sentence1 : List[str]
        First tokenized sentence (stereotype)
    sentence2 : List[str]
        Second tokenized sentence (anti-stereotype)

    Returns
    -------
    Tuple[Set[int], Set[int]]
        (modified_indices, unmodified_indices)
        modified_indices: positions where tokens differ
        unmodified_indices: positions where tokens are identical

    Examples
    --------
    >>> s1 = ["Women", "are", "bad", "at", "math"]
    >>> s2 = ["Men", "are", "bad", "at", "math"]
    >>> modified, unmodified = _categorize_tokens(s1, s2)
    >>> modified  # {0}
    >>> unmodified  # {1, 2, 3, 4}
    """
    modified = set()
    unmodified = set()

    for i, (token1, token2) in enumerate(zip(sentence1, sentence2)):
        if token1 != token2:
            modified.add(i)
        else:
            unmodified.add(i)

    return modified, unmodified


def _compute_log_probability_sum(log_probs: List[float]) -> float:
    """
    Compute sum of log probabilities (PRIVATE).

    Helper for pseudo-log-likelihood computation.
    Handles edge cases like empty lists.

    Parameters
    ----------
    log_probs : List[float]
        List of log probabilities

    Returns
    -------
    float
        Sum of log probabilities

    Raises
    ------
    ValueError
        If list is empty or contains invalid values
    """
    if len(log_probs) == 0:
        raise ValueError("Cannot sum empty list of log probabilities")

    # Check for invalid values
    log_probs_array = np.array(log_probs)
    if np.isnan(log_probs_array).any():
        raise ValueError("Log probabilities contain NaN")
    if np.isinf(log_probs_array).any():
        raise ValueError("Log probabilities contain Inf")

    return float(np.sum(log_probs_array))


def _score_wordpiece_pair_crows(scorer, s_more: str, s_less: str) -> Tuple[float, float]:
    """CrowS-Pairs WordPiece-mode scoring: PLL over UNMODIFIED subwords only.

    Used by ``CrowSPairs(mode='wordpiece')``. Returns (pll_more, pll_less).
    """
    ids_a = scorer.encode(s_more)
    ids_b = scorer.encode(s_less)
    pos_a, pos_b = scorer.align_unmodified(ids_a, ids_b)
    pll_a = scorer.pll_over_positions(ids_a, pos_a)
    pll_b = scorer.pll_over_positions(ids_b, pos_b)
    return pll_a, pll_b


def _score_wordpiece_pair_aul(scorer, s_more: str, s_less: str) -> Tuple[float, float]:
    """AUL WordPiece-mode scoring per Kaneko & Bollegala 2022 (unmasked)."""
    aul_a, _ = scorer.aul_aula(scorer.encode(s_more))
    aul_b, _ = scorer.aul_aula(scorer.encode(s_less))
    return aul_a, aul_b


def _score_wordpiece_pair_aula(scorer, s_more: str, s_less: str) -> Tuple[float, float]:
    """AULA WordPiece-mode scoring per Kaneko & Bollegala 2022."""
    _, aula_a = scorer.aul_aula(scorer.encode(s_more))
    _, aula_b = scorer.aul_aula(scorer.encode(s_less))
    return aula_a, aula_b


def _normalize_probabilities(probs: np.ndarray) -> np.ndarray:
    """
    Normalize probabilities to sum to 1 (PRIVATE).

    Used when model outputs need to be normalized.

    Parameters
    ----------
    probs : np.ndarray
        Unnormalized probabilities

    Returns
    -------
    np.ndarray
        Normalized probabilities summing to 1

    Raises
    ------
    ValueError
        If all probabilities are zero
    """
    total = np.sum(probs)

    if total < 1e-10:
        raise ValueError("Cannot normalize probabilities - all values near zero")

    return probs / total
