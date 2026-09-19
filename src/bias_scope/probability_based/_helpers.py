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
    pos_a, pos_b = _filter_special_aligned_positions(
        scorer, ids_a, ids_b, pos_a, pos_b
    )
    if not pos_a:
        raise ValueError(
            "No shared non-special WordPiece tokens found after alignment. "
            "CrowS-Pairs cannot score a pair whose only shared tokens are special tokens."
        )
    pll_a = scorer.pll_over_positions(ids_a, pos_a)
    pll_b = scorer.pll_over_positions(ids_b, pos_b)
    return pll_a, pll_b


def _special_positions(scorer, input_ids: List[int]) -> Set[int]:
    """Return positions occupied by tokenizer special tokens."""
    tokenizer = getattr(scorer, "tokenizer", None)
    get_mask = getattr(tokenizer, "get_special_tokens_mask", None)
    if callable(get_mask):
        mask = get_mask(input_ids, already_has_special_tokens=True)
        return {i for i, is_special in enumerate(mask) if is_special}

    special_ids = getattr(tokenizer, "all_special_ids", None)
    if special_ids is None:
        special_ids = getattr(scorer, "all_special_ids", None)
    if special_ids is None:
        return set()

    special_id_set = set(special_ids)
    return {i for i, token_id in enumerate(input_ids) if token_id in special_id_set}


def _filter_special_aligned_positions(
    scorer,
    ids_a: List[int],
    ids_b: List[int],
    pos_a: List[int],
    pos_b: List[int],
) -> Tuple[List[int], List[int]]:
    """Drop aligned positions where either side is a tokenizer special token."""
    specials_a = _special_positions(scorer, ids_a)
    specials_b = _special_positions(scorer, ids_b)
    filtered_a: List[int] = []
    filtered_b: List[int] = []
    for a, b in zip(pos_a, pos_b):
        if a in specials_a or b in specials_b:
            continue
        filtered_a.append(a)
        filtered_b.append(b)
    return filtered_a, filtered_b


def _reject_masked_scorer(scorer, metric_name: str) -> None:
    """Refuse a masking-based scorer in AUL/AULA's whitespace mode.

    ``BertPLLScorer`` and ``WordPieceBertScorer`` implement the
    ``TokenPredictionScorer`` protocol (``token_probability`` /
    ``token_probability_with_attention``) by masking the scored position
    before predicting it -- correct for CrowS-Pairs' pseudo-log-likelihood,
    but the opposite of AUL/AULA (Kaneko & Bollegala 2022 eq. 4-5), which
    scores every token from the complete UNMASKED sentence. Silently
    accepting either scorer here would compute PLL, not AUL/AULA.
    """
    if scorer is None:
        return
    from bias_scope.probability_based.scorers import BertPLLScorer, WordPieceBertScorer

    if isinstance(scorer, (BertPLLScorer, WordPieceBertScorer)):
        raise ValueError(
            f"{metric_name} mode='whitespace' cannot use a "
            f"{type(scorer).__name__} instance: its token_probability / "
            "token_probability_with_attention methods mask the scored token "
            "and compute PLL, not AUL/AULA. Use mode='wordpiece' (passing "
            "model_name= or a WordPieceBertScorer) or a custom callback that "
            "scores from the complete unmasked sentence."
        )


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
