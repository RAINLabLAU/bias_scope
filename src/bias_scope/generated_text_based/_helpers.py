"""
Private helper functions for generated text-based metrics.

These are internal implementation details and should NOT be imported by users.
The underscore prefix in the filename signals this is a private module.
"""

import re
from collections import Counter
from typing import Counter as CounterType
from typing import List, Sequence, Set

import numpy as np

from bias_scope.base import GeneratedTextMetric
from bias_scope.generated_text_based.perspective_api import PerspectiveAPIClient

# Small epsilon for numerical stability in log operations
EPSILON = 1e-12


def tokenize(text: str) -> List[str]:
    """
    Simple deterministic tokenizer (PRIVATE).

    Tokenizes text by:
    - Converting to lowercase
    - Splitting on non-alphanumeric boundaries
    - Preserving apostrophes within words

    Parameters
    ----------
    text : str
        Input text to tokenize

    Returns
    -------
    List[str]
        List of tokens (lowercase)

    Examples
    --------
    >>> tokenize("Hello, world!")
    ['hello', 'world']

    >>> tokenize("She's working")
    ["she's", 'working']

    >>> tokenize("The CEO's car")
    ['the', "ceo's", 'car']

    Notes
    -----
    This is a simple baseline tokenizer. For production use, consider
    using a proper NLP tokenizer (spaCy, NLTK, etc.).
    """
    # Lowercase
    text = text.lower()

    # Split on word boundaries, keeping apostrophes
    # Pattern: match sequences of letters/numbers/apostrophes
    tokens = re.findall(r"[a-z0-9']+", text)

    return tokens


def normalize_lexicon(lex: Sequence[str]) -> Set[str]:
    """
    Normalize lexicon to lowercase set (PRIVATE).

    Parameters
    ----------
    lex : Sequence[str]
        Input lexicon (list of terms)

    Returns
    -------
    Set[str]
        Normalized lexicon (lowercase, deduplicated)

    Examples
    --------
    >>> normalize_lexicon(["Man", "Woman", "MAN"])
    {'man', 'woman'}
    """
    return {term.lower() for term in lex}


def count_lexicon_mentions(tokens: List[str], lex: Set[str]) -> int:
    """
    Count mentions of lexicon terms in token list (PRIVATE).

    Parameters
    ----------
    tokens : List[str]
        List of tokens (assumed lowercase)
    lex : Set[str]
        Normalized lexicon (lowercase set)

    Returns
    -------
    int
        Count of lexicon mentions

    Examples
    --------
    >>> tokens = ['the', 'man', 'and', 'woman']
    >>> lex = {'man', 'woman'}
    >>> count_lexicon_mentions(tokens, lex)
    2
    """
    return sum(1 for token in tokens if token in lex)


def find_token_positions(tokens: List[str], lex: Set[str]) -> List[int]:
    """
    Find positions of lexicon terms in token list (PRIVATE).

    Parameters
    ----------
    tokens : List[str]
        List of tokens
    lex : Set[str]
        Normalized lexicon

    Returns
    -------
    List[int]
        List of positions (indices) where lexicon terms appear

    Examples
    --------
    >>> tokens = ['the', 'man', 'walked', 'home']
    >>> lex = {'man', 'woman'}
    >>> find_token_positions(tokens, lex)
    [1]
    """
    return [i for i, token in enumerate(tokens) if token in lex]


def count_cooccurrence_in_window(
    tokens: List[str],
    anchor_positions: List[int],
    target_lex: Set[str],
    window_size: int,
) -> Counter:
    """
    Count co-occurrence of target terms within window of anchors (PRIVATE).

    For each anchor position, looks within ±window_size tokens and counts
    occurrences of target lexicon terms.

    Behavior: If a token position is within window of multiple anchors,
    it is counted once per anchor (simpler and aligns with "opportunities" scaling).

    Parameters
    ----------
    tokens : List[str]
        List of tokens
    anchor_positions : List[int]
        Positions of anchor terms
    target_lex : Set[str]
        Target lexicon to count
    window_size : int
        Window size (±window_size around each anchor)

    Returns
    -------
    Counter
        Counter of target term occurrences

    Examples
    --------
    >>> tokens = ['the', 'man', 'is', 'smart', 'and', 'kind']
    >>> anchors = [1]  # 'man' at position 1
    >>> target = {'smart', 'kind'}
    >>> count_cooccurrence_in_window(tokens, anchors, target, window_size=3)
    Counter({'smart': 1, 'kind': 1})
    """
    counts = Counter()

    for anchor_pos in anchor_positions:
        # Define window boundaries
        start = max(0, anchor_pos - window_size)
        end = min(len(tokens), anchor_pos + window_size + 1)

        # Count target terms in window
        for i in range(start, end):
            # Skip the anchor position itself
            if i == anchor_pos:
                continue

            token = tokens[i]
            if token in target_lex:
                counts[token] += 1

    return counts


def compute_log_odds_with_prior(
    c_marked: int,
    c_unmarked: int,
    n_marked: int,
    n_unmarked: int,
    a_w: float,
    a_total: float,
) -> tuple[float, float, float]:
    """
    Compute log-odds with informative Dirichlet prior (Monroe et al., 2008) (PRIVATE).

    Computes:
    - delta: log-odds difference
    - variance: variance estimate
    - z: z-score (delta / sqrt(variance))

    Parameters
    ----------
    c_marked : int
        Count of term in marked corpus
    c_unmarked : int
        Count of term in unmarked corpus
    n_marked : int
        Total tokens in marked corpus
    n_unmarked : int
        Total tokens in unmarked corpus
    a_w : float
        Prior count for this word
    a_total : float
        Total prior mass

    Returns
    -------
    tuple[float, float, float]
        (delta, variance, z_score)

    Notes
    -----
    Formula:
        delta = log((c_m + a_w) / (n_m - c_m + a_¬w))
              - log((c_u + a_w) / (n_u - c_u + a_¬w))
        var = 1/(c_m + a_w) + 1/(c_u + a_w)
        z = delta / sqrt(var)

    Where a_¬w = a_total - a_w
    """
    a_notw = a_total - a_w

    # Log-odds for marked
    numerator_m = c_marked + a_w
    denominator_m = n_marked - c_marked + a_notw
    log_odds_m = (
        float(np.log(numerator_m / denominator_m)) if denominator_m > 0 else 0.0
    )

    # Log-odds for unmarked
    numerator_u = c_unmarked + a_w
    denominator_u = n_unmarked - c_unmarked + a_notw
    log_odds_u = (
        float(np.log(numerator_u / denominator_u)) if denominator_u > 0 else 0.0
    )

    # Delta
    delta = log_odds_m - log_odds_u

    # Variance
    variance = 1.0 / (c_marked + a_w) + 1.0 / (c_unmarked + a_w)

    # Z-score
    z_score = delta / (variance**0.5) if variance > 0 else 0.0

    return delta, variance, z_score


class ToxicityMetric(GeneratedTextMetric):
    """
    Base class for toxicity-based metrics (PRIVATE).

    Provides shared Perspective API integration for TF, TP, and EMT.
    All toxicity metrics inherit from this class to reuse the API client.
    """

    def __init__(self, api_key: str):
        """
        Initialize ToxicityMetric.

        Args:
            api_key (str): Google Cloud API key for Perspective API
        """
        self.perspective = PerspectiveAPIClient(api_key)

    def _score_texts(self, texts: List[str]) -> List[float]:
        """
        Get toxicity scores for a batch of texts (PRIVATE).

        Args:
            texts (List[str]): texts to score

        Returns:
            List[float]: toxicity scores
        """
        return self.perspective.score_batch(texts)
