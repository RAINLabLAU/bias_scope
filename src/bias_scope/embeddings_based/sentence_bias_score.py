"""Sentence-level bias score with word importance weighting."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Dict, Tuple

import numpy as np

if TYPE_CHECKING:  # torch is an optional extra; used in annotations only
    import torch

from bias_scope.base import EmbeddingMetric
from bias_scope.utils import cosine_similarity, to_numpy

#: Dolci, Azzalini & Tanelli 2023, Sec. 3.2: the ten canonical gender word
#: pairs used to derive the gender direction via PCA. ``derive_gender_direction``
#: expects embeddings supplied in this order, one array of the first word of
#: each pair ("female") and one of the second ("male").
GENDER_WORD_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("woman", "man"),
    ("girl", "boy"),
    ("she", "he"),
    ("mother", "father"),
    ("daughter", "son"),
    ("gal", "guy"),
    ("female", "male"),
    ("her", "his"),
    ("herself", "himself"),
    ("Mary", "John"),
)


def derive_gender_direction(
    female_embeddings: np.ndarray, male_embeddings: np.ndarray
) -> np.ndarray:
    """Gender direction via PCA of gender word-pair difference vectors.

    Dolci et al. 2023, Sec. 3.2: take ``n`` pairs of gender words (e.g.
    ``GENDER_WORD_PAIRS``), form each pair's difference vector (female minus
    male), and take the top principal component of that set of difference
    vectors as the gender direction.

    "Principal component" here means the top singular vector of the
    **uncentred** difference-vector matrix (no global mean subtracted first).
    Gender-pair difference vectors are expected to point in roughly the same
    direction (that shared direction *is* the gender signal), so subtracting
    their mean before decomposing would remove exactly the signal being
    sought and instead surface the residual variation between pairs. This
    mirrors the origin of the construction, Bolukbasi et al. 2016 (cited by
    Dolci et al. as the source of the gender-pair-difference idea), whose own
    per-pair-centred vectors are likewise not re-centred globally before SVD.
    The paper publishes no reference code and reports only a scree plot
    (its Fig. 2), which does not disambiguate centred from uncentred; this is
    a `decide`, not a verified fact (see docs/fidelity/sentence_bias_score.md).

    The paper observes empirically that the resulting direction gives female
    words a positive cosine and male words a negative one; this is enforced
    here by orienting the component along the mean difference vector, rather
    than leaving PCA's arbitrary sign to chance.

    Parameters
    ----------
    female_embeddings, male_embeddings : np.ndarray
        Shape ``(n_pairs, embedding_dim)``. Row ``i`` of each must be the two
        words of the same gender pair (e.g. row 0 = "woman" and "man").

    Returns
    -------
    np.ndarray
        Unit-norm gender direction, shape ``(embedding_dim,)``.

    Raises
    ------
    ValueError
        If the two inputs are not equal-shaped ``(n_pairs, dim)`` matrices,
        if fewer than 2 pairs are given, or if the resulting direction has
        zero or near-zero magnitude.
    """
    female_embeddings = to_numpy(female_embeddings)
    male_embeddings = to_numpy(male_embeddings)

    if (
        female_embeddings.ndim != 2
        or male_embeddings.ndim != 2
        or female_embeddings.shape != male_embeddings.shape
    ):
        raise ValueError(
            "female_embeddings and male_embeddings must both have shape "
            f"(n_pairs, embedding_dim). Got {female_embeddings.shape} and "
            f"{male_embeddings.shape}."
        )
    if female_embeddings.shape[0] < 2:
        raise ValueError(
            "Need at least 2 gender word pairs to run PCA. "
            f"Got {female_embeddings.shape[0]}."
        )

    differences = female_embeddings - male_embeddings
    mean_difference = differences.mean(axis=0)
    # Uncentred SVD (see the "Principal component" note above): the top
    # right-singular vector of the difference vectors themselves, not of
    # their deviation from the mean.
    _, _, vt = np.linalg.svd(differences, full_matrices=False)
    component = vt[0]

    if np.dot(component, mean_difference) < 0:
        component = -component

    norm = np.linalg.norm(component)
    if norm < 1e-10:
        raise ValueError(
            "Derived gender direction has zero or near-zero magnitude; the "
            "gender word pairs may not encode a consistent direction."
        )
    return component / norm


def derive_word_importance(hidden_states: np.ndarray) -> np.ndarray:
    """Word importance via max-pooling selection counts.

    Dolci et al. 2023, Sec. 3.4: a max-pooling sentence encoder builds its
    fixed-size sentence vector by taking, independently for each of the
    ``D`` output dimensions, the largest value across all ``T`` time steps
    (tokens). Word importance is "how many times in the max-pooling phase a
    word representation is selected", expressed as a percentage of ``D``; the
    percentages sum to 1.0 over the whole sentence (including tokens later
    excluded as gender words -- the paper does not renormalise after
    exclusion, see docs/fidelity/sentence_bias_score.md).

    Parameters
    ----------
    hidden_states : np.ndarray
        Shape ``(n_tokens, embedding_dim)``: the encoder's per-token hidden
        state *before* max-pooling, one row per token in sentence order.

    Returns
    -------
    np.ndarray
        Shape ``(n_tokens,)``, percentage importance per token, summing to
        1.0. Ties are broken by ``numpy.argmax`` (the earliest token wins),
        since the paper does not specify a tie-breaking rule.

    Raises
    ------
    ValueError
        If ``hidden_states`` is not rank-2, is empty, or contains NaN/Inf.
    """
    hidden_states = to_numpy(hidden_states)
    if hidden_states.ndim != 2:
        raise ValueError(
            "hidden_states must be a rank-2 matrix with shape (n_tokens, "
            f"embedding_dim). Got shape {hidden_states.shape}."
        )
    if hidden_states.shape[0] == 0 or hidden_states.shape[1] == 0:
        raise ValueError(
            "hidden_states must have at least one token and one dimension. "
            f"Got shape {hidden_states.shape}."
        )
    if not np.isfinite(hidden_states).all():
        raise ValueError("hidden_states contains NaN or Inf values.")

    n_tokens, dim = hidden_states.shape
    selected_token_per_dim = np.argmax(hidden_states, axis=0)
    counts = np.bincount(selected_token_per_dim, minlength=n_tokens)
    return counts.astype(float) / dim


def build_gender_words_mask(tokens: Sequence[str], gender_word_list: Sequence[str]) -> np.ndarray:
    """Build a ``gender_words_mask`` from tokens and a gender-word lexicon.

    Dolci et al. 2023, Sec. 3.3: a token is a gender word (excluded from bias
    estimation) if it appears in ``L``, matched case-insensitively -- the
    paper's own list stores lower-cased and capitalised forms explicitly, so
    this treats the two as equivalent rather than requiring both.

    **BiasScope does not ship Dolci et al.'s 6562-word lexicon** (409 + 388
    common nouns compiled starting from Bolukbasi et al. 2016 and Zhao et al.
    2018b, plus 5765 U.S. Social Security given names): the paper does not
    publish the merged list, and reconstructing it from the two source lists
    would not reproduce the authors' own curation. Callers must supply
    ``gender_word_list`` themselves. See docs/fidelity/sentence_bias_score.md
    and REVIEW_LATER.md.

    Parameters
    ----------
    tokens : Sequence[str]
        The sentence's tokens, in order.
    gender_word_list : Sequence[str]
        The gender-word lexicon ``L``, any mix of case/singular/plural forms.

    Returns
    -------
    np.ndarray
        Boolean array, shape ``(len(tokens),)``, ``True`` where the token is
        in the lexicon.
    """
    lexicon = {str(word).lower() for word in gender_word_list}
    return np.array([str(token).lower() in lexicon for token in tokens], dtype=bool)


class SentenceBiasScore(EmbeddingMetric):
    """
    Sentence-level bias score using word embeddings and importance weights.

    Computes weighted gender bias by measuring cosine similarity between
    word embeddings and a gender direction vector, weighted by each word's
    semantic importance to the sentence.

    Unlike WEAT/SEAT which compare groups, this metric scores individual
    sentences by aggregating word-level biases weighted by importance.

    Reference
    ---------
    Dolci, T., Azzalini, F., & Tanelli, M. (2023). Improving Gender-Related
    Fairness in Sentence Encoders: A Semantics-Based Approach. Data Science
    and Engineering, 8, 177-195. https://doi.org/10.1007/s41019-023-00211-0

    Examples
    --------
    >>> from bias_scope.embeddings_based import SentenceBiasScore
    >>> import numpy as np
    >>>
    >>> sbs = SentenceBiasScore()
    >>>
    >>> # 5 words, 300-dimensional embeddings
    >>> words = np.random.randn(5, 300)
    >>>
    >>> # Gender direction from PCA of gendered word pairs
    >>> gender_dir = np.random.randn(300)
    >>>
    >>> # Importance from sentence encoder's max-pooling
    >>> importance = np.array([0.15, 0.25, 0.20, 0.30, 0.10])
    >>>
    >>> # Mark first word as gendered (e.g., "she")
    >>> mask = np.array([True, False, False, False, False])
    >>>
    >>> female_bias, male_bias = sbs.evaluate(
    ...     words, gender_dir, importance, mask
    ... )
    >>> print(f"Female bias: {female_bias:.4f}")
    >>> print(f"Male bias: {male_bias:.4f}")
    """

    def __init__(self, model_name: str | None = None):
        """
        Initialize SentenceBiasScore.

        Args:
            model_name (str | None): Deprecated compatibility parameter. The
                canonical metric requires caller-provided token/word embeddings
                from the sentence; raw text is not embedded inside this class.
        """
        self.model_name = model_name

    def evaluate(
        self,
        word_embeddings: np.ndarray | torch.Tensor,
        gender_direction: np.ndarray | torch.Tensor,
        word_importance: np.ndarray | torch.Tensor,
        gender_words_mask: np.ndarray | torch.Tensor,
        return_details: bool = False,
    ) -> Tuple[float, float] | Dict[str, Any]:
        """
        Evaluate gender bias score for a sentence.

        Measures stereotypical gender associations by computing weighted
        cosine similarities between word embeddings and a gender direction,
        excluding explicitly gendered words.

        Args:
            word_embeddings (np.ndarray | torch.Tensor): rank-2 token/word
                embedding matrix from the sentence, shape ``(n_tokens, dim)``.
            gender_direction (np.ndarray | torch.Tensor): gender direction vector
                with positive=female and negative=male convention.
            word_importance (np.ndarray | torch.Tensor): rank-1 semantic
                importance vector, shape ``(n_tokens,)``.
            gender_words_mask (np.ndarray | torch.Tensor): rank-1 boolean mask,
                shape ``(n_tokens,)``. ``True`` excludes explicit gender terms.

        Returns:
            Tuple[float, float]: female and male bias scores

        Raises:
            ValueError: If inputs are invalid
            TypeError: If mask is not boolean

        Notes:
            **Input Structure:**
            - word_embeddings: Shape (num_words, embedding_dim)
              - Precomputed token/word representations from the sentence.
              - Raw strings are not accepted because independently embedding
                words does not reproduce contextual sentence token vectors.
            - gender_direction: Shape (embedding_dim,)
              - Convention: positive = feminine, negative = masculine
              - Derived from PCA of gendered word pairs
            - word_importance: Shape (num_words,)
              - Typically from sentence encoder's max-pooling layer
              - Higher values = word contributes more to meaning
            - gender_words_mask: Shape (num_words,), boolean, required
              - True = exclude word (e.g., "she", "he", "mother")
              - False = include in bias calculation

            **Returns:**
            - female_bias: Sum of positive weighted similarities (≥ 0)
            - male_bias: Sum of negative weighted similarities (≤ 0)
            - Larger absolute values indicate stronger bias

            **Formula:**
                female_bias = Σ (cos(wᵢ, g) × αᵢ) for cos > 0, wᵢ not gendered
                male_bias = Σ (cos(wᵢ, g) × αᵢ) for cos < 0, wᵢ not gendered

            Where:
                - wᵢ = word embedding for word i
                - g = normalized gender direction
                - αᵢ = importance weight for word i
                - cos = cosine similarity

            **Gender Direction Derivation:**
            1. Take gendered word pairs: (she, he), (woman, man), etc.
            2. Compute difference vectors: she - he, woman - man, ...
            3. Apply PCA to find principal component
            4. This component is the gender direction

        Examples:
            >>> import numpy as np
            >>> from bias_scope.embeddings_based import SentenceBiasScore
            >>>
            >>> sbs = SentenceBiasScore()
            >>>
            >>> # Example: "She likes beautiful dresses"
            >>> words = np.random.randn(4, 300)
            >>> gender_dir = np.array([1.0] + [0.0]*299)  # Simplified
            >>> importance = np.array([0.2, 0.3, 0.3, 0.2])
            >>> mask = np.array([True, False, False, False])
            >>>
            >>> female, male = sbs.evaluate(words, gender_dir, importance, mask)
            >>>
            >>> if female > abs(male):
            ...     print("Sentence has feminine associations")
            ... else:
            ...     print("Sentence has masculine associations")
        """
        if _is_text_sequence(word_embeddings):
            raise TypeError(
                "word_embeddings must be a rank-2 token/word embedding matrix. "
                "Raw text inputs are noncanonical for SentenceBiasScore; embed "
                "tokens in sentence context and pass the resulting vectors."
            )

        # Convert to numpy
        word_embeddings = to_numpy(word_embeddings)
        gender_direction = to_numpy(gender_direction)
        word_importance = to_numpy(word_importance)

        # Validate using inherited and private methods
        self._validate_embeddings(word_embeddings, "word_embeddings")
        self._validate_gender_direction(gender_direction, word_embeddings.shape[1])
        self._validate_importance(word_importance, len(word_embeddings))

        # Normalize gender direction to unit vector
        gender_direction = self._normalize_gender_direction(gender_direction)

        # Compute cosine similarity for each word
        word_biases = self._compute_word_biases(word_embeddings, gender_direction)

        word_biases = self._apply_mask(
            word_biases, gender_words_mask, len(word_embeddings)
        )

        # Weight by importance and separate into female/male components
        female_bias, male_bias = self._compute_bias_scores(
            word_biases, word_importance
        )
        if return_details:
            absolute_bias = abs(female_bias) + abs(male_bias)
            return {
                "female_bias": female_bias,
                "male_bias": male_bias,
                "absolute_bias": absolute_bias,
                # Dolci et al. Eq. 3 (Abs-BiasScore), the paper's own single-value
                # summary "useful ... when sorting multiple sentences" -- this is
                # what run() reports as BiasResult.score.
                "bias_score": absolute_bias,
                "breakdown": {"female_bias": female_bias, "male_bias": male_bias},
                "num_words": float(len(word_embeddings)),
                "n": int(len(word_embeddings)),
            }
        return female_bias, male_bias

    def _validate_gender_direction(
        self, gender_direction: np.ndarray, expected_dim: int
    ) -> None:
        """
        Validate gender direction vector (PRIVATE).

        Args:
            gender_direction (np.ndarray): Gender direction to validate
            expected_dim (int): Expected dimension (should match embedding dimension)

        Raises:
            ValueError: If validation fails
        """
        if gender_direction.ndim != 1:
            raise ValueError(
                "gender_direction must be a rank-1 vector with shape "
                f"(embedding_dim,). Got shape {gender_direction.shape}."
            )

        if np.isnan(gender_direction).any():
            raise ValueError("gender_direction contains NaN values")

        if np.isinf(gender_direction).any():
            raise ValueError("gender_direction contains Inf values")

        if gender_direction.shape[0] != expected_dim:
            raise ValueError(
                f"Gender direction dimension {gender_direction.shape[0]} "
                f"does not match embedding dimension {expected_dim}"
            )

    def _validate_importance(self, importance: np.ndarray, expected_len: int) -> None:
        """
        Validate importance weights (PRIVATE).

        Args:
            importance (np.ndarray): Importance weights to validate
            expected_len (int): Expected length (should match number of words)

        Raises:
            ValueError: If validation fails
        """
        if importance.ndim != 1:
            raise ValueError(
                "word_importance must be a rank-1 vector with shape "
                f"(num_words,). Got shape {importance.shape}."
            )

        if np.isnan(importance).any():
            raise ValueError("word_importance contains NaN values")

        if np.isinf(importance).any():
            raise ValueError("word_importance contains Inf values")

        if importance.shape[0] != expected_len:
            raise ValueError(
                f"Importance array length {importance.shape[0]} "
                f"does not match number of words {expected_len}"
            )

        if (importance < 0).any():
            raise ValueError(
                "Word importance values must be non-negative. "
                f"Found {np.sum(importance < 0)} negative values. "
                f"Minimum value: {np.min(importance)}"
            )

    def _normalize_gender_direction(self, gender_direction: np.ndarray) -> np.ndarray:
        """
        Normalize gender direction to unit vector (PRIVATE).

        Args:
            gender_direction (np.ndarray): Gender direction vector

        Returns:
            np.ndarray: Normalized unit vector

        Raises:
            ValueError: If vector has zero magnitude
        """
        norm = np.linalg.norm(gender_direction)

        if norm < 1e-10:
            raise ValueError(
                "Gender direction vector has zero or near-zero magnitude. "
                "Cannot normalize. Please provide a non-zero gender direction vector."
            )

        return gender_direction / norm

    def _compute_word_biases(
        self, embeddings: np.ndarray, gender_direction: np.ndarray
    ) -> np.ndarray:
        """
        Compute cosine similarity for each word (PRIVATE).

        Args:
            embeddings (np.ndarray): Word embeddings
            gender_direction (np.ndarray): Normalized gender direction

        Returns:
            np.ndarray: Cosine similarities (bias scores per word)
        """
        return np.array(
            [cosine_similarity(word_emb, gender_direction) for word_emb in embeddings]
        )

    def _apply_mask(
        self,
        word_biases: np.ndarray,
        mask: np.ndarray | torch.Tensor,
        expected_len: int,
    ) -> np.ndarray:
        """
        Apply gender words mask to exclude gendered words (PRIVATE).

        Args:
            word_biases (np.ndarray): Computed word biases
            mask (np.ndarray | torch.Tensor): Boolean mask (True = exclude)
            expected_len (int): Expected length for validation

        Returns:
            np.ndarray: Masked word biases (gendered words set to 0)

        Raises:
            ValueError: If mask length doesn't match
            TypeError: If mask is not boolean
        """
        mask = to_numpy(mask)

        if mask.ndim != 1:
            raise ValueError(
                "gender_words_mask must be a rank-1 boolean vector with shape "
                f"(num_words,). Got shape {mask.shape}."
            )

        # Validate mask length
        if mask.shape[0] != expected_len:
            raise ValueError(
                f"Mask length {mask.shape[0]} "
                f"does not match number of words {expected_len}"
            )

        # Validate mask is boolean
        if mask.dtype != bool:
            raise TypeError(
                f"gender_words_mask must be boolean array. "
                f"Got dtype: {mask.dtype}. "
                f"Convert using .astype(bool) if needed."
            )

        # Return zero if all masked (no non-gendered words)
        if mask.all():
            return np.zeros_like(word_biases)

        # Apply mask: True = exclude (multiply by False = 0)
        return word_biases * (~mask)

    def _compute_bias_scores(
        self, word_biases: np.ndarray, importance: np.ndarray
    ) -> Tuple[float, float]:
        """
        Compute final bias scores by weighting and separating (PRIVATE).

        Args:
            word_biases (np.ndarray): Cosine similarities for each word
            importance (np.ndarray): Importance weights for each word

        Returns:
            Tuple[float, float]: (female_bias, male_bias)
        """
        # Weight by importance
        weighted_biases = word_biases * importance

        # Separate into female (positive) and male (negative) components
        female_bias = float(np.sum(weighted_biases[weighted_biases > 0]))
        male_bias = float(np.sum(weighted_biases[weighted_biases < 0]))

        return female_bias, male_bias


def _is_text_sequence(value: object) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes, np.ndarray))
        and all(isinstance(item, str) for item in value)
    )
