"""Word Embedding Association Test (WEAT)."""

from __future__ import annotations

import itertools
import math
from typing import TYPE_CHECKING, Dict, Sequence, Tuple

import numpy as np

if TYPE_CHECKING:  # torch is an optional extra; used in annotations only
    import torch

from bias_scope.base import EmbeddingMetric
from bias_scope.embeddings_based._helpers import (
    _compute_similarity_measure,
    _validate_embedding_dimensions,
    _validate_tuple_length,
)
from bias_scope.embeddings_based.encoder import (
    DEFAULT_EMBEDDING_MODEL,
    _resolve_embedding_pair,
)
from bias_scope.utils import to_numpy

#: Enumerate partitions exactly up to this many; sample beyond it. C(16,8) is
#: 12,870, so Caliskan's own 8-per-group tests are always exact.
EXACT_PERMUTATION_LIMIT = 100_000

#: Samples used when the partition count exceeds the limit.
DEFAULT_PERMUTATION_SAMPLES = 10_000



def _partition_statistic(pooled: np.ndarray, left_indices) -> float:
    """s(Xi, Yi, A, B) for one partition: sum(left) - sum(the rest)."""
    mask = np.zeros(pooled.size, dtype=bool)
    mask[list(left_indices)] = True
    return float(pooled[mask].sum() - pooled[~mask].sum())


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
    >>> from bias_scope.embeddings_based import WEAT
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
    >>> # Evaluate bias
    >>> score = weat.evaluate(
    ...     (male_names, female_names),
    ...     (career_words, family_words)
    ... )
    >>> print(f"Gender-career bias effect size: {score:.3f}")
    >>>
    >>> print(weat.category)    # "embedding"
    """

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        *,
        pooling: str = "mean",
    ):
        """
        Initialize WEAT.

        Args:
            model_name (str): Default SentenceTransformer/Hugging Face model used
                when raw text inputs need to be embedded automatically. This
                default is used unless ``evaluate(..., model_name=...)`` overrides
                it for a single call.
            pooling (str): 'mean' (default, sentence-transformers) or 'cls'
                (raw `[CLS]` embedding from the underlying LM — matches Caliskan/May
                paper protocols when using bert-base-*). Only applies when raw
                text inputs are provided; ignored for precomputed arrays.
        """
        self.model_name = model_name
        self.pooling = pooling

    def evaluate(
        self,
        target_embeddings: Tuple[
            np.ndarray | torch.Tensor | Sequence[str],
            np.ndarray | torch.Tensor | Sequence[str],
        ],
        attribute_embeddings: Tuple[
            np.ndarray | torch.Tensor | Sequence[str],
            np.ndarray | torch.Tensor | Sequence[str],
        ],
        model_name: str | None = None,
        return_details: bool = False,
        *,
        pooling: str | None = None,
        n_permutation_samples: int = DEFAULT_PERMUTATION_SAMPLES,
        permutation_seed: int = 42,
    ) -> float | Dict[str, float]:
        """
        Evaluate WEAT effect size.

        Args:
            target_embeddings (Tuple[np.ndarray | torch.Tensor, ...]):
                target group word embeddings
            attribute_embeddings (Tuple[np.ndarray | torch.Tensor, ...]):
                attribute group word embeddings
            model_name (str | None): SentenceTransformer/Hugging Face model used
                when text inputs are provided. If omitted, uses the ``model_name``
                configured on ``__init__``. If passed here, it overrides the
                instance default for this call only.

        Returns:
            float: WEAT effect size score

        Raises:
            ValueError: If inputs are invalid

        Notes:
            **Input Structure:**
            - target_embeddings: (target_group1, target_group2)
              - Each array shape: (n_words, embedding_dim)
              - Example: (male_names, female_names)
            - attribute_embeddings: (attribute_group1, attribute_group2)
              - Each array shape: (n_words, embedding_dim)
              - Example: (career_words, family_words)

            **Effect Size Formula:**
                d = (mean(s(X, A, B)) - mean(s(Y, A, B))) / std(s(X ∪ Y, A, B))

            Where:
                s(w, A, B) = mean(cos(w, a) for a in A) - mean(cos(w, b) for b in B)
                X, Y = target groups
                A, B = attribute groups
                cos = cosine similarity

            **Interpretation:**
            - Positive values: target_group1 associates more with attribute_group1
            - Negative values: target_group1 associates more with attribute_group2
            - Larger absolute values indicate stronger bias

        Examples:
            >>> import numpy as np
            >>> weat = WEAT()
            >>>
            >>> # Simple 2D example
            >>> targets1 = np.array([[1.0, 0.0], [0.9, 0.1]])
            >>> targets2 = np.array([[0.0, 1.0], [0.1, 0.9]])
            >>> attrs1 = np.array([[1.0, 0.0], [0.95, 0.05]])
            >>> attrs2 = np.array([[0.0, 1.0], [0.05, 0.95]])
            >>>
            >>> score = weat.evaluate((targets1, targets2), (attrs1, attrs2))
            >>> print(f"Effect size: {score:.3f}")  # Positive score expected
        """
        effective_model_name = model_name or self.model_name
        effective_pooling = pooling or self.pooling

        # Validate tuple structure
        _validate_tuple_length(target_embeddings, "target_embeddings")
        _validate_tuple_length(attribute_embeddings, "attribute_embeddings")

        target_embeddings = _resolve_embedding_pair(
            target_embeddings,
            model_name=effective_model_name,
            pooling=effective_pooling,
        )
        attribute_embeddings = _resolve_embedding_pair(
            attribute_embeddings,
            model_name=effective_model_name,
            pooling=effective_pooling,
        )

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
        cos_target1 = [_compute_similarity_measure(w, attr1, attr2) for w in target1]
        cos_target2 = [_compute_similarity_measure(w, attr1, attr2) for w in target2]

        # Union for standard deviation
        union_targets = np.concatenate([target1, target2])
        cos_union = [
            _compute_similarity_measure(w, attr1, attr2) for w in union_targets
        ]

        # Compute and return effect size (private method)
        score = self._compute_effect_size(cos_target1, cos_target2, cos_union)
        if not return_details:
            return score

        p_value, exact, num_partitions, note = self._permutation_test(
            cos_target1, cos_target2, n_permutation_samples, permutation_seed
        )
        return {
            "weat_score": score,
            "effect_size": score,
            "p_value": p_value,
            "p_value_exact": exact,
            "num_partitions": num_partitions,
            "p_value_note": note,
            "n_target_group_1": float(len(target1)),
            "n_target_group_2": float(len(target2)),
            "n_attribute_group_1": float(len(attr1)),
            "n_attribute_group_2": float(len(attr2)),
        }

    @staticmethod
    def _permutation_test(
        scores1: list,
        scores2: list,
        n_samples: int,
        seed: int,
    ) -> tuple:
        """
        One-sided permutation p-value (Caliskan et al. 2017).

        The paper: let {(Xi, Yi)} be all partitions of X u Y into two sets of
        equal size; the p-value is Pr_i[s(Xi, Yi, A, B) > s(X, Y, A, B)], where
        s(X, Y, A, B) = sum_{x in X} s(x,A,B) - sum_{y in Y} s(y,A,B).

        Enumerated exactly when there are few enough partitions, as the
        reference implementation does (`sent-bias/sentbias/weat.py:82-152`);
        sampled otherwise. The observed partition is always counted, so the
        p-value has a floor of 1/num_partitions rather than reaching zero.

        Returns:
            (p_value, exact, num_partitions, note)
        """
        # s(w, A, B) is already computed per word; the test statistic is a sum,
        # so partitions can be evaluated on the association scores directly.
        n1, n2 = len(scores1), len(scores2)
        if n1 != n2:
            return (
                None,
                False,
                0,
                "Caliskan's permutation test partitions X u Y into two sets of "
                f"equal size, but |X| = {n1} and |Y| = {n2}. No p-value is "
                "defined for unequal target sets.",
            )

        pooled = np.asarray(list(scores1) + list(scores2), dtype=float)
        observed = float(np.sum(scores1) - np.sum(scores2))
        total = len(pooled)
        num_partitions = math.comb(total, n1)

        if num_partitions <= EXACT_PERMUTATION_LIMIT:
            at_least = sum(
                1
                for left in itertools.combinations(range(total), n1)
                if _partition_statistic(pooled, left) >= observed - 1e-12
            )
            return (
                at_least / num_partitions,
                True,
                num_partitions,
                f"Exact test over all {num_partitions} partitions.",
            )

        rng = np.random.default_rng(seed)
        indices = np.arange(total)
        at_least = 1  # count the observed partition itself
        for _ in range(n_samples):
            rng.shuffle(indices)
            if _partition_statistic(pooled, indices[:n1]) >= observed - 1e-12:
                at_least += 1
        return (
            at_least / (n_samples + 1),
            False,
            num_partitions,
            f"Sampled test: {n_samples} of {num_partitions} partitions "
            "(too many to enumerate).",
        )

    def _compute_effect_size(
        self, scores1: list, scores2: list, scores_union: list
    ) -> float:
        """
        Compute WEAT effect size (PRIVATE).

        Calculates: (mean1 - mean2) / std(union)

        Args:
            scores1 (list): Similarity scores for target group 1
            scores2 (list): Similarity scores for target group 2
            scores_union (list): Combined similarity scores for standard deviation calculation

        Returns:
            float: Effect size

        Raises:
            ValueError: If standard deviation is zero or insufficient data
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
