"""Sentence Encoder Association Test (SEAT)."""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np

from bias_scope.base import EmbeddingMetric
from bias_scope.embeddings_based.encoder import DEFAULT_EMBEDDING_MODEL
from bias_scope.embeddings_based.weat import WEAT


class SEAT(EmbeddingMetric):
    """
    Sentence Encoder Association Test.

    Adapts WEAT to contextualized sentence embeddings. Uses the same
    effect size calculation as WEAT, but operates on sentence-level
    representations instead of static word embeddings.

    SEAT generates sentence embeddings using templates and measures
    bias through the same differential association test as WEAT.

    Reference
    ---------
    May, C., Wang, A., Bordia, S., Bowman, S. R., & Rudinger, R. (2019).
    On measuring social biases in sentence encoders. NAACL-HLT 2019.

    Examples
    --------
    >>> from bias_scope.embeddings_based import SEAT
    >>> import numpy as np
    >>>
    >>> # Test with BERT sentence embeddings (768-dimensional)
    >>> seat = SEAT()
    >>>
    >>> # Sentences encoded with "This is [WORD]" template
    >>> male_sentences = np.random.randn(10, 768)
    >>> female_sentences = np.random.randn(10, 768)
    >>> career_sentences = np.random.randn(8, 768)
    >>> family_sentences = np.random.randn(8, 768)
    >>>
    >>> score = seat.evaluate(
    ...     (male_sentences, female_sentences),
    ...     (career_sentences, family_sentences)
    ... )
    >>> print(f"Gender-career bias (SEAT): {score:.3f}")
    """

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        *,
        pooling: str = "mean",
    ):
        """
        Initialize SEAT.

        Args:
            model_name (str): Default SentenceTransformer/Hugging Face model used
                when raw text inputs need to be embedded automatically. This
                default is used unless ``evaluate(..., model_name=...)`` overrides
                it for a single call.
            pooling (str): 'mean' (default) or 'cls'. Use 'cls' with a raw
                bert-base-* model name to match May 2019's SEAT protocol.
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
    ) -> float | Dict[str, float]:
        """
        Evaluate SEAT score.

        Args:
            target_embeddings (Tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor]): target group sentence embeddings
            attribute_embeddings (Tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor]): attribute group sentence embeddings
            model_name (str | None): SentenceTransformer/Hugging Face model used
                when text inputs are provided. If omitted, uses the ``model_name``
                configured on ``__init__``. If passed here, it overrides the
                instance default for this call only.

        Returns:
            float: SEAT effect size score

        Raises:
            ValueError: If inputs are invalid

        Notes:
            **Method:**
            SEAT uses the identical calculation as WEAT, simply applied to
            sentence embeddings instead of word embeddings. This method
            delegates to WEAT internally.

            **Input Structure:**
            - target_embeddings: (target_group1, target_group2)
              - Each array shape: (n_sentences, embedding_dim)
              - Example: Sentences with male vs female terms
            - attribute_embeddings: (attribute_group1, attribute_group2)
              - Each array shape: (n_sentences, embedding_dim)
              - Example: Sentences with career vs family words

            **Typical Workflow:**
            1. Create sentences using templates (e.g., "This is [WORD]")
            2. Encode with sentence encoder (BERT, RoBERTa, etc.)
            3. Pass sentence embeddings to SEAT

        Examples:
            >>> import numpy as np
            >>> from bias_scope.embeddings_based import SEAT
            >>>
            >>> seat = SEAT()
            >>>
            >>> # BERT [CLS] token embeddings (768-dim)
            >>> male_sent = np.random.randn(5, 768)
            >>> female_sent = np.random.randn(5, 768)
            >>> career_sent = np.random.randn(5, 768)
            >>> family_sent = np.random.randn(5, 768)
            >>>
            >>> score = seat.evaluate(
            ...     (male_sent, female_sent),
            ...     (career_sent, family_sent)
            ... )
        """
        effective_model_name = model_name or self.model_name
        effective_pooling = pooling or self.pooling

        # SEAT is just WEAT applied to sentence embeddings
        weat_instance = WEAT(model_name=self.model_name, pooling=self.pooling)
        score = weat_instance.evaluate(
            target_embeddings,
            attribute_embeddings,
            model_name=effective_model_name,
            pooling=effective_pooling,
        )
        if return_details:
            return {"seat_score": float(score), "effect_size": float(score)}
        return score
