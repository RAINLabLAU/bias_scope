"""Sentence Encoder Association Test (SEAT)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Sequence, Tuple

import numpy as np

if TYPE_CHECKING:  # torch is an optional extra; used in annotations only
    import torch

from bias_scope.base import EmbeddingMetric
from bias_scope.embeddings_based.encoder import DEFAULT_EMBEDDING_MODEL
from bias_scope.embeddings_based.weat import WEAT

#: May et al. 2019, Appendix A: the sampled p-value draws 99,999 partitions and
#: hallucinates one more hit, so it can never fall below 1 / 100,000. WEAT's
#: ``tie_policy="conservative"`` branch already adds the observed partition, so
#: 100,000 total evaluations reproduce that 1e-5 precision floor exactly.
SEAT_PERMUTATION_SAMPLES = 100_000


class SEAT(EmbeddingMetric):
    """
    Sentence Encoder Association Test.

    Adapts WEAT to contextualized sentence embeddings. Uses the same effect
    size calculation as WEAT (mean-of-cosines association, ``ddof=1``
    standardised mean difference), but operates on sentence-level
    representations instead of static word embeddings.

    The permutation p-value follows May et al.'s Appendix A rather than
    Caliskan's: the inequality is **non-strict** (``Pr[s(Xi,Yi,A,B) >=
    s(X,Y,A,B)]``, "the more conservative non-strict inequality", because "the
    equality has positive probability" in the nonparametric version), and the
    sampled estimate is floored at 1e-5. That maps to ``tie_policy="conservative"``
    and ``n_permutation_samples=100_000`` here, both of which are the SEAT
    defaults. Pass ``tie_policy="strict"`` for Caliskan's ``>`` convention.

    Canonical input is precomputed sentence embeddings, or fully formed SEAT
    sentence stimuli that callers construct themselves. BiasScope does not
    generate May et al.'s semantically bleached templates automatically, and it
    does not reproduce their per-encoder pooling table.

    Reference
    ---------
    May, C., Wang, A., Bordia, S., Bowman, S. R., & Rudinger, R. (2019).
    On measuring social biases in sentence encoders. NAACL-HLT 2019.
    https://arxiv.org/abs/1903.10561

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
        pooling: str = "cls",
    ):
        """
        Initialize SEAT.

        Args:
            model_name (str): Default SentenceTransformer/Hugging Face model used
                when raw text inputs need to be embedded automatically. This
                default is used unless ``evaluate(..., model_name=...)`` overrides
                it for a single call.
            pooling (str): 'cls' (default) uses Hugging Face CLS pooling for
                raw strings; 'mean' uses sentence-transformer pooling. Both
                are BiasScope convenience paths, not an exact reproduction of
                May et al.'s original per-encoder extraction setup.
        """
        if pooling not in ("mean", "cls"):
            raise ValueError(f"pooling must be 'mean' or 'cls', got {pooling!r}")
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
        n_permutation_samples: int = SEAT_PERMUTATION_SAMPLES,
        permutation_seed: int = 42,
        tie_policy: str = "conservative",
    ) -> float | Dict[str, float]:
        """
        Evaluate SEAT score.

        Args:
            target_embeddings (Tuple[np.ndarray | torch.Tensor, ...]):
                target group sentence embeddings
            attribute_embeddings (Tuple[np.ndarray | torch.Tensor, ...]):
                attribute group sentence embeddings
            model_name (str | None): SentenceTransformer/Hugging Face model used
                when text inputs are provided. If omitted, uses the ``model_name``
                configured on ``__init__``. If passed here, it overrides the
                instance default for this call only.
            n_permutation_samples (int): Positive total number of partition
                evaluations in sampled p-values. Ignored when exact enumeration
                is used. Defaults to 100,000 (May et al. Appendix A).
            permutation_seed (int): RNG seed for sampled permutations.
            tie_policy (str): ``"conservative"`` (default, May et al.'s non-strict
                ``>=``) or ``"strict"`` (Caliskan's ``>``).

        Returns:
            float: SEAT effect size score

        Raises:
            ValueError: If inputs are invalid

        Notes:
            **Method:**
            SEAT uses the identical effect size calculation as WEAT, applied to
            sentence embeddings instead of word embeddings. This method
            delegates the mathematics to WEAT internally. The one deliberate
            difference (May et al. Appendix A) is the non-strict permutation
            inequality, carried through as ``tie_policy="conservative"``.

            **Input Structure:**
            - target_embeddings: (target_group1, target_group2)
              - Each array shape: (n_sentences, embedding_dim)
              - Example: precomputed representations of caller-supplied
                sentences with male vs female terms
            - attribute_embeddings: (attribute_group1, attribute_group2)
              - Each array shape: (n_sentences, embedding_dim)
              - Example: Sentences with career vs family words

            **Typical Workflow:**
            1. Construct appropriate semantically bleached sentence stimuli.
            2. Encode them with the chosen sentence encoder.
            3. Pass the precomputed sentence embeddings to SEAT.

            Raw string sequences are also accepted as a BiasScope convenience.
            They are encoded directly; BiasScope does not construct templates
            around individual words. For exact experimental reproduction,
            supply precomputed sentence embeddings from the intended protocol.

            evaluate() returns the effect size by default. With
            return_details=True, it also returns the permutation p-value and
            metadata; run() exposes that p-value through BiasResult.p_value.

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

        # SEAT is WEAT applied to sentence embeddings. It inherits WEAT's effect
        # size and ``ddof=1``, but May et al. (Appendix A) count
        # s(Xi,Yi,A,B) >= s(X,Y,A,B) rather than Caliskan's strict >, so the
        # default ``tie_policy`` differs from WEAT's.
        weat_instance = WEAT(model_name=self.model_name, pooling=self.pooling)
        details = weat_instance.evaluate(
            target_embeddings,
            attribute_embeddings,
            model_name=effective_model_name,
            pooling=effective_pooling,
            return_details=True,
            n_permutation_samples=n_permutation_samples,
            permutation_seed=permutation_seed,
            tie_policy=tie_policy,
        )
        score = float(details["effect_size"])
        if not return_details:
            return score

        # Carry WEAT's group sizes and p-value through so ``run()`` can form a
        # Hedges-Olkin interval and pass its ``n > 0`` guard. Drop the internal
        # ``weat_score`` alias; SEAT reports ``seat_score``.
        result = {key: value for key, value in details.items() if key != "weat_score"}
        result["seat_score"] = score
        result["effect_size"] = score
        return result

    def run(self, *args, seed: int = 42, protocol_kwargs=None, **kwargs):
        """Run SEAT, using and recording ``seed`` for sampled permutations.

        Mirrors ``WEAT.run``: without threading the seed into
        ``permutation_seed``, ``run(seed=...)`` would not reach the permutation
        RNG and the protocol block would not record it.
        """
        effective_permutation_seed = kwargs.setdefault("permutation_seed", seed)
        effective_protocol_kwargs = dict(protocol_kwargs or {})
        effective_protocol_kwargs["permutation_seed"] = effective_permutation_seed
        return super().run(
            *args,
            seed=seed,
            protocol_kwargs=effective_protocol_kwargs,
            **kwargs,
        )
