"""Contextualized Embedding Association Test (CEAT)."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    import torch

from bias_scope.base import EmbeddingMetric
from bias_scope.embeddings_based._helpers import (
    _ceat_random_effects,
    _validate_tuple_length,
    _weat_effect_components,
)
from bias_scope.embeddings_based.encoder import DEFAULT_EMBEDDING_MODEL
from bias_scope.utils import to_numpy


ContextualGroup = Mapping[str, Any]


class CEAT(EmbeddingMetric):
    """Contextualized Embedding Association Test.

    CEAT accepts precomputed, stimulus-aligned contextual token embeddings.
    Each mapping is stimulus -> array(n_contexts, embedding_dimension). It is
    not a bag of arbitrary vectors and does not encode strings or pool whole
    sentences. On each iteration CEAT retains every stimulus, selects one
    contextual occurrence, computes a WEAT effect size, and pools the results
    with a DerSimonian--Laird random-effects model.

    Selection for each stimulus is without replacement when it has at least
    n_samples contexts, and with replacement otherwise.
    """

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL, *, pooling: str = "cls"):
        """Create CEAT.

        model_name and pooling are inert compatibility attributes. CEAT no
        longer encodes text: callers must supply contextual token embeddings
        extracted with their own stimulus-alignment procedure.
        """
        self.model_name = model_name
        self.pooling = pooling

    def evaluate(
        self,
        target_embeddings: Tuple[ContextualGroup, ContextualGroup],
        attribute_embeddings: Tuple[ContextualGroup, ContextualGroup],
        n_samples: int = 10_000,
        sample_size: Optional[int] = None,
        random_seed: Optional[int] = None,
        model_name: str | None = None,
        return_details: bool = False,
        *,
        pooling: str | None = None,
    ) -> Dict[str, Any]:
        """Compute CEAT's combined effect size (CES).

        target_embeddings is (X, Y), attribute_embeddings is (A, B), and every
        group maps a stimulus to an array of that stimulus's contextual token
        embeddings. The effect_size and ceat_score alias are CES; p_value is a
        two-sided normal random-effects meta-analysis p-value.
        """
        _validate_tuple_length(target_embeddings, "target_embeddings")
        _validate_tuple_length(attribute_embeddings, "attribute_embeddings")
        if not isinstance(n_samples, (int, np.integer)) or n_samples <= 0:
            raise ValueError(f"n_samples must be a positive integer. Got {n_samples!r}.")
        if sample_size is not None:
            raise ValueError(
                "sample_size is not part of canonical CEAT: every iteration "
                "contains every stimulus. Remove sample_size and provide "
                "stimulus-aligned contextual token embeddings."
            )
        if model_name is not None or pooling is not None:
            raise ValueError(
                "CEAT does not encode text or pool sentences. Provide "
                "precomputed stimulus-aligned contextual token embeddings."
            )

        x = self._prepare_group(target_embeddings[0], "target_embeddings[0]")
        y = self._prepare_group(target_embeddings[1], "target_embeddings[1]")
        a = self._prepare_group(attribute_embeddings[0], "attribute_embeddings[0]")
        b = self._prepare_group(attribute_embeddings[1], "attribute_embeddings[1]")
        if len(x) != len(y):
            raise ValueError(
                "Canonical CEAT requires X and Y to contain equal numbers of stimuli. "
                f"Got {len(x)} and {len(y)}."
            )
        dimensions = {matrix.shape[1] for group in (x, y, a, b) for matrix in group.values()}
        if len(dimensions) != 1:
            raise ValueError(
                "All contextual token embeddings must have the same dimension. "
                f"Got dimensions: {sorted(dimensions)}."
            )

        groups = (x, y, a, b)
        families = ("target", "target", "attribute", "attribute")
        selections = [
            {
                stimulus: self._sample_context_indices(
                    len(contexts),
                    n_samples,
                    self._rng_for_stimulus(random_seed, family, stimulus),
                )
                for stimulus, contexts in group.items()
            }
            for group, family in zip(groups, families)
        ]

        effect_sizes = np.empty(n_samples, dtype=float)
        variances = np.empty(n_samples, dtype=float)
        for index in range(n_samples):
            sample_groups = [
                np.stack(
                    [
                        contexts[selections[group_index][stimulus][index]]
                        for stimulus, contexts in group.items()
                    ]
                )
                for group_index, group in enumerate(groups)
            ]
            _, _, pooled_sd, effect_size = _weat_effect_components(*sample_groups)
            effect_sizes[index] = effect_size
            variances[index] = pooled_sd**2

        meta = _ceat_random_effects(effect_sizes, variances)
        z_score = meta["effect_size"] / meta["standard_error"]
        p_value = float(math.erfc(abs(z_score) / math.sqrt(2.0)))
        if not np.isfinite(p_value):
            raise ValueError("CEAT p-value is undefined due to invalid random-effects output.")

        result: Dict[str, Any] = {
            "effect_size": meta["effect_size"],
            "ceat_score": meta["effect_size"],
            "p_value": p_value,
            "standard_error": meta["standard_error"],
            "between_context_variance": meta["between_context_variance"],
            "n_samples": int(n_samples),
            "weat_mean": float(np.mean(effect_sizes)),
            "weat_std": float(np.std(effect_sizes, ddof=1)) if n_samples > 1 else 0.0,
            "weat_variance": float(np.var(effect_sizes, ddof=1)) if n_samples > 1 else 0.0,
            "n_target_group_1": len(x),
            "n_target_group_2": len(y),
            "n_attribute_group_1": len(a),
            "n_attribute_group_2": len(b),
        }
        if return_details:
            result.update(
                {
                    "sample_effect_sizes": effect_sizes.tolist(),
                    "sample_variances": variances.tolist(),
                    "random_effect_weights": meta["random_effect_weights"].tolist(),
                    "fixed_effect_mean": meta["fixed_effect_mean"],
                    "Q": meta["Q"],
                    "sampled_context_indices": {
                        name: {stimulus: indices.tolist() for stimulus, indices in selected.items()}
                        for name, selected in zip(("X", "Y", "A", "B"), selections)
                    },
                }
            )
        return result

    def run(self, *args, seed: int = 42, protocol_kwargs=None, **kwargs):
        """Run CEAT, using and recording ``seed`` for context sampling.

        Without this override, ``run(seed=...)`` never reaches ``random_seed``
        (`evaluate`'s own default falls back to OS entropy, PLAN.md Section 1
        notwithstanding), so every call would resample and report a different
        CES. Mirrors ``WEAT.run`` / ``SEAT.run``.
        """
        effective_random_seed = kwargs.setdefault("random_seed", seed)
        effective_protocol_kwargs = dict(protocol_kwargs or {})
        effective_protocol_kwargs["random_seed"] = effective_random_seed
        return super().run(
            *args,
            seed=seed,
            protocol_kwargs=effective_protocol_kwargs,
            **kwargs,
        )

    def _call_evaluate(self, *args, **kwargs):
        """Stash the random-effects standard error so `_interval` can use it."""
        raw = super()._call_evaluate(*args, **kwargs)
        self._last_standard_error = raw.get("standard_error") if isinstance(raw, dict) else None
        return raw

    def _interval(self, score, per_item, n, ci, seed):
        """CEAT's own uncertainty is SE(CES) from the random-effects model
        (Guo & Caliskan, Appendix "Random-Effects Model Details"), not a
        Hedges-Olkin interval on target-group sizes: CEAT's "sample" is the
        `n_samples` drawn context-combinations, not the stimulus counts.
        """
        if ci == "none":
            return None, "none", None
        se = getattr(self, "_last_standard_error", None)
        if se is None:
            return super()._interval(score, per_item, n, ci, seed)

        from bias_scope.stats import Z_95

        return (score - Z_95 * se, score + Z_95 * se), "random_effects", None

    @staticmethod
    def _count_items(details, per_item):
        """CEAT's `n` is the number of sampled context-combinations (the
        meta-analysis sample size), not the number of target stimuli.
        """
        if per_item is not None:
            return len(per_item)
        n_samples = details.get("n_samples")
        if isinstance(n_samples, int) and n_samples > 0:
            return n_samples
        return EmbeddingMetric._count_items(details, per_item)

    @staticmethod
    def _sample_context_indices(
        n_contexts: int, n_samples: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Select contextual occurrences for one stimulus."""
        return rng.choice(n_contexts, size=n_samples, replace=n_contexts < n_samples)

    @staticmethod
    def _rng_for_stimulus(
        random_seed: int | None, family: str, stimulus: str
    ) -> np.random.Generator:
        """Create a stable per-stimulus generator independent of CEAT side."""
        if random_seed is None:
            random_seed = int(np.random.SeedSequence().entropy)
        material = f"CEAT-v1\\0{int(random_seed)}\\0{family}\\0{stimulus}".encode("utf-8")
        derived_seed = int.from_bytes(hashlib.sha256(material).digest()[:16], "big")
        return np.random.default_rng(derived_seed)

    @staticmethod
    def _prepare_group(group: Any, name: str) -> dict[str, np.ndarray]:
        """Validate and normalize one CEAT stimulus group (PRIVATE)."""
        if not isinstance(group, Mapping):
            raise ValueError(
                f"{name} must be a non-empty mapping of stimulus -> contextual "
                "token embedding matrix; flat arrays and raw strings cannot "
                "represent canonical CEAT."
            )
        if not group:
            raise ValueError(f"{name} cannot be empty.")
        prepared: dict[str, np.ndarray] = {}
        for stimulus, contexts in group.items():
            if not isinstance(stimulus, str):
                raise ValueError(f"{name} stimulus keys must be strings.")
            array = to_numpy(contexts)
            if array.ndim != 2:
                raise ValueError(
                    f"{name}[{stimulus!r}] must have shape (n_contexts, embedding_dim), "
                    f"got shape {array.shape}."
                )
            if array.shape[0] == 0:
                raise ValueError(f"{name}[{stimulus!r}] has zero contextual embeddings.")
            if array.shape[1] == 0:
                raise ValueError(f"{name}[{stimulus!r}] has zero embedding dimensions.")
            if not np.issubdtype(array.dtype, np.number):
                raise ValueError(f"{name}[{stimulus!r}] must contain numeric embeddings.")
            array = np.asarray(array, dtype=float)
            if not np.isfinite(array).all():
                raise ValueError(f"{name}[{stimulus!r}] contains NaN or Inf values.")
            prepared[stimulus] = array
        return prepared
