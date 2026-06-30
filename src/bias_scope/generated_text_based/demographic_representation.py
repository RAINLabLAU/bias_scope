"""Demographic Representation - Diversity and distribution analysis."""

from typing import Dict, Literal, Mapping, Optional, Sequence

import numpy as np

from bias_scope.base import GeneratedTextMetric
from bias_scope.generated_text_based._helpers import (
    EPSILON,
    count_lexicon_mentions,
    normalize_lexicon,
    tokenize,
)


class DemographicRepresentation(GeneratedTextMetric):
    """
    Demographic Representation.

    Measures representation and diversity of demographic groups in generated text.
    Computes distribution of group mentions, diversity metrics (entropy, Gini),
    and optional comparison to reference distributions.

    Reference
    ---------
    Lahoti, P., Beutel, A., Chen, J., Lee, K., Prost, F., Thain, N., Wang, X., &
    Chi, E. (2023). Fairness Indicators: Scalable Infrastructure for Fair ML Systems.
    Google Research.

    Examples
    --------
    >>> from bias_scope.generated_text_based import DemographicRepresentation
    >>>
    >>> dr = DemographicRepresentation()
    >>>
    >>> generations = [
    ...     "The man walked to work.",
    ...     "The woman drove home.",
    ...     "A man and woman talked."
    ... ]
    >>>
    >>> group_lexicons = {
    ...     'male': ['man', 'he', 'his'],
    ...     'female': ['woman', 'she', 'her']
    ... }
    >>>
    >>> result = dr.evaluate(
    ...     generations=generations,
    ...     group_lexicons=group_lexicons,
    ...     normalize='mentions'
    ... )
    >>>
    >>> print(f"Distribution: {result['distribution']}")
    >>> print(f"Entropy: {result['diversity']['entropy']:.3f}")
    """

    def evaluate(
        self,
        generations: Sequence[str],
        group_lexicons: Mapping[str, Sequence[str]],
        *,
        normalize: Literal["mentions", "tokens"] = "mentions",
        compare_to: Optional[Mapping[str, float]] = None,
        smoothing: float = 1e-12,
        return_details: bool = False,
    ) -> Dict:
        """
        Evaluate Demographic Representation.

        Args:
            generations (Sequence[str]): Generated texts to analyze
            group_lexicons (Mapping[str, Sequence[str]]): Group name -> terms mapping
            normalize (Literal["mentions", "tokens"]): Normalization mode (default: "mentions")
            compare_to (Optional[Mapping[str, float]]): Reference distribution for comparison
            smoothing (float): Smoothing for KL/JS divergence (default: 1e-12)

        Returns:
            Dict: Results including counts, distribution, and diversity metrics

        Raises:
            ValueError: If inputs are invalid

        Notes:
            **Normalization Modes:**
            - "mentions": p(g) = count(g) / sum of all group counts
            - "tokens": p(g) = count(g) / total tokens in corpus

            **Diversity Metrics:**
            - entropy: H = -Σ p(g) log(p(g))
            - normalized_entropy: H / log(K) where K = number of groups
            - gini_impurity: 1 - Σ p(g)²

            **Reference Comparison (if compare_to provided):**
            - KL divergence: KL(p||q) = Σ p(g) log(p(g)/q(g))
            - Jensen-Shannon divergence: symmetric version of KL

            **Return Dictionary Structure:**
            {
                'metric': 'DemographicRepresentation',
                'category': 'generated_text',
                'groups': List[str],
                'counts': {<group>: int},
                'total_mentions': int,
                'distribution': {<group>: float},
                'diversity': {
                    'entropy': float,
                    'normalized_entropy': float,
                    'gini_impurity': float
                },
                'reference': {
                    'provided': bool,
                    'distribution': {<group>: float} or None,
                    'kl_pq': float or None,
                    'jsd': float or None
                }
            }
        """
        # Validate inputs
        generations_list = self._validate_texts(generations, "generations")

        if not isinstance(group_lexicons, Mapping):
            raise TypeError("group_lexicons must be a Mapping (dict)")

        if len(group_lexicons) == 0:
            raise ValueError("group_lexicons cannot be empty")

        for group_name, terms in group_lexicons.items():
            self._validate_texts(terms, f"group_lexicons['{group_name}']")

        if normalize not in {"mentions", "tokens"}:
            raise ValueError(
                f"normalize must be 'mentions' or 'tokens', got '{normalize}'"
            )

        if smoothing <= 0:
            raise ValueError(f"smoothing must be > 0, got {smoothing}")

        # Normalize group lexicons
        normalized_groups = {
            name: normalize_lexicon(terms) for name, terms in group_lexicons.items()
        }

        # Count mentions
        group_counts = {name: 0 for name in normalized_groups.keys()}
        total_tokens = 0

        for text in generations_list:
            tokens = tokenize(text)
            total_tokens += len(tokens)

            for group_name, group_lex in normalized_groups.items():
                count = count_lexicon_mentions(tokens, group_lex)
                group_counts[group_name] += count

        # Compute distribution
        total_mentions = sum(group_counts.values())

        if normalize == "mentions":
            if total_mentions == 0:
                raise ValueError(
                    "No group mentions found in generations. Cannot compute distribution."
                )
            distribution = {
                name: float(count / total_mentions)
                for name, count in group_counts.items()
            }
        else:  # tokens
            if total_tokens == 0:
                raise ValueError(
                    "No tokens found in generations. Cannot compute distribution."
                )
            distribution = {
                name: float(count / total_tokens)
                for name, count in group_counts.items()
            }

        # Compute diversity metrics
        K = len(group_counts)
        probs = np.array(list(distribution.values()))

        # Entropy
        # Add epsilon to avoid log(0)
        probs_safe = np.where(probs > 0, probs, EPSILON)
        entropy = float(-np.sum(probs_safe * np.log(probs_safe)))

        # Normalized entropy
        if K > 1:
            max_entropy = np.log(K)
            normalized_entropy = float(entropy / max_entropy)
        else:
            normalized_entropy = 0.0

        # Gini impurity
        gini_impurity = float(1.0 - np.sum(probs**2))

        # Reference comparison
        reference_result = {
            "provided": False,
            "distribution": None,
            "kl_pq": None,
            "jsd": None,
        }

        if compare_to is not None:
            if not isinstance(compare_to, Mapping):
                raise TypeError("compare_to must be a Mapping (dict)")

            # Validate keys match
            ref_keys = set(compare_to.keys())
            group_keys = set(group_counts.keys())

            if ref_keys != group_keys:
                missing = group_keys - ref_keys
                extra = ref_keys - group_keys
                msg = "compare_to keys must match group_lexicons keys."
                if missing:
                    msg += f" Missing: {missing}."
                if extra:
                    msg += f" Extra: {extra}."
                raise ValueError(msg)

            # Validate sums to ~1
            ref_sum = sum(compare_to.values())
            if not (0.99 <= ref_sum <= 1.01):
                raise ValueError(
                    f"compare_to distribution must sum to ~1, got {ref_sum:.4f}"
                )

            # Compute KL(p||q)
            q_probs = np.array([compare_to[name] for name in group_counts.keys()])
            p_probs = np.array([distribution[name] for name in group_counts.keys()])

            # Add smoothing
            p_smooth = p_probs + smoothing
            q_smooth = q_probs + smoothing

            # Normalize
            p_smooth = p_smooth / p_smooth.sum()
            q_smooth = q_smooth / q_smooth.sum()

            kl_pq = float(np.sum(p_smooth * np.log(p_smooth / q_smooth)))

            # Compute Jensen-Shannon divergence
            m = (p_smooth + q_smooth) / 2
            jsd = float(
                0.5 * np.sum(p_smooth * np.log(p_smooth / m))
                + 0.5 * np.sum(q_smooth * np.log(q_smooth / m))
            )

            reference_result = {
                "provided": True,
                "distribution": dict(compare_to),
                "kl_pq": kl_pq,
                "jsd": jsd,
            }

        # Return results
        return {
            "metric": "DemographicRepresentation",
            "category": self.category,
            "groups": list(group_counts.keys()),
            "counts": group_counts,
            "total_mentions": total_mentions,
            "distribution": distribution,
            "diversity": {
                "entropy": entropy,
                "normalized_entropy": normalized_entropy,
                "gini_impurity": gini_impurity,
            },
            "reference": reference_result,
        }
