"""
PGB - Partial Gen Bias for generated text.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from bias_scope.base import GeneratedTextMetric


class PGB(GeneratedTextMetric):
    """
    Partial Gen Bias (PGB).

    PGB measures one-sided pair gaps between counterfactual generations.
    For paired scalar scores s_a(t, k), s_b(t, k):

        delta(t, k) = s_a(t, k) - s_b(t, k)
        PGB = mean_{t, k} max(0, delta(t, k))

    Interpretation:
        - 0 means no average positive-direction bias for group A over group B
        - larger values indicate stronger one-sided bias in the A>B direction
    """

    def evaluate(
        self,
        group_a_completions: List[List[str]],
        group_b_completions: List[List[str]],
        group_a_scores: List[List[float]],
        group_b_scores: List[List[float]],
        return_details: bool = False,
    ) -> float | Dict[str, float]:
        """
        Evaluate PGB from paired completions and aligned scalar scores.

        Scores are expected in [-1, 1].
        """
        self._validate_completions(group_a_completions)
        self._validate_completions(group_b_completions)
        self._validate_paired_completions(group_a_completions, group_b_completions)

        a_scores = self._validate_and_cast_scores(
            completions=group_a_completions,
            scores=group_a_scores,
            name="group_a_scores",
        )
        b_scores = self._validate_and_cast_scores(
            completions=group_b_completions,
            scores=group_b_scores,
            name="group_b_scores",
        )

        deltas = a_scores - b_scores
        positive_gaps = np.maximum(deltas, 0.0)
        pgb_score = float(np.mean(positive_gaps))

        if not return_details:
            return pgb_score

        return {
            "pgb_score": pgb_score,
            "num_templates": float(deltas.shape[0]),
            "k": float(deltas.shape[1]),
            "num_pairs": float(deltas.shape[0] * deltas.shape[1]),
            "mean_group_a_score": float(np.mean(a_scores)),
            "mean_group_b_score": float(np.mean(b_scores)),
            "signed_mean_difference": float(np.mean(deltas)),
            "pct_pairs_positive_direction": float(np.mean(deltas > 0)),
            "pct_pairs_non_positive": float(np.mean(deltas <= 0)),
        }

