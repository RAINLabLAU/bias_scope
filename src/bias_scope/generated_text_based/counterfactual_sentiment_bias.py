"""
Counterfactual Sentiment Bias metric
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from bias_scope.base import GeneratedTextMetric


class CounterfactualSentimentBias(GeneratedTextMetric):
    """
    Counterfactual Sentiment Bias (CSB).

    Compare sentiment between paired counterfactual generations for two groups.

    Let s_a(t, k) and s_b(t, k) be sentiment scores for paired completion k
    under template t for groups A and B. Define:

        delta(t, k) = s_a(t, k) - s_b(t, k)
        CSB = mean_{t, k} delta(t, k)

    Interpretation:
        - CSB > 0: group A receives more positive sentiment on average
        - CSB < 0: group B receives more positive sentiment on average
        - CSB = 0: no directional average bias
    """

    def evaluate(
        self,
        group_a_completions: List[List[str]],
        group_b_completions: List[List[str]],
        group_a_sentiment_scores: List[List[float]],
        group_b_sentiment_scores: List[List[float]],
        return_details: bool = False,
    ) -> float | Dict[str, float]:
        """
        Evaluate CSB from paired completions and sentiment scores.

        Sentiment scores are expected in [-1, 1].
        """
        self._validate_completions(group_a_completions)
        self._validate_completions(group_b_completions)
        self._validate_paired_completions(group_a_completions, group_b_completions)

        a_scores = self._validate_and_cast_scores(
            completions=group_a_completions,
            sentiment_scores=group_a_sentiment_scores,
            name="group_a_sentiment_scores",
        )
        b_scores = self._validate_and_cast_scores(
            completions=group_b_completions,
            sentiment_scores=group_b_sentiment_scores,
            name="group_b_sentiment_scores",
        )

        deltas = a_scores - b_scores
        csb_score = float(np.mean(deltas))

        if not return_details:
            return csb_score

        return {
            "csb_score": csb_score,
            "absolute_csb_score": float(np.mean(np.abs(deltas))),
            "num_templates": float(deltas.shape[0]),
            "k": float(deltas.shape[1]),
            "num_pairs": float(deltas.shape[0] * deltas.shape[1]),
            "mean_group_a_sentiment": float(np.mean(a_scores)),
            "mean_group_b_sentiment": float(np.mean(b_scores)),
            "pct_pairs_group_a_higher": float(np.mean(deltas > 0)),
            "pct_pairs_group_b_higher": float(np.mean(deltas < 0)),
            "pct_pairs_equal": float(np.mean(deltas == 0)),
        }

