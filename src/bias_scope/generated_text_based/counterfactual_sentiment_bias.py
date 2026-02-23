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

    def _validate_paired_completions(
        self,
        group_a_completions: List[List[str]],
        group_b_completions: List[List[str]],
    ) -> None:
        if len(group_a_completions) != len(group_b_completions):
            raise ValueError(
                "group_a_completions and group_b_completions must have the same number of templates. "
                f"Got {len(group_a_completions)} and {len(group_b_completions)}."
            )
        for i, (a_template, b_template) in enumerate(
            zip(group_a_completions, group_b_completions)
        ):
            if len(a_template) != len(b_template):
                raise ValueError(
                    "group_a_completions and group_b_completions must be shape-aligned. "
                    f"Template index {i} has K={len(a_template)} vs K={len(b_template)}."
                )

    def _validate_and_cast_scores(
        self,
        completions: List[List[str]],
        sentiment_scores: List[List[float]],
        name: str,
    ) -> np.ndarray:
        if len(sentiment_scores) != len(completions):
            raise ValueError(
                f"{name} must have the same number of templates as its completions. "
                f"Got {len(sentiment_scores)} and {len(completions)}."
            )

        rows: List[List[float]] = []
        for i, (template_completions, template_scores) in enumerate(
            zip(completions, sentiment_scores)
        ):
            if len(template_scores) != len(template_completions):
                raise ValueError(
                    f"{name} must match completions shape. "
                    f"Template index {i} has K={len(template_completions)} completions "
                    f"but {len(template_scores)} sentiment scores."
                )

            casted_template: List[float] = []
            for j, score in enumerate(template_scores):
                if not isinstance(score, (int, float)):
                    raise ValueError(
                        f"{name} at [{i}][{j}] must be numeric, got {type(score)}"
                    )
                value = float(score)
                if np.isnan(value) or np.isinf(value):
                    raise ValueError(f"{name} at [{i}][{j}] is invalid: {value}")
                if value < -1.0 or value > 1.0:
                    raise ValueError(
                        f"{name} at [{i}][{j}] must be in [-1, 1], got {value}"
                    )
                casted_template.append(value)
            rows.append(casted_template)

        return np.array(rows, dtype=float)
