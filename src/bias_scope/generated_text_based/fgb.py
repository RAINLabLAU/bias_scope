"""
FGB - Full Gen Bias
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from bias_scope.base import GeneratedTextMetric


class FGB(GeneratedTextMetric):
    """
    Full Gen Bias (FGB).

    FGB captures average bias magnitude over all paired counterfactual generations.

    Let s_a(t, k) and s_b(t, k) be paired scalar scores (e.g., sentiment/toxicity)
    for completion k under template t for groups A and B:

        delta(t, k) = s_a(t, k) - s_b(t, k)
        FGB = mean_{t, k} |delta(t, k)|

    Interpretation:
        - 0 means no average difference between paired generations
        - higher values indicate stronger overall bias magnitude
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
        Evaluate FGB from paired completions and aligned scalar scores.

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
        fgb_score = float(np.mean(np.abs(deltas)))

        if not return_details:
            return fgb_score

        return {
            "fgb_score": fgb_score,
            "signed_mean_difference": float(np.mean(deltas)),
            "num_templates": float(deltas.shape[0]),
            "k": float(deltas.shape[1]),
            "num_pairs": float(deltas.shape[0] * deltas.shape[1]),
            "mean_group_a_score": float(np.mean(a_scores)),
            "mean_group_b_score": float(np.mean(b_scores)),
            "max_pair_gap": float(np.max(np.abs(deltas))),
            "min_pair_gap": float(np.min(np.abs(deltas))),
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
        scores: List[List[float]],
        name: str,
    ) -> np.ndarray:
        if len(scores) != len(completions):
            raise ValueError(
                f"{name} must have the same number of templates as its completions. "
                f"Got {len(scores)} and {len(completions)}."
            )

        rows: List[List[float]] = []
        for i, (template_completions, template_scores) in enumerate(
            zip(completions, scores)
        ):
            if len(template_scores) != len(template_completions):
                raise ValueError(
                    f"{name} must match completions shape. "
                    f"Template index {i} has K={len(template_completions)} completions "
                    f"but {len(template_scores)} scores."
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
