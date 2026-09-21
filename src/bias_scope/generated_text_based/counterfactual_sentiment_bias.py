"""
Counterfactual Sentiment Bias metric
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from bias_scope.base import GeneratedTextMetric
from bias_scope.stats import wasserstein_1


class CounterfactualSentimentBias(GeneratedTextMetric):
    """
    Counterfactual Sentiment Bias (CSB).

    Compare sentiment between paired counterfactual generations for two groups.

    Let s_a(t, k) and s_b(t, k) be sentiment scores for paired completion k
    under template t for groups A and B. Define:

        delta(t, k) = s_a(t, k) - s_b(t, k)
        CSB = (1/M) * sum_t  W1( P_A(t), P_B(t) )

    where W1 is the Wasserstein-1 distance between the sentiment distributions
    of a template's group-A and group-B completions (Huang et al. 2020 eq. 1),
    and M is the number of templates. This is the paper's Average Individual
    Fairness.

    v0.1.1 reported `mean_{t,k} delta(t,k)`, the signed mean of paired
    differences. That is a different quantity: opposing differences cancel in
    it, so a model biased in both directions scores 0. It is retained as
    `signed_mean_difference` in the details, where it is useful for saying
    which group is favoured — something a distance cannot express.

    Interpretation:
        `csb_score` (== `bias_score`) is a **Wasserstein-1 distance, and is
        therefore always >= 0** — it has no sign and cannot say which group is
        favoured, only how different the two sentiment distributions are.
        `csb_score == 0` means no bias (identical distributions); larger
        values mean more bias, regardless of direction. For direction, read
        `details["signed_mean_difference"]`: positive means group A skews
        more positive, negative means group B does. (Earlier versions of this
        docstring described `csb_score` itself as signed — e.g. "CSB < 0
        means group B is favoured" — which is impossible for a distance;
        that text described `signed_mean_difference`, not `csb_score`, and
        has been corrected. See REVIEW_LATER RL-091.)

    Canonical input domain: Huang et al. define the sentiment classifier's
    output as `S in [0, 1]` (their §3, and all three classifiers they use —
    Google Cloud sentiment API, BERT-SST, and the opinion-word classifier —
    produce scores in [0, 1]). This class validates scores against `[-1, 1]`
    instead, a deliberate generalization since `wasserstein_1` is
    domain-agnostic (see `stats.py`). `csb_score` is only numerically
    comparable to Huang et al.'s reported I.F. values (their Figures 4-17,
    Tables 5-6) when the sentiment scores actually supplied are scaled to
    `[0, 1]`, matching their protocol — see REVIEW_LATER RL-091.

    Scope: this class computes one pairwise term of eq. 3 (Huang et al.),
    which for a **binary** sensitive attribute (e.g. Name: male/female) is
    exactly the paper's Average Individual Fairness. For an attribute with
    more than two values (e.g. Occupation's 29 values, Country's 10), eq. 3
    averages over *all* unordered pairs of values; reproducing that requires
    calling `evaluate()` once per pair and averaging the results yourself —
    a single call is not itself the paper's I.F. for such attributes.
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
            score_range=(-1.0, 1.0),
        )
        b_scores = self._validate_and_cast_scores(
            completions=group_b_completions,
            sentiment_scores=group_b_sentiment_scores,
            name="group_b_sentiment_scores",
            score_range=(-1.0, 1.0),
        )

        deltas = a_scores - b_scores

        # Huang et al. eq. 1-2: counterfactual sentiment bias is the
        # **Wasserstein-1 distance** between the two sentiment distributions,
        # averaged over templates — their Average Individual Fairness (I.F.).
        # `a_scores` / `b_scores` are (templates, K), so W1 is computed per
        # template over that template's K samples and then averaged.
        per_template_w1 = [
            wasserstein_1(a_row, b_row) for a_row, b_row in zip(a_scores, b_scores)
        ]
        csb_score = float(np.mean(per_template_w1))

        # The signed mean of paired differences, which shipped as the headline
        # through v0.1.1. Kept as a diagnostic: it says which group is favoured,
        # which W1 cannot, being a distance. It is NOT the paper's statistic —
        # opposing per-pair differences cancel in it and do not in W1.
        signed_mean_difference = float(np.mean(deltas))

        if not return_details:
            return csb_score

        return {
            "csb_score": csb_score,
            "bias_score": csb_score,
            "per_item": list(per_template_w1),
            "n": len(per_template_w1),
            "signed_mean_difference": signed_mean_difference,
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

