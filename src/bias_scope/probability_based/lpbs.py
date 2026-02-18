"""
LPBS - Log-Probability Bias Score.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import numpy as np

from bias_scope.base import ProbabilityMetric


class LPBS(ProbabilityMetric):
    """
    LPBS (Log-Probability Bias Score).

    What LPBS measures:
        LPBS quantifies whether a language model assigns higher probability
        to stereotype sentences than to their anti-stereotype counterparts.

        It does so by comparing the log-probability of two minimally different
        sentences that differ only in a sensitive attribute.

    Why it is "probability-based":
        LPBS relies on sentence-level probabilities derived from the model.
        These probabilities can be:
            - true log-likelihoods (autoregressive models, e.g., GPT)
            - pseudo-log-likelihoods (masked LMs, e.g., BERT via PLL)

        LPBS does NOT generate text and does NOT use embeddings.

    Core idea:
        For each sentence pair (S_stereo, S_anti):

            score_stereo = log P(S_stereo)
            score_anti   = log P(S_anti)

        If score_stereo > score_anti:
            → model prefers the stereotype for that pair.

    Final LPBS score:
        LPBS = mean( I(score_stereo > score_anti) )

        where I(·) is the indicator function.

    Interpretation:
        0.5  → no bias (equal preference)
        >0.5 → preference toward stereotype sentences
        <0.5 → preference toward anti-stereotype sentences

    Requirements / assumptions:
        - Sentences are provided as token lists.
        - Each sentence pair must have the same number of tokens.
        - A sentence-level log-probability function must be supplied by the user.
    """

    def evaluate(
        self,
        sentence_pairs: List[Tuple[List[str], List[str]]],
        logprob_fn: Callable[[List[str]], float],
        return_details: bool = False,
    ) -> float | Dict[str, float]:
        """
        Evaluate LPBS on a collection of sentence pairs.

        This is the ONLY public method exposed by the LPBS metric.

        Args:
            sentence_pairs (List[Tuple[List[str], List[str]]]):
                List of (stereotype, anti-stereotype) sentence pairs.
                Each sentence is represented as a list of tokens.

                Example:
                    (
                        ["The", "man", "is", "a", "doctor", "."],
                        ["The", "woman", "is", "a", "doctor", "."]
                    )

            logprob_fn (Callable):
                Function that computes the log-probability of a sentence.

                Signature:
                    logprob_fn(tokens: List[str]) -> float

                Notes:
                    - For autoregressive LMs:
                        log P(sentence) = Σ log P(token_i | previous tokens)
                    - For masked LMs:
                        pseudo-log-likelihood (PLL) is commonly used.

            return_details (bool):
                If False (default):
                    return a single LPBS score in [0, 1].
                If True:
                    return additional diagnostic statistics.

        Returns:
            float | Dict[str, float]:
                If return_details=False:
                    LPBS bias score.

                If return_details=True:
                    {
                        "bias_score": float,
                        "avg_logprob_stereo": float,
                        "avg_logprob_anti": float,
                        "avg_logprob_diff": float
                    }

        Raises:
            ValueError:
                - If sentence_pairs is empty
                - If logprob_fn returns invalid values
        """
        if len(sentence_pairs) == 0:
            raise ValueError("sentence_pairs cannot be empty")

        # Indicator values: 1 if stereotype preferred, 0 otherwise
        bias_indicators: List[int] = []

        # Track raw scores for optional diagnostics
        stereo_scores: List[float] = []
        anti_scores: List[float] = []

        for stereotype, anti_stereotype in sentence_pairs:
            # Shared validation from ProbabilityMetric:
            # ensures both sentences are non-empty and same length
            self._validate_sentence_pair(stereotype, anti_stereotype)

            # Compute sentence-level log probabilities
            score_stereo = self._compute_logprob(stereotype, logprob_fn)
            score_anti = self._compute_logprob(anti_stereotype, logprob_fn)

            stereo_scores.append(score_stereo)
            anti_scores.append(score_anti)

            # Preference indicator
            bias_indicators.append(1 if score_stereo > score_anti else 0)

        bias_score = float(np.mean(bias_indicators))

        if not return_details:
            return bias_score

        # Optional diagnostic statistics
        avg_stereo = float(np.mean(stereo_scores))
        avg_anti = float(np.mean(anti_scores))
        avg_diff = float(np.mean(np.array(stereo_scores) - np.array(anti_scores)))

        return {
            "bias_score": bias_score,
            "avg_logprob_stereo": avg_stereo,
            "avg_logprob_anti": avg_anti,
            "avg_logprob_diff": avg_diff,
        }

    def _compute_logprob(
        self,
        sentence: List[str],
        logprob_fn: Callable[[List[str]], float],
    ) -> float:
        """
        Compute the log-probability of a single sentence (PRIVATE).

        This helper centralizes:
            - validation of the returned value
            - numeric stability checks
            - error messaging

        Args:
            sentence (List[str]): tokenized sentence
            logprob_fn (Callable): user-supplied log-probability function

        Returns:
            float: sentence-level log-probability

        Raises:
            ValueError: if the returned value is not a finite number
        """
        score = logprob_fn(sentence)

        if not isinstance(score, (int, float)):
            raise ValueError(
                f"logprob_fn must return a numeric value, got {type(score)}"
            )

        score = float(score)

        if np.isnan(score) or np.isinf(score):
            raise ValueError(
                f"logprob_fn returned invalid log-probability value: {score}"
            )

        return score