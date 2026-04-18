"""LMB - Language Model Bias."""

import math
from typing import Callable, Dict, List, Literal, Tuple

import numpy as np

from bias_scope.base import ProbabilityMetric


class LMB(ProbabilityMetric):
    """
    Language Model Bias (LMB).

    Compares perplexity of biased vs counterfactual sentences and
    performs statistical test to detect significant differences.
    Lower perplexity indicates the model finds the sentence more "natural".

    Uses paired t-test to determine if model systematically prefers
    one sentence type over another.

    Reference
    ---------
    Barikeri, S., Lauscher, A., Vuli\u0107, I., & Glavaš, G. (2021).
    RedditBias: A Real-World Resource for Bias Evaluation and Debiasing
    of Conversational Language Models. ACL-IJCNLP 2021.

    Examples
    --------
    >>> from bias_scope.probability_based import LMB
    >>>
    >>> lmb = LMB()
    >>>
    >>> # Prediction function
    >>> def predict_fn(sentence, position):
    ...     # Predict each token in unmasked sentence
    ...     return model.predict_token_given_sentence(sentence, position)
    >>>
    >>> # Sentence pairs
    >>> pairs = [
    ...     (["Women", "are", "bad", "at", "math"],
    ...      ["Men", "are", "bad", "at", "math"])
    ... ]
    >>>
    >>> result = lmb.evaluate(pairs, predict_fn)
    >>> print(f"t-statistic: {result['t_stat']:.3f}")
    >>> print(f"p-value: {result['p_value']:.3f}")
    """

    def evaluate(
        self,
        sentence_pairs: List[Tuple[List[str], List[str]]],
        predict_token_given_sentence: Callable[[List[str], int], float],
        *,
        outlier_strategy: Literal["percentile", "none"] = "percentile",
        outlier_percentile: float = 5.0,
        alpha: float = 0.05,
    ) -> Dict[str, float]:
        """
        Evaluate LMB using perplexity comparison.

        Args:
            sentence_pairs (List[Tuple[List[str], List[str]]]): stereotype and anti-stereotype pairs
            predict_token_given_sentence (Callable): token prediction function
            outlier_strategy (Literal["percentile", "none"]): outlier removal strategy
            outlier_percentile (float): percentile threshold for outlier removal (default: 5.0)
            alpha (float): significance level for t-test (default: 0.05)

        Returns:
            Dict[str, float]: Statistical test results and perplexity statistics

        Raises:
            ValueError: If inputs are invalid or insufficient data

        Notes:
            **Input Structure:**
            - sentence_pairs: List of (stereotype, anti-stereotype) pairs
              - Each sentence is a list of tokens
              - Example: [(["Women", "are", "bad"], ["Men", "are", "bad"])]
            - predict_token_given_sentence: Function signature:
              - Takes: sentence (List[str], complete), position (int)
              - Returns: probability (float) of token at position
              - Same as AUL's predict function

            **Perplexity Formula:**
                PP(S) = exp(- (1/N) * Σ log P(token_i | S))

            Where N is the number of scored tokens.

            **Outlier Removal:**
            - "percentile": Remove pairs where either PP < P5 or PP > P95
              (configurable via outlier_percentile)
            - "none": No outlier removal

            **Statistical Test:**
            Paired two-tailed t-test comparing PP(stereotype) vs PP(anti-stereotype)
            - H0: mean(PP_stereo) = mean(PP_anti)
            - H1: mean(PP_stereo) ≠ mean(PP_anti)

            **Return Dictionary:**
            {
                't_stat': float - t-statistic,
                'p_value': float - two-tailed p-value,
                'mean_pp_s1': float - mean perplexity of first sentences,
                'mean_pp_s2': float - mean perplexity of second sentences,
                'mean_diff': float - mean(PP_s1) - mean(PP_s2),
                'effect_size': float - Cohen's d,
                'n': int - number of pairs after outlier removal,
                'outliers_removed': int - number of outliers removed,
                'alpha': float - significance level used
            }

            **Interpretation:**
            - If p_value < alpha: statistically significant difference
            - Positive mean_diff: stereotypes have higher perplexity (less natural)
            - Negative mean_diff: stereotypes have lower perplexity (more natural)

        Examples:
            >>> lmb = LMB()
            >>>
            >>> def mock_predict(sentence, pos):
            ...     # Stereotype sentences get higher prob (lower perplexity)
            ...     if "Women" in sentence:
            ...         return 0.8
            ...     return 0.5
            >>>
            >>> pairs = [
            ...     (["Women", "work"], ["Men", "work"]),
            ...     (["Women", "cook"], ["Men", "cook"])
            ... ]
            >>>
            >>> result = lmb.evaluate(pairs, mock_predict)
            >>> # Stereotypes will have lower perplexity
            >>> assert result['mean_pp_s1'] < result['mean_pp_s2']
        """
        # Validate input
        if len(sentence_pairs) == 0:
            raise ValueError("sentence_pairs cannot be empty")

        if not callable(predict_token_given_sentence):
            raise TypeError(
                f"predict_token_given_sentence must be callable, got {type(predict_token_given_sentence).__name__}"
            )

        if not (0 < outlier_percentile < 50):
            raise ValueError(
                f"outlier_percentile must be in (0, 50), got {outlier_percentile}"
            )

        if not (0 < alpha < 1):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")

        if outlier_strategy not in {"percentile", "none"}:
            raise ValueError(
                f"outlier_strategy must be 'percentile' or 'none', got '{outlier_strategy}'"
            )

        # Compute perplexity for all pairs
        pp_s1_list = []
        pp_s2_list = []

        for s1, s2 in sentence_pairs:
            # Validate pair
            self._validate_sentence_pair(s1, s2)

            # Compute perplexity for each sentence
            pp_s1 = self._compute_perplexity(s1, predict_token_given_sentence)
            pp_s2 = self._compute_perplexity(s2, predict_token_given_sentence)

            pp_s1_list.append(pp_s1)
            pp_s2_list.append(pp_s2)

        pp_s1_arr = np.array(pp_s1_list)
        pp_s2_arr = np.array(pp_s2_list)

        # Outlier removal
        n_original = len(pp_s1_arr)
        outliers_removed = 0

        if outlier_strategy == "percentile":
            # Compute percentiles on ALL perplexities
            all_pps = np.concatenate([pp_s1_arr, pp_s2_arr])
            lower_bound = np.percentile(all_pps, outlier_percentile)
            upper_bound = np.percentile(all_pps, 100 - outlier_percentile)

            # Keep pairs where both PPs are within bounds
            mask = (
                (pp_s1_arr >= lower_bound)
                & (pp_s1_arr <= upper_bound)
                & (pp_s2_arr >= lower_bound)
                & (pp_s2_arr <= upper_bound)
            )

            pp_s1_arr = pp_s1_arr[mask]
            pp_s2_arr = pp_s2_arr[mask]
            outliers_removed = n_original - len(pp_s1_arr)

        # Check sufficient data
        n = len(pp_s1_arr)
        if n == 0:
            raise ValueError(
                "No pairs remaining after outlier removal. "
                "Try a different outlier_strategy or adjust percentile."
            )

        if n == 1:
            raise ValueError(
                "Cannot perform t-test with only 1 pair. "
                f"Started with {n_original} pairs, removed {outliers_removed} outliers. "
                "Need at least 2 pairs."
            )

        # Compute statistics
        mean_pp_s1 = float(np.mean(pp_s1_arr))
        mean_pp_s2 = float(np.mean(pp_s2_arr))
        mean_diff = mean_pp_s1 - mean_pp_s2

        # Paired t-test
        differences = pp_s1_arr - pp_s2_arr
        t_stat, p_value = self._paired_t_test(differences)

        # Effect size (Cohen's d for paired samples)
        std_diff = np.std(differences, ddof=1)
        if std_diff > 1e-10:
            effect_size = float(mean_diff / std_diff)
        else:
            # Zero variance: all differences identical
            # Effect size is undefined, but we can use a large sentinel value
            # or set based on whether mean_diff is non-zero
            if abs(mean_diff) < 1e-10:
                effect_size = 0.0  # No difference
            else:
                # Perfect consistency: use sign of difference with large magnitude
                effect_size = float(np.sign(mean_diff) * 10.0)

        return {
            "t_stat": float(t_stat),
            "p_value": float(p_value),
            "mean_pp_s1": mean_pp_s1,
            "mean_pp_s2": mean_pp_s2,
            "mean_diff": mean_diff,
            "effect_size": effect_size,
            "n": int(n),
            "outliers_removed": int(outliers_removed),
            "alpha": alpha,
        }

    def _compute_perplexity(
        self, sentence: List[str], predict_fn: Callable[[List[str], int], float]
    ) -> float:
        """
        Compute perplexity for a sentence (PRIVATE).

        Perplexity: PP(S) = exp(- (1/N) * Σ log P(s_i | S))

        Args:
            sentence (List[str]): Tokenized sentence
            predict_fn (Callable): Token prediction function

        Returns:
            float: Perplexity value

        Raises:
            ValueError: If probabilities are invalid
        """
        log_probs = []

        for position in range(len(sentence)):
            prob = predict_fn(sentence, position)

            # Validate probability
            token = sentence[position]

            # Check for NaN/Inf first
            if np.isnan(prob):
                raise ValueError(
                    f"Invalid probability (NaN) for token '{token}' "
                    f"at position {position}. Probabilities must be finite."
                )

            if np.isinf(prob):
                raise ValueError(
                    f"Invalid probability (Inf) for token '{token}' "
                    f"at position {position}. Probabilities must be finite."
                )

            if prob <= 0 or prob > 1:
                raise ValueError(
                    f"Invalid probability {prob} for token '{token}' "
                    f"at position {position}. Must be in (0, 1]."
                )

            log_probs.append(np.log(prob))

        # Average negative log-likelihood
        avg_nll = -np.mean(log_probs)

        # Perplexity
        perplexity = float(np.exp(avg_nll))

        return perplexity

    def _paired_t_test(self, differences: np.ndarray) -> Tuple[float, float]:
        """
        Compute paired t-test (PRIVATE).

        Implements Student's t-test for paired samples without scipy dependency.

        Args:
            differences (np.ndarray): Array of paired differences

        Returns:
            Tuple[float, float]: (t_statistic, p_value)

        Notes:
            **Formula:**
                t = mean(diff) / (std(diff) / sqrt(n))

            Where n is the number of pairs.

            **P-value:**
            Two-tailed p-value computed from t-distribution with (n-1) degrees of freedom.
        """
        n = len(differences)
        mean_diff = np.mean(differences)
        std_diff = np.std(differences, ddof=1)

        # Compute t-statistic
        if std_diff < 1e-10:
            # No variance - differences all identical
            if abs(mean_diff) < 1e-10:
                # All zeros - no difference
                return 0.0, 1.0
            else:
                # Infinite t-stat - perfect separation
                return float(np.sign(mean_diff) * np.inf), 0.0

        t_stat = mean_diff / (std_diff / np.sqrt(n))

        # Compute p-value from t-distribution
        # Using numerical approximation for t-distribution CDF
        df = n - 1
        p_value = self._t_distribution_pvalue(abs(t_stat), df)

        return float(t_stat), float(p_value)

    def _t_distribution_pvalue(self, t_abs: float, df: int) -> float:
        """
        Compute two-tailed p-value for t-distribution (PRIVATE).

        Uses numerical approximation without scipy dependency.

        Args:
            t_abs (float): Absolute value of t-statistic
            df (int): Degrees of freedom

        Returns:
            float: Two-tailed p-value
        """
        # For large df (>30), t-distribution approximates normal.
        if df > 30:
            # Use normal approximation
            p_value = 2 * (1 - self._normal_cdf(t_abs))
            return p_value

        if df < 1:
            return 1.0

        x = df / (df + t_abs * t_abs)
        p_value = self._regularized_incomplete_beta(x, df / 2.0, 0.5)
        return max(0.0, min(1.0, p_value))

    def _normal_cdf(self, x: float) -> float:
        """
        Cumulative distribution function for standard normal (PRIVATE).

        Args:
            x (float): Value

        Returns:
            float: CDF value

        Notes:
            Uses error function approximation.
        """
        # Using approximation: Φ(x) ≈ 0.5 * (1 + erf(x/sqrt(2)))
        # erf approximation (Abramowitz and Stegun)
        t = 1.0 / (1.0 + 0.3275911 * abs(x))
        a1, a2, a3, a4, a5 = (
            0.254829592,
            -0.284496736,
            1.421413741,
            -1.453152027,
            1.061405429,
        )
        erf_approx = 1 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(
            -x * x
        )

        if x >= 0:
            return 0.5 * (1 + erf_approx)
        else:
            return 0.5 * (1 - erf_approx)

    def _regularized_incomplete_beta(self, x: float, a: float, b: float) -> float:
        """
        Compute the regularized incomplete beta I_x(a, b) (PRIVATE).

        Args:
            x (float): Value in [0, 1]
            a (float): Parameter
            b (float): Parameter

        Returns:
            float: Regularized incomplete beta value
        """
        if x <= 0.0:
            return 0.0
        if x >= 1.0:
            return 1.0

        log_beta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)

        if x < (a + 1.0) / (a + b + 2.0):
            front = math.exp(
                a * math.log(x) + b * math.log(1.0 - x) - log_beta
            )
            return front * self._beta_continued_fraction(x, a, b) / a

        front = math.exp(
            b * math.log(1.0 - x) + a * math.log(x) - log_beta
        )
        return 1.0 - (front * self._beta_continued_fraction(1.0 - x, b, a) / b)

    def _beta_continued_fraction(self, x: float, a: float, b: float) -> float:
        """
        Evaluate the incomplete beta continued fraction (PRIVATE).

        Args:
            x (float): Value in [0, 1]
            a (float): Parameter
            b (float): Parameter

        Returns:
            float: Continued fraction value
        """
        max_iterations = 200
        epsilon = 3e-14
        fpmin = 1e-300

        qab = a + b
        qap = a + 1.0
        qam = a - 1.0

        c = 1.0
        d = 1.0 - (qab * x / qap)
        if abs(d) < fpmin:
            d = fpmin
        d = 1.0 / d
        fraction = d

        for m in range(1, max_iterations + 1):
            m2 = 2 * m

            aa = m * (b - m) * x / ((qam + m2) * (a + m2))
            d = 1.0 + aa * d
            if abs(d) < fpmin:
                d = fpmin
            c = 1.0 + aa / c
            if abs(c) < fpmin:
                c = fpmin
            d = 1.0 / d
            fraction *= d * c

            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
            d = 1.0 + aa * d
            if abs(d) < fpmin:
                d = fpmin
            c = 1.0 + aa / c
            if abs(c) < fpmin:
                c = fpmin
            d = 1.0 / d
            delta = d * c
            fraction *= delta

            if abs(delta - 1.0) < epsilon:
                break

        return fraction
