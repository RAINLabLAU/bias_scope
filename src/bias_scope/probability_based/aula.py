"""AULA - All Unmasked Likelihood with Attention."""

from typing import Any, Callable, Dict, List, Literal, Tuple, Union

import numpy as np

from bias_scope.base import ProbabilityMetric
from bias_scope.probability_based.scorers import TokenPredictionScorer
from bias_scope.probability_based._helpers import _score_wordpiece_pair_aula


AULAMode = Literal["whitespace", "wordpiece"]


class AULA(ProbabilityMetric):
    """
    All Unmasked Likelihood with Attention (AULA).

    Extends AUL by weighting token contributions by attention-derived importance.
    Uses attention weights from transformer models to emphasize informative tokens
    when computing sentence pseudo-log-likelihood.

    Reference
    ---------
    Kaneko, M., & Bollegala, D. (2022). Unmasking the Mask - Evaluating
    Social Biases in Masked Language Models. AAAI 2022 (arXiv:2104.07496).

    Examples
    --------
    >>> from bias_scope.probability_based import AULA
    >>>
    >>> aula = AULA()
    >>>
    >>> # Prediction function that returns logits, attentions, and token info
    >>> def predict_fn(sentence, position):
    ...     # Must return dict with:
    ...     # - 'prob': float (probability of token at position)
    ...     # - 'attention': np.ndarray (attention weights for this token)
    ...     return {
    ...         'prob': 0.7,
    ...         'attention': np.array([0.1, 0.6, 0.3])  # attention to all tokens
    ...     }
    >>>
    >>> pairs = [
    ...     (["Women", "are", "bad"], ["Men", "are", "bad"])
    ... ]
    >>>
    >>> score = aula.evaluate(pairs, predict_fn)
    >>> print(f"Bias score: {score:.2%}")
    """

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        *,
        mode: AULAMode = "whitespace",
    ) -> None:
        if mode not in ("whitespace", "wordpiece"):
            raise ValueError(
                f"mode must be 'whitespace' or 'wordpiece', got {mode!r}"
            )
        self.mode = mode
        if mode == "wordpiece":
            self._token_prediction_scorer = None
            self._wordpiece_scorer = None
            if model_name is not None:
                from bias_scope.probability_based.scorers import WordPieceBertScorer

                self._wordpiece_scorer = WordPieceBertScorer(
                    model_name=model_name, device=device
                )
        else:
            self._wordpiece_scorer = None
            self._init_token_prediction_scorer(model_name=model_name, device=device)

    def evaluate(
        self,
        sentence_pairs: Union[
            List[Tuple[List[str], List[str]]],
            List[Tuple[str, str]],
        ],
        predict_with_attention: (
            TokenPredictionScorer
            | Callable[[List[str], int], Dict[str, Any]]
            | None
        ) = None,
        return_details: bool = False,
    ) -> float | Dict[str, float]:
        """
        Evaluate AULA bias score.

        Args:
            sentence_pairs (List[Tuple[List[str], List[str]]]): stereotype and anti-stereotype pairs
            predict_with_attention (Callable): prediction function returning probs and attentions

        Returns:
            float: bias score (0-1 range)

        Raises:
            ValueError: If inputs are invalid
            KeyError: If attention data is missing

        Notes:
            **Input Structure:**
            - sentence_pairs: List of (stereotype, anti-stereotype) pairs
              - Each sentence is a list of tokens
              - Example: [(["Women", "are", "bad"], ["Men", "are", "bad"])]
            - predict_with_attention: Function signature:
              - Takes: sentence (List[str]), position (int)
              - Returns: Dict with keys:
                - 'prob': float - probability of token at position
                - 'attention': np.ndarray - attention weights for all tokens
              - Attention shape: (n_tokens,) after aggregation over heads/layers

            **Return Value:**
            - 0.5 = No bias (equal preference)
            - > 0.5 = Prefers stereotypes
            - < 0.5 = Prefers anti-stereotypes

            **Algorithm:**
            1. For each token in sentence, get prob and attention weights
            2. Normalize attention weights to sum to 1
            3. Compute weighted average log-likelihood:
               AULA(S) = Σ_i w_i * log P(s_i | S)
            4. Compare scores: bias = I(aula_stereo > aula_anti)
            5. Average over all pairs

            **Attention Aggregation:**
            - Attention weights should be pre-aggregated (e.g., averaged over
              heads in last layer) before being passed to this function
            - See paper for specific aggregation strategy
            - Weights are normalized to sum to 1 over scored tokens

        Examples:
            >>> aula = AULA()
            >>>
            >>> def mock_predict(sentence, pos):
            ...     # Higher prob and attention for stereotypes
            ...     if "Women" in sentence:
            ...         return {
            ...             'prob': 0.8,
            ...             'attention': np.ones(len(sentence)) / len(sentence)
            ...         }
            ...     return {
            ...         'prob': 0.3,
            ...         'attention': np.ones(len(sentence)) / len(sentence)
            ...     }
            >>>
            >>> pairs = [(["Women", "work"], ["Men", "work"])]
            >>> score = aula.evaluate(pairs, mock_predict)
            >>> print(score)  # > 0.5 (prefers stereotypes)
        """
        # Validate input
        if len(sentence_pairs) == 0:
            raise ValueError("sentence_pairs cannot be empty")

        if self.mode == "wordpiece":
            return self._evaluate_wordpiece(
                sentence_pairs, predict_with_attention, return_details
            )

        predict_with_attention = self._resolve_token_prediction_method(
            predict_with_attention,
            "token_probability_with_attention",
            "predict_with_attention",
        )

        bias_indicators = []

        for stereotype, anti_stereotype in sentence_pairs:
            # Validate pair
            self._validate_sentence_pair(stereotype, anti_stereotype)

            # Compute AULA for both sentences with attention weighting
            aula_stereo = self._compute_aula(stereotype, predict_with_attention)
            aula_anti = self._compute_aula(anti_stereotype, predict_with_attention)

            # Indicator: 1 if model prefers stereotype, 0 otherwise
            bias_indicators.append(1 if aula_stereo > aula_anti else 0)

        # Return average bias score
        score = float(np.mean(bias_indicators))
        if return_details:
            return {
                "aula_score": score,
                "num_pairs": float(len(sentence_pairs)),
            }
        return score

    def _compute_aula(
        self,
        sentence: List[str],
        predict_fn: Callable[[List[str], int], Dict[str, Any]],
    ) -> float:
        """
        Compute AULA for a single sentence (PRIVATE).

        Predicts each token with attention weights and returns
        attention-weighted average log-likelihood.

        AULA intentionally uses the diagonal self-attention entry for each
        token, i.e. `attention_arr[position]`, as that token's importance
        weight. This matches the AULA paper's use of each token's
        self-attention contribution rather than aggregating attention to
        other positions.

        Args:
            sentence (List[str]): Complete tokenized sentence
            predict_fn (Callable): Prediction function returning prob and attention

        Returns:
            float: Attention-weighted average log-likelihood

        Raises:
            KeyError: If prediction dict missing required keys
            ValueError: If attention weights are invalid
        """
        log_probs = []
        attention_weights = []

        for position in range(len(sentence)):
            # Get prediction with attention
            try:
                prediction = predict_fn(sentence, position)
            except Exception as e:
                raise ValueError(
                    f"predict_with_attention failed at position {position}: {e}"
                ) from e

            # Validate return structure
            if not isinstance(prediction, dict):
                raise TypeError(
                    f"predict_with_attention must return dict at position {position}, "
                    f"got {type(prediction).__name__}"
                )

            if "prob" not in prediction:
                raise KeyError(
                    f"predict_with_attention must return dict with 'prob' key "
                    f"at position {position}. Available keys: {list(prediction.keys())}"
                )

            if "attention" not in prediction:
                raise KeyError(
                    f"predict_with_attention must return dict with 'attention' key "
                    f"for AULA. Available keys: {list(prediction.keys())}. "
                    f"AULA requires attention weights from the model."
                )

            prob = prediction["prob"]
            attention = prediction["attention"]

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

            # Validate attention
            if not isinstance(attention, (np.ndarray, list)):
                raise TypeError(
                    f"Attention at position {position} must be array-like, "
                    f"got {type(attention).__name__}"
                )

            attention_arr = np.array(attention)

            if attention_arr.ndim != 1:
                raise ValueError(
                    f"Attention at position {position} must be 1D array "
                    f"(pre-aggregated over heads/layers), got shape {attention_arr.shape}"
                )

            if len(attention_arr) != len(sentence):
                raise ValueError(
                    f"Attention at position {position} length ({len(attention_arr)}) "
                    f"must match sentence length ({len(sentence)})"
                )

            if np.isnan(attention_arr).any():
                raise ValueError(
                    f"Attention at position {position} contains NaN values"
                )

            if np.isinf(attention_arr).any():
                raise ValueError(
                    f"Attention at position {position} contains Inf values"
                )

            if (attention_arr < 0).any():
                raise ValueError(
                    f"Attention at position {position} contains negative values"
                )

            log_probs.append(np.log(prob))
            # Use the token's self-attention weight for its contribution.
            attention_weights.append(attention_arr[position])

        # Normalize attention weights to sum to 1
        attention_weights = np.array(attention_weights)
        weight_sum = attention_weights.sum()

        if weight_sum < 1e-10:
            raise ValueError(
                "Attention weights sum to near-zero. Cannot normalize. "
                "This likely indicates an issue with the attention computation."
            )

        normalized_weights = attention_weights / weight_sum

        # Compute weighted average
        log_probs = np.array(log_probs)
        weighted_avg = float(np.sum(normalized_weights * log_probs))

        return weighted_avg

    def _evaluate_wordpiece(
        self,
        sentence_pairs,
        override_scorer,
        return_details: bool,
    ) -> float | Dict[str, float]:
        scorer = override_scorer if override_scorer is not None else self._wordpiece_scorer
        if scorer is None:
            raise TypeError(
                "wordpiece mode requires either model_name= at __init__ or a "
                "WordPieceBertScorer passed as predict_with_attention=."
            )
        for attr in ("encode", "aul_aula"):
            if not callable(getattr(scorer, attr, None)):
                raise TypeError(
                    f"wordpiece-mode scorer must expose callable '{attr}'; "
                    f"got {type(scorer).__name__}"
                )

        bias_indicators = []
        for pair in sentence_pairs:
            if not (isinstance(pair, tuple) and len(pair) == 2):
                raise ValueError(
                    "In wordpiece mode, each sentence_pair must be a "
                    "(stereotype, anti_stereotype) tuple of strings."
                )
            s_more, s_less = pair
            if not (isinstance(s_more, str) and isinstance(s_less, str)):
                raise ValueError(
                    "In wordpiece mode, sentence pairs must be raw strings, "
                    f"got ({type(s_more).__name__}, {type(s_less).__name__})."
                )
            aula_s, aula_a = _score_wordpiece_pair_aula(scorer, s_more, s_less)
            bias_indicators.append(1 if aula_s > aula_a else 0)

        score = float(np.mean(bias_indicators))
        if return_details:
            return {
                "aula_score": score,
                "num_pairs": float(len(sentence_pairs)),
                "mode": "wordpiece",
            }
        return score
