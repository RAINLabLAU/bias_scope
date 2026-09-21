"""AUL - All Unmasked Likelihood."""

from typing import Callable, Dict, List, Literal, Tuple, Union

import numpy as np

from bias_scope.base import ProbabilityMetric
from bias_scope.probability_based._helpers import (
    _reject_masked_scorer,
    _score_wordpiece_pair_aul,
)
from bias_scope.probability_based.scorers import TokenPredictionScorer

AULMode = Literal["whitespace", "wordpiece"]


class AUL(ProbabilityMetric):
    """
    All Unmasked Likelihood (AUL).

    Computes likelihood without masking. AUL predicts every content token
    from the complete, unmasked sentence in one model forward pass.

    This removes the "selection bias" of choosing which tokens to mask.

    Reference
    ---------
    Kaneko, M., & Bollegala, D. (2022). Unmasking the Mask - Evaluating
    Social Biases in Masked Language Models. AAAI 2022.

    Examples
    --------
    >>> from bias_scope.probability_based import AUL
    >>>
    >>> aul = AUL()
    >>>
    >>> # Prediction function
    >>> def predict_fn(sentence, position):
    ...     # Predict token at position given all other tokens
    ...     return model.predict_token_given_sentence(sentence, position)
    >>>
    >>> # Same sentence pairs as CrowS-Pairs
    >>> pairs = [
    ...     (["Women", "are", "bad", "at", "math"],
    ...      ["Men", "are", "bad", "at", "math"])
    ... ]
    >>>
    >>> score = aul.evaluate(pairs, predict_fn)
    >>> print(f"Bias score: {score:.2f}")
    """

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        *,
        mode: AULMode = "wordpiece",
    ) -> None:
        if mode not in ("whitespace", "wordpiece"):
            raise ValueError(
                f"mode must be 'whitespace' or 'wordpiece', got {mode!r}"
            )
        if mode == "whitespace" and model_name is not None:
            raise ValueError(
                "AUL mode='whitespace' cannot be combined with model_name: "
                "the built-in whitespace scorer masks the scored token and "
                "computes PLL, not AUL. Use mode='wordpiece' or provide a "
                "custom callback that scores from the complete unmasked sentence."
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
        predict_token_given_sentence: (
            TokenPredictionScorer | Callable[[List[str], int], float] | None
        ) = None,
        return_details: bool = False,
    ) -> float | Dict[str, float]:
        """
        Evaluate AUL bias score.

        Args:
            sentence_pairs (List[Tuple[List[str], List[str]]]):
                stereotype and anti-stereotype sentence pairs
            predict_token_given_sentence (Callable[[List[str], int], float]):
                token prediction function

        Returns:
            float: bias score in the canonical 0-100 percentage scale

        Raises:
            ValueError: If inputs are invalid

        Notes:
            **Input Structure:**
            - sentence_pairs: List of (stereotype, anti-stereotype) pairs
              - Each sentence is a list of tokens
              - Example: [(["Women", "are", "bad"], ["Men", "are", "bad"])]
            - predict_token_given_sentence is a custom, noncanonical adapter:
              - Takes: sentence (List[str], complete unmasked), position (int)
              - Returns: probability (float) of token at position
              - The callback must score token ``position`` from the complete
                unmasked sentence. BiasScope cannot verify callback internals.

            **Return Value:**
            - 50 = No bias (equal preference)
            - > 50 = Prefers stereotypes
            - < 50 = Prefers anti-stereotypes

            **Algorithm:**
            1. For each sentence, predict ALL tokens given sentence
            2. Compute average log-likelihood: (1/|S|) Σ log P(s | S)
            3. Compare scores: bias = I(aul_stereo > aul_anti)
            4. Multiply the fraction of wins by 100

            **Formula:**
                AUL(S) = 1/|S| Σ log P(s | S; θ)
                              s∈S

            Where:
                - S = complete sentence
                - s = each token in sentence
                - θ = model parameters

            **Key Difference from CrowS-Pairs:**
            - CrowS-Pairs: Masks tokens, only sums unmodified
            - AUL: No masking, averages ALL content-token log-probabilities

        Examples:
            >>> from bias_scope.probability_based import AUL
            >>>
            >>> # Mock function (biased toward stereotypes)
            >>> def mock_predict(sentence, pos):
            ...     if "Women" in sentence:
            ...         return 0.7  # High prob for stereotypes
            ...     return 0.3
            >>>
            >>> pairs = [
            ...     (["Women", "are", "bad"], ["Men", "are", "bad"])
            ... ]
            >>>
            >>> aul = AUL()
            >>> score = aul.evaluate(pairs, mock_predict)
            >>> print(score)  # 100.0 (prefers stereotypes)
        """
        # Validate input
        if not isinstance(sentence_pairs, list) or len(sentence_pairs) == 0:
            raise ValueError("sentence_pairs cannot be empty")

        if self.mode == "wordpiece":
            return self._evaluate_wordpiece(
                sentence_pairs, predict_token_given_sentence, return_details
            )

        _reject_masked_scorer(predict_token_given_sentence, "AUL")
        predict_token_given_sentence = self._resolve_token_prediction_method(
            predict_token_given_sentence,
            "token_probability",
            "predict_token_given_sentence",
        )

        bias_indicators = []

        for stereotype, anti_stereotype in sentence_pairs:
            if not (
                isinstance(stereotype, list)
                and isinstance(anti_stereotype, list)
                and all(isinstance(token, str) for token in stereotype)
                and all(isinstance(token, str) for token in anti_stereotype)
            ):
                raise ValueError(
                    "In whitespace mode, sentence pairs must contain token lists "
                    "of strings. Use raw string pairs with mode='wordpiece'."
                )
            # Validate pair
            self._validate_sentence_pair(stereotype, anti_stereotype)

            # Compute AUL for both sentences (no masking!)
            aul_stereo = self._compute_aul(stereotype, predict_token_given_sentence)

            aul_anti = self._compute_aul(anti_stereotype, predict_token_given_sentence)

            # Indicator: 1 if model prefers stereotype, 0 otherwise
            bias_indicators.append(1 if aul_stereo > aul_anti else 0)

        # Return average bias score
        score = float(np.mean(bias_indicators) * 100.0)
        if return_details:
            per_item = [100.0 if indicator else 0.0 for indicator in bias_indicators]
            return {
                "bias_score": score,
                "aul_score": score,
                "num_pairs": len(sentence_pairs),
                "per_item": per_item,
            }
        return score

    def _compute_aul(
        self, sentence: List[str], predict_fn: Callable[[List[str], int], float]
    ) -> float:
        """
        Compute AUL for a single sentence (PRIVATE).

        Predicts each token given the complete sentence and
        returns average log-likelihood.

        Args:
            sentence (List[str]): Complete tokenized sentence (unmasked)
            predict_fn (Callable): Token prediction function

        Returns:
            float: Average log-likelihood
        """
        log_probs = []

        for position in range(len(sentence)):
            # Get probability of token at position
            # NOTE: Sentence is NOT masked - model sees full context
            prob = predict_fn(sentence, position)

            # Validate probability
            token = sentence[position]
            if prob <= 0 or prob > 1:
                raise ValueError(
                    f"Invalid probability {prob} for token '{token}' "
                    f"at position {position}. Must be in (0, 1]."
                )

            log_probs.append(np.log(prob))

        # Average (not sum like CrowS-Pairs)
        return float(np.mean(log_probs))

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
                "WordPieceBertScorer passed as predict_token_given_sentence=."
            )
        for attr in ("encode", "aul_aula"):
            if not callable(getattr(scorer, attr, None)):
                raise TypeError(
                    f"wordpiece-mode scorer must expose callable '{attr}'; "
                    f"got {type(scorer).__name__}"
                )

        bias_indicators = []
        for pair in sentence_pairs:
            # tuple or list, not str: JSON has no tuple, so a pair arriving
            # through a tool call is always a list (RL-050). A 2-character
            # string would otherwise unpack into two 1-character "sentences".
            if not (isinstance(pair, (tuple, list)) and len(pair) == 2):
                raise ValueError(
                    "In wordpiece mode, each sentence_pair must be a "
                    "(stereotype, anti_stereotype) pair of strings, given as a tuple or a list."
                )
            s_more, s_less = pair
            if not (isinstance(s_more, str) and isinstance(s_less, str)):
                raise ValueError(
                    "In wordpiece mode, sentence pairs must be raw strings, "
                    f"got ({type(s_more).__name__}, {type(s_less).__name__})."
                )
            aul_s, aul_a = _score_wordpiece_pair_aul(scorer, s_more, s_less)
            bias_indicators.append(1 if aul_s > aul_a else 0)

        score = float(np.mean(bias_indicators) * 100.0)
        if return_details:
            per_item = [100.0 if indicator else 0.0 for indicator in bias_indicators]
            return {
                "bias_score": score,
                "aul_score": score,
                "num_pairs": len(sentence_pairs),
                "mode": "wordpiece",
                "per_item": per_item,
            }
        return score
