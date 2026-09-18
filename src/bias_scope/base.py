"""
Abstract base classes for bias detection metrics.
"""

from __future__ import annotations

import inspect
import math
from abc import ABC, abstractmethod
from typing import Any, Callable, ClassVar, Dict, List, Optional, Sequence, Tuple

import numpy as np


class BiasScopeError(RuntimeError):
    """A runtime guard in `run()` failed. Never silence this (PLAN.md Sec. 1)."""


class BiasMetric(ABC):
    """
    Abstract base class for all bias detection metrics.

    All bias metrics must implement the `evaluate` method and inherit
    their category from the appropriate intermediate base class.

    Examples
    --------
    >>> class MyMetric(BiasMetric):
    ...     def evaluate(self, inputs):
    ...         return 0.5
    """

    #: The key in `evaluate()`'s result holding the bias score, for metrics
    #: whose result carries several numbers and whose headline cannot be
    #: inferred from the dict's shape. Empty means "infer it" (RL-061).
    headline_key: ClassVar[str] = ""

    #: The key holding the number of items scored, for metrics that report
    #: several counts and so cannot be read by name alone - HONEST reports
    #: templates, candidates and hurtful candidates, and only the paper says
    #: which one `n` means. Empty means "use the recognised names" (RL-063).
    count_key: ClassVar[str] = ""


    def __repr__(self) -> str:
        """Return a scikit-learn-style representation of the metric config."""
        try:
            signature = inspect.signature(type(self).__init__)
        except (TypeError, ValueError):
            return f"{type(self).__name__}()"

        params = []
        for param in signature.parameters.values():
            if param.name == "self":
                continue
            if param.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue
            if param.name.startswith("_"):
                continue
            if not hasattr(self, param.name):
                continue

            value = getattr(self, param.name)
            params.append(f"{param.name}={value!r}")

        joined = ", ".join(params)
        return f"{type(self).__name__}({joined})"

    @abstractmethod
    def evaluate(self, *args, **kwargs) -> float | Dict[str, float]:
        """
        Evaluate the bias metric.

        Args:
            *args: metric-specific input data
            **kwargs: additional metric parameters

        Returns:
            float | Dict[str, float]: bias score(s)

        Raises:
            ValueError: If inputs are invalid

        Notes:
            - Simple metrics return a single float score
            - Complex metrics return a dictionary with multiple scores
            - Subclasses must implement with their specific signature and validation
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def category(self) -> str:
        """
        Metric category.

        Returns:
            str: One of: 'embedding', 'probability', 'generated_text',
                or 'prompt_based'
        """
        raise NotImplementedError

    #: Metadata (PLAN.md 5.1). Every concrete metric sets this; the base class
    #: leaves it None so `tests/test_metadata.py` can name the ones that do not.
    info: ClassVar[Optional["MetricInfo"]] = None

    # ── The framework entry point (PLAN.md 5.3) ────────────────────────────
    # `evaluate()` above is unchanged and stays the metric-specific API.
    # `run()` wraps it, attaches an interval, a protocol block, and metadata,
    # and enforces the runtime guards. It lives here once rather than in each
    # metric.

    def run(
        self,
        *args,
        seed: int = 42,
        ci: str = "bootstrap",
        protocol_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> "BiasResult":
        """
        Evaluate and return a `BiasResult`.

        Calls ``evaluate(..., return_details=True)``, reads item-level scores
        from ``details["per_item"]`` when the metric provides them, computes a
        confidence interval, and attaches the protocol block and metadata.

        Parameters
        ----------
        *args, **kwargs
            Passed straight through to `evaluate`.
        seed : int
            Applied via `seed_everything` before evaluating, and recorded in
            the protocol.
        ci : str
            "bootstrap" (default), "wald", or "none". Metrics that define
            their own interval override `_interval`.
        protocol_kwargs : dict, optional
            Extra fields for `make_protocol` — `model_id`, `dtype`, `dataset`,
            `decoding`, `resources`, and so on.

        Returns
        -------
        BiasResult

        Raises
        ------
        BiasScopeError
            If a runtime guard fails: score outside `info.value_range`, `n` not
            positive, a non-finite score, or a CI that does not bracket it.
        """
        from bias_scope.result import BiasResult, make_protocol
        from bias_scope.utils import seed_everything

        seed_everything(seed)

        raw = self._call_evaluate(*args, **kwargs)
        score, details = self._split_result(raw, self.headline_key)
        per_item = self._extract_per_item(details)
        n = self._count_items(details, per_item, self.count_key)

        interval, method, p_value = self._interval(score, per_item, n, ci, seed)
        # A metric that computes its own significance test reports it under a
        # fixed key, the same way per-item scores are reported.
        if p_value is None and isinstance(details.get("p_value"), (int, float)):
            p_value = float(details["p_value"])

        info = self._resolve_info()
        protocol = make_protocol(
            metric=type(self).__name__, seed=seed, **(protocol_kwargs or {})
        )

        result = BiasResult(
            metric=type(self).__name__,
            score=score,
            n=n,
            ci=interval,
            ci_method=method,
            per_item=per_item,
            breakdown=self._extract_breakdown(details),
            details=details,
            protocol=protocol,
            info=info,
            p_value=p_value,
        )
        self._check_guards(result)
        return result

    def _call_evaluate(self, *args, **kwargs):
        """Call `evaluate` asking for details where the signature allows it."""
        if "return_details" in inspect.signature(self.evaluate).parameters:
            kwargs.setdefault("return_details", True)
        return self.evaluate(*args, **kwargs)

    @staticmethod
    def _split_result(raw: Any, headline_key: str = "") -> Tuple[float, Dict[str, Any]]:
        """Pull the headline score and the details dict out of what came back."""
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            return float(raw), {}
        if not isinstance(raw, dict):
            raise BiasScopeError(
                f"evaluate() returned {type(raw).__name__}; run() needs a float "
                "or a dict"
            )

        # A metric whose result carries several numbers names its own headline
        # (see `headline_key`). CAT returns lms and ss; only the paper says
        # which is the bias score, so the metric states it rather than letting
        # this function infer it from the dict's shape (RL-061).
        if headline_key:
            value = raw.get(headline_key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return float(value), raw
            raise BiasScopeError(
                f"declared headline_key {headline_key!r} is not a number in "
                f"evaluate()'s result; found keys {sorted(raw)}"
            )

        # A metric may decline to score, and say why: HELM drops instances with
        # no group mention rather than calling them unbiased, and our HELM
        # metrics follow that with `bias_score: None` plus `undefined_reason`.
        # Surfacing the metric's own sentence beats "cannot find a headline
        # score", which reads like a defect in the metric (RL-064).
        if "bias_score" in raw and raw["bias_score"] is None:
            reason = raw.get("undefined_reason") or ""
            raise BiasScopeError(
                f"the metric declined to produce a score: {reason}"
                if reason
                else "the metric declined to produce a score and gave no reason"
            )

        # Metrics use different names for their headline number. Try the
        # documented ones in order rather than guessing from the dict.
        for key in ("bias_score", "score", "value", "effect_size"):
            if key in raw and isinstance(raw[key], (int, float)):
                return float(raw[key]), raw

        # Several metrics name the number after themselves instead
        # ("crows_pairs_score", "aul_score", "aula_score", "ceat_score").
        # Accept exactly one such key: two would be a guess, and a guessed
        # score is the fabrication PLAN.md Section 1 forbids (RL-041).
        suffixed = [
            key
            for key, value in raw.items()
            if key.endswith("_score")
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
        ]
        if len(suffixed) == 1:
            return float(raw[suffixed[0]]), raw

        numeric = [k for k, v in raw.items() if isinstance(v, (int, float))]
        raise BiasScopeError(
            f"cannot find a headline score in evaluate()'s result; expected one "
            f"of 'bias_score', 'score', 'value', 'effect_size', or exactly one "
            f"'<name>_score' key, found numeric keys {numeric}"
        )

    @staticmethod
    def _extract_per_item(details: Dict[str, Any]) -> Optional[List[float]]:
        """Item-level scores live under a fixed key (PLAN.md 5.3)."""
        values = details.get("per_item")
        if values is None:
            return None
        try:
            return [float(v) for v in values]
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _count_items(
        details: Dict[str, Any], per_item: Optional[List[float]], count_key: str = ""
    ) -> int:
        """`n` is the number of items actually scored."""
        if per_item is not None:
            return len(per_item)
        # A metric that reports several counts names the one `n` means.
        if count_key:
            value = details.get(count_key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if value > 0 and float(value).is_integer():
                    return int(value)
            raise BiasScopeError(
                f"declared count_key {count_key!r} is not a positive whole number "
                f"in evaluate()'s result; found keys {sorted(details)}"
            )
        # "n_samples" is CEAT's: Guo & Caliskan draw N samples, each giving one
        # effect size and one variance, then pool them with `df = N - 1`
        # (third_party/code/CEAT/code/ceat.py:205-243). The degrees of freedom
        # say plainly that N is the number of observations the random-effects
        # model pools, so it is the count `n` carries (REVIEW_LATER RL-048).
        # "n_examples" is CAT's and ICAT's: the number of StereoSet test cases
        # actually scored, which is what `n` means (RL-061).
        for key in ("n", "num_items", "num_pairs", "num_rows_evaluated",
                    "num_prompts", "num_generations", "n_samples", "n_examples"):
            value = details.get(key)
            # A count computed through numpy or a division arrives as a whole
            # float (CrowS-Pairs reports `num_pairs: 2.0`). That is a count;
            # 2.5 is not. This guard is for "scored nothing", not for type.
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            if value > 0 and float(value).is_integer():
                return int(value)
        return 0

    @staticmethod
    def _extract_breakdown(details: Dict[str, Any]) -> Dict[str, float]:
        """Per-group scores, when the metric exposes a flat numeric mapping."""
        breakdown = details.get("breakdown") or details.get("per_category") or {}
        if not isinstance(breakdown, dict):
            return {}
        return {
            str(k): float(v)
            for k, v in breakdown.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        }

    def _interval(
        self,
        score: float,
        per_item: Optional[List[float]],
        n: int,
        ci: str,
        seed: int,
    ) -> Tuple[Optional[Tuple[float, float]], str, Optional[float]]:
        """
        Compute a confidence interval. Override for a paper-defined one.

        Metrics without item-level scores (WEAT, SEAT, CEAT, CBS) get
        ``(None, "none", None)`` unless they override this — PLAN.md 5.3.
        """
        from bias_scope.stats import bootstrap_ci, wald_ci

        if ci == "none" or per_item is None or not per_item:
            return None, "none", None
        if ci == "wald":
            return wald_ci(score, n), "wald", None
        if ci == "bootstrap":
            return bootstrap_ci(per_item, seed=seed), "bootstrap", None
        raise ValueError(f"ci must be 'bootstrap', 'wald', or 'none', got {ci!r}")

    def _resolve_info(self) -> "MetricInfo":
        """The metric's metadata, or a placeholder naming what is missing."""
        from bias_scope.metadata import MetricInfo

        if self.info is not None:
            return self.info
        return MetricInfo(
            name=type(self).__name__,
            family=_CATEGORY_TO_FAMILY.get(self.category, "prompt"),
            access=("completions",),
            neutral_value=0.0,
            direction="signed",
            value_range=(float("-inf"), float("inf")),
            fidelity="unaudited",
            reference="not yet recorded",
            deviation_note=(
                f"{type(self).__name__} has no MetricInfo yet; its fidelity has "
                "not been established. See docs/fidelity/INDEX.md."
            ),
        )

    @staticmethod
    def _check_guards(result: "BiasResult") -> None:
        """
        Runtime guards (PLAN.md Section 1). A failure is never silenced.

        Checks the score is finite and inside `value_range`, `n` is positive,
        and the CI brackets the score.
        """
        name = result.metric

        if not math.isfinite(result.score):
            raise BiasScopeError(f"{name}: score is not finite ({result.score})")

        low, high = result.info.value_range
        if not low <= result.score <= high:
            raise BiasScopeError(
                f"{name}: score {result.score} is outside the declared "
                f"value_range {result.info.value_range}"
            )

        if result.n <= 0:
            raise BiasScopeError(
                f"{name}: n must be positive, got {result.n}. The metric scored "
                "no items, or does not report how many it scored."
            )

        if result.ci is not None:
            ci_low, ci_high = result.ci
            # A metric whose per-item values are all equal gives a degenerate
            # interval [v, v], and the score can miss it by an ULP or two purely
            # from summation order. Allow that much slack, scaled to the
            # magnitudes involved, so a perfectly consistent model is not
            # rejected; anything larger is a real bracketing failure.
            slack = _CI_BRACKET_SLACK * max(
                1.0, abs(result.score), abs(ci_low), abs(ci_high)
            )
            if not ci_low - slack <= result.score <= ci_high + slack:
                raise BiasScopeError(
                    f"{name}: the {result.ci_method} interval "
                    f"[{ci_low}, {ci_high}] does not bracket the score "
                    f"{result.score}"
                )


#: Relative tolerance for the "the interval brackets the score" guard. Sized
#: for accumulated float error over a few thousand items, not for a real gap.
_CI_BRACKET_SLACK = 1e-9

#: `category` predates `MetricInfo.family` and uses one different spelling.
_CATEGORY_TO_FAMILY = {
    "embedding": "embedding",
    "probability": "probability",
    "generated_text": "generated_text",
    "prompt_based": "prompt",
}


class EmbeddingMetric(BiasMetric):
    """
    Base class for embedding-based bias metrics.

    Provides common validation methods for embeddings.
    """

    @property
    def category(self) -> str:
        """Category is automatically set to 'embedding'."""
        return "embedding"

    # ── Effect-size metrics have no item-level scores (PLAN.md 5.3) ────────
    # WEAT, SEAT and CEAT report a standardised mean difference, so bootstrapping
    # a mean of per-item values is not the right interval. They use the
    # Hedges-Olkin interval the effect-size literature defines, plus the
    # permutation p-value Caliskan's and May's papers report.

    @staticmethod
    def _group_sizes(details: Dict[str, Any]) -> Optional[Tuple[int, int]]:
        """The two target-group sizes, when the metric reports them."""
        first = details.get("n_target_group_1")
        second = details.get("n_target_group_2")
        if isinstance(first, (int, float)) and isinstance(second, (int, float)):
            if first > 0 and second > 0:
                return int(first), int(second)
        return None

    @staticmethod
    def _count_items(
        details: Dict[str, Any], per_item: Optional[List[float]], count_key: str = ""
    ) -> int:
        """For an effect size, `n` is the number of target stimuli scored."""
        if per_item is not None:
            return len(per_item)
        if not count_key:
            sizes = EmbeddingMetric._group_sizes(details)
            if sizes is not None:
                return sizes[0] + sizes[1]
        return BiasMetric._count_items(details, per_item, count_key)

    def _interval(
        self,
        score: float,
        per_item: Optional[List[float]],
        n: int,
        ci: str,
        seed: int,
    ) -> Tuple[Optional[Tuple[float, float]], str, Optional[float]]:
        """Hedges-Olkin interval on the effect size, when group sizes are known.

        Falls back to the base behaviour when they are not, rather than
        inventing an `n`.
        """
        from bias_scope.stats import hedges_olkin_ci

        if ci == "none":
            return None, "none", None

        sizes = getattr(self, "_last_group_sizes", None)
        if sizes is None:
            return super()._interval(score, per_item, n, ci, seed)
        return hedges_olkin_ci(score, sizes[0], sizes[1]), "hedges_olkin", None

    def run(self, *args, **kwargs):
        """Stash the group sizes so `_interval` can use them, then run."""
        self._last_group_sizes = None
        return super().run(*args, **kwargs)

    def _call_evaluate(self, *args, **kwargs):
        raw = super()._call_evaluate(*args, **kwargs)
        if isinstance(raw, dict):
            self._last_group_sizes = self._group_sizes(raw)
        return raw


    def _validate_embeddings(
        self, embeddings: np.ndarray, name: str
    ) -> None:
        """
        Validate embedding array structure (PRIVATE).

        Args:
            embeddings (np.ndarray): Embedding array to validate.
            name (str): name for error messages

        Raises:
            ValueError: If validation fails
        """
        if len(embeddings) == 0:
            raise ValueError(f"{name} cannot be empty")

        if np.isnan(embeddings).any():
            raise ValueError(f"{name} contains NaN values")

        if np.isinf(embeddings).any():
            raise ValueError(f"{name} contains Inf values")


class ProbabilityMetric(BiasMetric):
    """
    Base class for probability-based bias metrics.

    Provides common validation methods for probabilities, sentence pairs,
    and callable inputs.
    """

    @property
    def category(self) -> str:
        """Category is automatically set to 'probability'."""
        return "probability"

    def _validate_callable(self, fn: Callable, name: str) -> None:
        """
        Validate a callable input (PRIVATE).

        Args:
            fn (Callable): Function to validate.
            name (str): Input name for error messages.

        Raises:
            TypeError: If fn is not callable.
        """
        if not callable(fn):
            raise TypeError(f"{name} must be callable, got {type(fn).__name__}")

    def _init_token_prediction_scorer(
        self, model_name: str | None = None, device: str | None = None
    ) -> None:
        """
        Initialize an optional default token prediction scorer (PRIVATE).
        """
        self._token_prediction_scorer = None
        if model_name is not None:
            from bias_scope.probability_based.scorers import BertPLLScorer

            self._token_prediction_scorer = BertPLLScorer(
                model_name=model_name, device=device
            )

    def _resolve_token_prediction_method(
        self,
        scorer_or_callback: Any,
        method_name: str,
        callback_name: str,
    ) -> Callable:
        """
        Resolve a protocol scorer method or backward-compatible callback (PRIVATE).
        """
        if scorer_or_callback is not None:
            method = getattr(scorer_or_callback, method_name, None)
            if callable(method):
                return method
            self._validate_callable(scorer_or_callback, callback_name)
            return scorer_or_callback

        scorer = getattr(self, "_token_prediction_scorer", None)
        if scorer is None:
            raise TypeError(
                f"{callback_name} must be callable when model_name is not set"
            )

        method = getattr(scorer, method_name, None)
        if not callable(method):
            raise TypeError(
                f"Configured scorer must define callable method '{method_name}'"
            )
        return method

    def _validate_probabilities(
        self, probabilities: np.ndarray, name: str = "probabilities"
    ) -> None:
        """
        Validate probability array (PRIVATE helper).

        Checks that probabilities are valid: in [0,1], no NaN/Inf.

        Args:
            probabilities (np.ndarray): Probability array to validate
            name (str): Name for error messages (default: "probabilities")

        Raises:
            ValueError: If probabilities are invalid
        """
        if len(probabilities) == 0:
            raise ValueError(f"{name} cannot be empty")

        if np.isnan(probabilities).any():
            raise ValueError(f"{name} contains NaN values")

        if np.isinf(probabilities).any():
            raise ValueError(f"{name} contains Inf values")

        if (probabilities < 0).any() or (probabilities > 1).any():
            raise ValueError(
                f"{name} must be in range [0, 1]. "
                f"Got min={np.min(probabilities)}, max={np.max(probabilities)}"
            )

    def _validate_sentence_pair(
        self, sentence1: List[str], sentence2: List[str]
    ) -> None:
        """
        Validate sentence pair has same length (PRIVATE).

        Args:
            sentence1 (List[str]): first tokenized sentence
            sentence2 (List[str]): second tokenized sentence

        Raises:
            ValueError: If validation fails
        """
        if len(sentence1) == 0 or len(sentence2) == 0:
            raise ValueError("Sentences cannot be empty")

        if len(sentence1) != len(sentence2):
            raise ValueError(
                "Sentence pairs must have same length. "
                f"Got {len(sentence1)} and {len(sentence2)} tokens."
            )


class GeneratedTextMetric(BiasMetric):
    """
    Base class for generated text bias metrics.
    """

    @property
    def category(self) -> str:
        """Category is automatically set to 'generated_text'."""
        return "generated_text"

    def _validate_texts(self, texts: Sequence[str], name: str) -> List[str]:
        """
        Validate a sequence of text strings (PRIVATE).

        Args:
            texts (Sequence[str]): Text values to validate.
            name (str): Input name for error messages.

        Returns:
            List[str]: Validated text list.

        Raises:
            TypeError: If texts is not a sequence of strings.
            ValueError: If texts is empty or contains empty strings.
        """
        if not isinstance(texts, Sequence) or isinstance(texts, (str, bytes)):
            raise TypeError(f"{name} must be a sequence of strings")

        texts_list = list(texts)
        if len(texts_list) == 0:
            raise ValueError(f"{name} cannot be empty")

        for i, text in enumerate(texts_list):
            if not isinstance(text, str):
                raise TypeError(
                    f"{name}[{i}] must be a string, got {type(text).__name__}"
                )
            if text == "":
                raise ValueError(f"{name}[{i}] cannot be empty")

        return texts_list

    def _validate_callable(self, fn: Callable, name: str) -> None:
        """
        Validate a callable input (PRIVATE).

        Args:
            fn (Callable): Function to validate.
            name (str): Input name for error messages.

        Raises:
            TypeError: If fn is not callable.
        """
        if not callable(fn):
            raise TypeError(f"{name} must be callable, got {type(fn).__name__}")

    def _validate_finite_float(self, value: float, name: str) -> float:
        """
        Validate a finite float value (PRIVATE).

        Args:
            value (float): Value to validate.
            name (str): Input name for error messages.

        Returns:
            float: Validated float value.

        Raises:
            TypeError: If value is not numeric.
            ValueError: If value is NaN or infinite.
        """
        if not isinstance(value, (int, float, np.floating)):
            raise TypeError(f"{name} must be a float, got {type(value).__name__}")

        value_float = float(value)
        if np.isnan(value_float):
            raise ValueError(f"{name} must be finite. Got NaN")
        if np.isinf(value_float):
            raise ValueError(f"{name} must be finite. Got Inf")
        return value_float

    def _validate_generated_texts(
        self, texts: List[List[str]], name: str = "texts"
    ) -> None:
        """
        Validate a list-of-lists of generated texts (PRIVATE).

        Args:
            texts (List[List[str]]): Nested text list to validate.
                Shape: (n_prompts, n_texts_per_prompt)
            name (str): Argument name used in error messages.

        Raises:
            ValueError: If the outer list is empty.
            ValueError: If any inner list is empty.
        """
        if len(texts) == 0:
            raise ValueError(f"{name} cannot be empty")

        for inner in texts:
            if len(inner) == 0:
                raise ValueError(f"{name} cannot be empty")

    def _validate_completions(
        self, completions: List[List[str]], name: str = "completions"
    ) -> None:
        """
        Validate nested completion lists (PRIVATE).

        Args:
            completions (List[List[str]]): Generated completions grouped by prompt.
            name (str): Argument name used in error messages.

        Raises:
            TypeError: If completions is not a nested list of strings.
            ValueError: If completions is empty or contains empty groups/strings.
        """
        self._validate_generated_texts(completions, name)
        expected_length = len(completions[0])

        for i, group in enumerate(completions):
            if len(group) != expected_length:
                raise ValueError(
                    f"{name} must have the same number of completions per group. "
                    f"Expected {expected_length}, got {len(group)} at index {i}."
                )
            for j, completion in enumerate(group):
                if not isinstance(completion, str):
                    raise TypeError(
                        f"{name}[{i}][{j}] must be a string, got {type(completion).__name__}"
                    )
                if completion == "":
                    raise ValueError(f"{name}[{i}][{j}] cannot be empty")

    def _validate_threshold(self, threshold: float, name: str = "threshold") -> float:
        """
        Validate a threshold in [0, 1] (PRIVATE).

        Args:
            threshold (float): Threshold value to validate.
            name (str): Argument name used in error messages.

        Returns:
            float: Validated threshold as float.

        Raises:
            TypeError: If threshold is not numeric.
            ValueError: If threshold is outside [0, 1].
        """
        if not isinstance(threshold, (int, float, np.floating)):
            raise TypeError(f"{name} must be numeric, got {type(threshold).__name__}")

        threshold_value = float(threshold)
        if not 0.0 <= threshold_value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]. Got {threshold}")

        return threshold_value

    def _validate_classifier_scores(
        self, scores: Sequence[float], name: str = "scores"
    ) -> None:
        """
        Validate classifier scores in [0, 1] (PRIVATE).

        Args:
            scores (Sequence[float]): Scores to validate.
            name (str): Argument name used in error messages.

        Raises:
            ValueError: If scores is empty or contains invalid values.
        """
        if len(scores) == 0:
            raise ValueError(f"{name} cannot be empty")

        for i, score in enumerate(scores):
            if not isinstance(score, (int, float, np.floating)):
                raise ValueError(
                    f"{name}[{i}] must be numeric, got {type(score).__name__}"
                )
            value = float(score)
            if np.isnan(value) or np.isinf(value):
                raise ValueError(f"{name}[{i}] must be finite. Got {score}")
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name}[{i}] must be in [0, 1]. Got {score}")

    def _validate_paired_completions(
        self,
        group_a_completions: List[List[str]],
        group_b_completions: List[List[str]],
    ) -> None:
        """
        Validate paired completion matrices are shape-aligned (PRIVATE).

        Args:
            group_a_completions (List[List[str]]): First completion matrix.
            group_b_completions (List[List[str]]): Second completion matrix.

        Raises:
            ValueError: If the two matrices are not shape-aligned.
        """
        if len(group_a_completions) != len(group_b_completions):
            raise ValueError(
                "group_a_completions and group_b_completions must have the same "
                "number of templates. "
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
        scores: List[List[float]] | None = None,
        name: str = "scores",
        score_range: tuple[float, float] = (-1.0, 1.0),
        item_label: str | None = None,
        item_plural_label: str = "scores",
        sentiment_scores: List[List[float]] | None = None,
    ) -> np.ndarray:
        """
        Validate nested score matrices against completion shape (PRIVATE).

        Supports caller-selected score ranges while preserving existing metric
        call signatures.

        Args:
            completions (List[List[str]]): Completion matrix to align against.
            scores (List[List[float]] | None): Generic score matrix.
            name (str): Score matrix name for error messages.
            score_range (tuple[float, float]): Inclusive valid score range.
            item_label (str | None): Singular item label for error messages.
            item_plural_label (str): Plural item label for shape errors.
            sentiment_scores (List[List[float]] | None): Sentiment score matrix.

        Returns:
            np.ndarray: Validated score matrix as float array.

        Raises:
            ValueError: If score inputs are missing, misaligned, or invalid.
        """
        if scores is not None and sentiment_scores is not None:
            raise ValueError(
                "Provide either scores or sentiment_scores, not both."
            )

        score_matrix = sentiment_scores if sentiment_scores is not None else scores
        if score_matrix is None:
            raise ValueError("scores cannot be None")

        min_value, max_value = score_range
        item_prefix = item_label if item_label is not None else name

        if len(score_matrix) != len(completions):
            raise ValueError(
                f"{name} must have the same number of templates as its completions. "
                f"Got {len(score_matrix)} and {len(completions)}."
            )

        rows: List[List[float]] = []
        for i, (template_completions, template_scores) in enumerate(
            zip(completions, score_matrix)
        ):
            if len(template_scores) != len(template_completions):
                raise ValueError(
                    f"{name} must match completions shape. "
                    f"Template index {i} has K={len(template_completions)} completions "
                    f"but {len(template_scores)} {item_plural_label}."
                )

            casted_template: List[float] = []
            for j, score in enumerate(template_scores):
                if not isinstance(score, (int, float, np.floating)):
                    raise ValueError(
                        f"{item_prefix} at [{i}][{j}] must be numeric, got {type(score)}"
                    )
                value = float(score)
                if np.isnan(value) or np.isinf(value):
                    raise ValueError(f"{item_prefix} at [{i}][{j}] is invalid: {value}")
                if value < min_value or value > max_value:
                    raise ValueError(
                        f"{item_prefix} at [{i}][{j}] must be in "
                        f"[{min_value:g}, {max_value:g}], got {value}"
                    )
                casted_template.append(value)
            rows.append(casted_template)

        return np.array(rows, dtype=float)


class PromptBasedMetric(BiasMetric):

    @property
    def category(self) -> str:
        return "prompt_based"

    def _validate_positive_int(self, value: int, name: str) -> int:
        """
        Validate a positive integer (PRIVATE).

        Args:
            value (int): Value to validate.
            name (str): Argument name used in error messages.

        Returns:
            int: Validated integer.

        Raises:
            ValueError: If value is not a positive integer.
        """
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer. Got {value}")
        return value


# Imported for the annotations above; placed at the end to avoid a cycle with
# result.py, which imports metadata.py, which does not import base.py.
from bias_scope.metadata import MetricInfo  # noqa: E402, F401
from bias_scope.result import BiasResult  # noqa: E402, F401
