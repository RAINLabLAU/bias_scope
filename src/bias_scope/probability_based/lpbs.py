"""LPBS — Log Probability Bias Score (Kurita et al. 2019).

Faithful reimplementation. The statistic that shipped under this name through
v0.1.1 was a different one and now lives in
`bias_scope.probability_based.pairwise_likelihood_preference`; see
`docs/fidelity/lpbs.md` for the audit.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

from bias_scope.base import ProbabilityMetric

TARGET_SLOT = "[TARGET]"
ATTRIBUTE_SLOT = "[ATTRIBUTE]"
MASK = "[MASK]"


class LPBS(ProbabilityMetric):
    """
    Log Probability Bias Score (Kurita, Vyas, Pareek, Black & Tsvetkov 2019).

    Formula (paper §2). For a template such as ``"[TARGET] is a [ATTRIBUTE]."``
    and two targets ``t₁``, ``t₂``:

        p_tgt(t)   = P([MASK] = t | TARGET masked, ATTRIBUTE filled)
        p_prior(t) = P([MASK] = t | BOTH masked, read at the TARGET slot)

        increased log probability score:  log p_tgt(t) − log p_prior(t)
        LPBS = [log p_tgt(t₁) − log p_prior(t₁)] − [log p_tgt(t₂) − log p_prior(t₂)]

    The prior term is the paper's contribution: raw ``p_tgt`` is confounded by
    the model's unconditional preference for one target word over the other, and
    dividing by ``p_prior`` removes it.

    The clearest paper-level case uses singleton target lists such as ``["he"]``
    and ``["she"]``. Targets may also be **word sets** (``["he", "him"]``);
    probabilities are summed within a set before the log, matching the authors'
    reference code (`lib/bias_calculator.py:51-65`).

    LPBS operates on masked-token probabilities. Target and attribute candidates
    must be valid under the supplied scorer/tokenization policy. This class does
    not automatically compose multi-word or multi-WordPiece candidate scores;
    pass only single-token candidates unless the supplied scorer explicitly
    defines another policy outside the canonical LPBS path.

    Interpretation:
        0     no association beyond the model's prior
        > 0   the attribute is associated with ``target_a``
        < 0   the attribute is associated with ``target_b``

    Unbounded and exactly antisymmetric: swapping the targets negates the score.

    Note on the reference code
    --------------------------
    The authors' implementation appears to read the prior at the *attribute*
    mask rather than the target mask, and its `get_index` `last` branch omits
    the ``[CLS]`` offset. This class follows the **paper's** §2 step 3 instead.
    Both readings and the reasoning are recorded in `docs/fidelity/lpbs.md` and
    `REVIEW_LATER.md` RL-012, and the discrepancy has not been confirmed by
    execution.
    """

    def evaluate(
        self,
        templates: Sequence[str],
        target_a: Sequence[str],
        target_b: Sequence[str],
        attributes: Sequence[str],
        fill_probabilities: Callable[..., Mapping[str, float]] | None = None,
        return_details: bool = False,
    ) -> float | Dict[str, Any]:
        """
        Compute the log probability bias score.

        Args:
            templates (Sequence[str]): Templates containing both ``[TARGET]``
                and ``[ATTRIBUTE]``, e.g. ``"[TARGET] is a [ATTRIBUTE]."``
            target_a (Sequence[str]): First target word set, e.g. ``["he"]``.
                The singleton case is the paper's clearest LPBS setting.
            target_b (Sequence[str]): Second target word set, e.g. ``["she"]``.
                Multi-target sets use the reference-code sum-before-log behavior.
            attributes (Sequence[str]): Attribute words to score. Each attribute
                is inserted into the template before querying the target mask.
            fill_probabilities (Callable): Called as
                ``fill_probabilities(sentence, candidates, mask_ordinal=0)`` and
                returning ``{word: probability}`` at the ``mask_ordinal``-th
                ``[MASK]`` in `sentence`, counting from the left. Probabilities
                must be positive.
            return_details (bool): Return the full breakdown instead of a float.

        Returns:
            float | Dict[str, Any]: The BiasScope mean over
            ``(template, attribute)`` pairs, or a dict with ``bias_score``,
            ``per_item``, ``breakdown``, ``n``, and the per-target increased log
            probability scores.

        Raises:
            ValueError: If a template lacks exactly one slot of each kind, an
                input sequence is empty, a target or attribute is not a
                canonical single token, `fill_probabilities` is not callable, or
                a returned probability is not positive.

        Examples:
            >>> table = {("[MASK] is a programmer.", 0): {"he": 0.4, "she": 0.1},
            ...          ("[MASK] is a [MASK].", 0): {"he": 0.2, "she": 0.2}}
            >>> fill = lambda s, c, mask_ordinal=0: {w: table[(s, mask_ordinal)][w]
            ...                                      for w in c}
            >>> score = LPBS().evaluate(["[TARGET] is a [ATTRIBUTE]."], ["he"],
            ...                         ["she"], ["programmer"], fill)
            >>> round(score, 6)
            1.386294
        """
        self._validate_inputs(
            templates, target_a, target_b, attributes, fill_probabilities
        )

        candidates = list(target_a) + list(target_b)
        per_item: List[float] = []
        labels: List[str] = []
        increased_a: List[float] = []
        increased_b: List[float] = []

        for template in templates:
            prior_sentence = template.replace(TARGET_SLOT, MASK).replace(
                ATTRIBUTE_SLOT, MASK
            )
            # Step 3: both slots masked, read at the TARGET slot. That is mask
            # ordinal 0 when [TARGET] precedes [ATTRIBUTE], 1 otherwise.
            target_ordinal = 0 if template.index(TARGET_SLOT) < template.index(
                ATTRIBUTE_SLOT
            ) else 1
            prior = self._probabilities(
                fill_probabilities, prior_sentence, candidates, target_ordinal
            )
            prior_gap = self._log_ratio(prior, target_a, target_b)

            for attribute in attributes:
                # Step 2: TARGET masked, ATTRIBUTE filled. One mask, ordinal 0.
                sentence = template.replace(TARGET_SLOT, MASK).replace(
                    ATTRIBUTE_SLOT, attribute
                )
                observed = self._probabilities(
                    fill_probabilities, sentence, candidates, 0
                )
                observed_gap = self._log_ratio(observed, target_a, target_b)

                per_item.append(observed_gap - prior_gap)
                labels.append(attribute)
                increased_a.append(
                    math.log(self._total(observed, target_a))
                    - math.log(self._total(prior, target_a))
                )
                increased_b.append(
                    math.log(self._total(observed, target_b))
                    - math.log(self._total(prior, target_b))
                )

        bias_score = sum(per_item) / len(per_item)

        if not return_details:
            return bias_score

        breakdown: Dict[str, float] = {}
        for label, value in zip(labels, per_item):
            breakdown.setdefault(label, 0.0)
        for label in breakdown:
            values = [v for lab, v in zip(labels, per_item) if lab == label]
            breakdown[label] = sum(values) / len(values)

        return {
            "bias_score": bias_score,
            "per_item": per_item,
            "n": len(per_item),
            "breakdown": breakdown,
            "attributes": labels,
            "increased_log_prob_a": increased_a,
            "increased_log_prob_b": increased_b,
            "num_templates": len(templates),
            "num_attributes": len(attributes),
        }

    # ── helpers ───────────────────────────────────────────────────────────
    @staticmethod
    def _validate_inputs(templates, target_a, target_b, attributes, fill_probabilities):  # noqa: E501, C901 - RL-093
        """Every input is present, non-empty, and shaped as documented."""
        for name, values in (
            ("templates", templates),
            ("target_a", target_a),
            ("target_b", target_b),
            ("attributes", attributes),
        ):
            if isinstance(values, str):
                raise ValueError(f"{name} must be a sequence of strings, not a string")
        if not templates:
            raise ValueError("templates must contain at least one template")
        if not target_a:
            raise ValueError("target_a must contain at least one word")
        if not target_b:
            raise ValueError("target_b must contain at least one word")
        if not attributes:
            raise ValueError("attributes must contain at least one word")
        if not callable(fill_probabilities):
            raise ValueError(
                "fill_probabilities must be callable as "
                "fill_probabilities(sentence, candidates, mask_ordinal=0)"
            )
        for template in templates:
            if not isinstance(template, str):
                raise ValueError(f"templates entries must be strings, got {template!r}")
            if not template:
                raise ValueError("templates entries must be non-empty strings")
            target_count = template.count(TARGET_SLOT)
            attribute_count = template.count(ATTRIBUTE_SLOT)
            if target_count != 1:
                raise ValueError(
                    f"template {template!r} must contain exactly one {TARGET_SLOT}; "
                    f"found {target_count}"
                )
            if attribute_count != 1:
                raise ValueError(
                    f"template {template!r} must contain exactly one {ATTRIBUTE_SLOT}; "
                    f"found {attribute_count}"
                )

        for name, values in (
            ("target_a", target_a),
            ("target_b", target_b),
            ("attributes", attributes),
        ):
            for value in values:
                if not isinstance(value, str):
                    raise ValueError(f"{name} entries must be strings, got {value!r}")
                if value.strip() != value or not value:
                    raise ValueError(
                        f"{name} entry {value!r} must be a non-empty canonical token"
                    )
                if any(char.isspace() for char in value):
                    raise ValueError(
                        f"{name} entry {value!r} is multi-word. LPBS expects "
                        "masked-token candidates; provide a scorer/tokenization "
                        "policy before using multi-token candidates."
                    )
                if value.startswith("##"):
                    raise ValueError(
                        f"{name} entry {value!r} looks like a continuation wordpiece. "
                        "LPBS does not automatically compose multi-WordPiece "
                        "candidate probabilities."
                    )

    @staticmethod
    def _probabilities(
        fill_probabilities: Callable[..., Mapping[str, float]],
        sentence: str,
        candidates: Sequence[str],
        mask_ordinal: int,
    ) -> Dict[str, float]:
        """Fill probabilities for `candidates`, validated as positive."""
        try:
            raw = fill_probabilities(sentence, candidates, mask_ordinal=mask_ordinal)
        except TypeError:
            # Allow a two-argument callable when the template has one mask only.
            raw = fill_probabilities(sentence, candidates)

        probabilities: Dict[str, float] = {}
        for word in candidates:
            if word not in raw:
                raise ValueError(
                    f"fill_probabilities did not return a probability for {word!r} "
                    f"in {sentence!r}"
                )
            value = float(raw[word])
            if not value > 0.0 or not math.isfinite(value):
                raise ValueError(
                    f"probability for {word!r} in {sentence!r} must be positive and "
                    f"finite, got {value}. LPBS takes logs, so a zero probability "
                    "has no defined score; widen the candidate set or use a model "
                    "that assigns it mass."
                )
            probabilities[word] = value
        return probabilities

    @staticmethod
    def _total(probabilities: Mapping[str, float], words: Sequence[str]) -> float:
        """Sum probabilities within a target set, before taking the log."""
        return sum(probabilities[w] for w in words)

    @classmethod
    def _log_ratio(
        cls,
        probabilities: Mapping[str, float],
        target_a: Sequence[str],
        target_b: Sequence[str],
    ) -> float:
        """log Σ p(target_a) − log Σ p(target_b)."""
        return math.log(cls._total(probabilities, target_a)) - math.log(
            cls._total(probabilities, target_b)
        )

    def _interval(
        self,
        score: float,
        per_item: List[float] | None,
        n: int,
        ci: str,
        seed: int,
    ) -> Tuple[Tuple[float, float] | None, str, float | None]:
        """Bootstrap over the per-(template, attribute) scores."""
        return super()._interval(score, per_item, n, ci, seed)
