"""
CBS - Categorical Bias Score.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from bias_scope.base import ProbabilityMetric


@dataclass
class CBSResult:
    """
    Container for CBS output.

    Fields:
        cbs:
            Final CBS score:
                CBS = (1/|T|)(1/|A|) sum_{t in T} sum_{a in A} Var_{n in N}( log P'(n | t,a) )

        per_template_attribute:
            Breakdown keyed by "template=... | attr=...".
            Each value stores:
                - variance: float
                - top_target: str
    """

    cbs: float
    per_template_attribute: Dict[str, Dict[str, float | str]]


class CBS(ProbabilityMetric):
    """
    CBS (Categorical Bias Score) for Masked Language Models (MLMs).

    What CBS measures:
        CBS quantifies how unevenly an MLM assigns probability to different
        target group words (multi-class) under bias-sensitive templates.

    Normalization:
        P'(n | template, attr) = P(n | template with attr) / P(n | prior template)

        The prior template replaces the attribute placeholder with as many
        mask tokens as the attribute has subwords (Ahn & Oh 2021 sec 3.2's
        whole-word-masking adaptation), so the prior sentence differs from
        the target sentence only in whether the attribute is filled in or
        masked, not also in length.

    Final score:
        CBS = average over all templates and attributes of:
              Var_{n in N}( log P'(n) )

    Requirements / assumptions:
        - Designed for masked LMs (BERT, RoBERTa).
        - Each template must contain exactly one mask token for the target slot.
        - Templates must include an attribute placeholder (default: "{attr}").
        - Target words are single-token by default; pass
          allow_multi_token_targets=True to whole-word-mask multi-subword
          target words the same way (one mask token per subword, aggregated
          by summing the per-subword log-probabilities).
    """

    def __init__(
        self, model_name: str = "bert-base-uncased", device: Optional[str] = None
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.model.to(self.device)
        self.model.eval()

        if self.tokenizer.mask_token is None or self.tokenizer.mask_token_id is None:
            raise ValueError(
                "This model/tokenizer has no mask token. Use a masked language model."
            )

        self.mask_token = self.tokenizer.mask_token
        self.mask_token_id = self.tokenizer.mask_token_id

    def evaluate(
        self,
        templates: List[str],
        target_words: List[str],
        attribute_words: List[str],
        placeholder: str = "{attr}",
        return_details: bool = False,
        allow_multi_token_targets: bool = False,
    ) -> float | Dict[str, float]:
        """
        Evaluate CBS.

        Args:
            templates:
                List of templates containing:
                    - exactly one mask token for the target slot
                    - an attribute placeholder like "{attr}"

            target_words:
                List of target category words.

            attribute_words:
                List of attribute words to plug into templates.

            placeholder:
                The placeholder used for attributes.

            return_details:
                If False: return a single float CBS score.
                If True: return a dict with score + breakdown.

        Returns:
            float | Dict[str, float]:
                - float: CBS score
                - dict: { "bias_score": ..., "cbs": ..., "per_item": [...],
                          "n": ..., "details": {...} }
        """
        self._validate_inputs(templates, target_words, attribute_words, placeholder)

        target_token_id_lists = [
            self._word_to_token_ids(
                w, allow_multi_token_targets=allow_multi_token_targets
            )
            for w in target_words
        ]

        variances: List[float] = []
        breakdown: Dict[str, Dict[str, float | str]] = {}

        for template in templates:
            # If the attribute placeholder is after the target mask in the
            # template, the target's mask tokens are the first occurrences
            # once both are expanded into the prior sentence; otherwise they
            # come after the (whole-word-masked) attribute's mask tokens.
            target_before_attr = self._target_before_attribute(template, placeholder)

            for attr in attribute_words:
                attribute_token_ids = self.tokenizer.encode(
                    attr, add_special_tokens=False
                )
                attribute_num = max(1, len(attribute_token_ids))

                log_norm_probs = self._log_normalized_target_scores(
                    template=template,
                    placeholder=placeholder,
                    attr=attr,
                    attribute_num=attribute_num,
                    target_token_id_lists=target_token_id_lists,
                    target_before_attr=target_before_attr,
                )

                # Sample variance (ddof=1), matching the reference's
                # pandas .var(); undefined for a single target, so fall back
                # to 0.0 (no variance with one observation) rather than NaN.
                var_value = (
                    float(np.var(log_norm_probs, ddof=1))
                    if len(log_norm_probs) > 1
                    else 0.0
                )
                top_idx = int(np.argmax(log_norm_probs))
                top_target = target_words[top_idx]

                key = f"template={template} | attr={attr}"
                breakdown[key] = {
                    "variance": var_value,
                    "top_target": top_target,
                }
                variances.append(var_value)

        cbs_score = float(np.mean(variances)) if variances else 0.0

        if not return_details:
            return cbs_score

        return {
            "bias_score": cbs_score,
            "cbs": cbs_score,
            "per_item": variances,
            "n": len(variances),
            "details": breakdown,
        }

    def _validate_inputs(
        self,
        templates: List[str],
        target_words: List[str],
        attribute_words: List[str],
        placeholder: str,
    ) -> None:
        if len(templates) == 0:
            raise ValueError("templates cannot be empty")
        if len(target_words) == 0:
            raise ValueError("target_words cannot be empty")
        if len(attribute_words) == 0:
            raise ValueError("attribute_words cannot be empty")

        for t in templates:
            if self.mask_token not in t:
                raise ValueError(
                    f"Each template must include the mask token {self.mask_token}."
                )
            if t.count(self.mask_token) != 1:
                raise ValueError(
                    "Each template must contain exactly ONE mask token (target slot)."
                )
            if placeholder not in t:
                raise ValueError(
                    f"Each template must include the placeholder {placeholder}."
                )

    def _word_to_token_ids(
        self, word: str, allow_multi_token_targets: bool
    ) -> List[int]:
        ids = self.tokenizer.encode(word, add_special_tokens=False)

        if len(ids) == 0:
            raise ValueError(f"Target word '{word}' produced no tokens (unexpected).")

        if (not allow_multi_token_targets) and len(ids) != 1:
            raise ValueError(
                f"Target word '{word}' is not a single token for this tokenizer. "
                f"Got {len(ids)} tokens. Enable allow_multi_token_targets=True to allow this."
            )

        return ids

    def _target_before_attribute(self, template: str, placeholder: str) -> bool:
        placeholder_pos = template.index(placeholder)
        target_mask_pos = template.index(self.mask_token)
        return target_mask_pos < placeholder_pos

    def _log_normalized_target_scores(
        self,
        template: str,
        placeholder: str,
        attr: str,
        attribute_num: int,
        target_token_id_lists: List[List[int]],
        target_before_attr: bool,
    ) -> np.ndarray:
        """
        Whole-word-masked log-normalized scores per target word (Ahn & Oh
        2021 sec 3.2): a target word split into W subwords gets W mask
        tokens at the target slot (one per subword), and the W per-subword
        log P_target - log P_prior differences are summed (the paper's
        "aggregate each token's probability by multiplying" restated in log
        space). The attribute is whole-word-masked the same way in the
        prior sentence -- matching the reference's `attribute_mask`
        construction -- so the prior sentence has the same length as the
        target sentence except for the attribute being masked instead of
        filled in.

        Target words are grouped by subword count: each group needs its own
        sentence pair (different number of target mask tokens) and gets one
        forward pass, shared across every target word in the group.
        """
        scores = np.empty(len(target_token_id_lists), dtype=float)

        groups: Dict[int, List[int]] = defaultdict(list)
        for idx, ids in enumerate(target_token_id_lists):
            groups[len(ids)].append(idx)

        for k, indices in groups.items():
            target_masks = " ".join([self.mask_token] * k)
            prompt_target = template.replace(
                self.mask_token, target_masks, 1
            ).replace(placeholder, attr)

            attribute_masks = " ".join([self.mask_token] * attribute_num)
            prompt_prior = template.replace(
                self.mask_token, target_masks, 1
            ).replace(placeholder, attribute_masks)

            if target_before_attr:
                target_ordinals_prior = list(range(k))
            else:
                target_ordinals_prior = list(
                    range(attribute_num, attribute_num + k)
                )

            logits_target = self._mask_logits_multi(prompt_target, list(range(k)))
            logits_prior = self._mask_logits_multi(prompt_prior, target_ordinals_prior)

            logp_target = [torch.log_softmax(lg, dim=-1) for lg in logits_target]
            logp_prior = [torch.log_softmax(lg, dim=-1) for lg in logits_prior]

            for idx in indices:
                ids = target_token_id_lists[idx]
                score = 0.0
                for pos, token_id in enumerate(ids):
                    score += float(
                        logp_target[pos][token_id].item()
                        - logp_prior[pos][token_id].item()
                    )
                scores[idx] = score

        return scores

    def _mask_logits_multi(
        self, prompt: str, ordinals: List[int]
    ) -> List[torch.Tensor]:
        """
        Return logits at the given zero-based [MASK] occurrences in the
        tokenized prompt, one forward pass shared across all requested
        ordinals.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits  # [1, seq_len, vocab]

        mask_positions = (inputs["input_ids"] == self.mask_token_id).nonzero(
            as_tuple=False
        )
        n_masks = mask_positions.shape[0]
        if n_masks == 0:
            raise ValueError("Mask token not found after tokenization (unexpected).")

        result = []
        for ordinal in ordinals:
            if ordinal < 0 or ordinal >= n_masks:
                raise ValueError(
                    f"Requested mask_ordinal={ordinal}, but prompt has {n_masks} "
                    "mask tokens."
                )
            mask_index = mask_positions[ordinal, 1].item()
            result.append(logits[0, mask_index, :])
        return result

    def _mask_logits(self, prompt: str, mask_ordinal: int = 0) -> torch.Tensor:
        """Single-ordinal convenience wrapper around _mask_logits_multi."""
        return self._mask_logits_multi(prompt, [mask_ordinal])[0]

    def _log_normalized_probs(
        self,
        prompt_target: str,
        prompt_prior: str,
        target_token_ids: List[int],
    ) -> np.ndarray:
        """
        Backward-compatible helper for single-mask, single-token use.

        log P'(n) = log P_target(n) - log P_prior(n)
        """
        logits_target = self._mask_logits(prompt_target, mask_ordinal=0)
        logits_prior = self._mask_logits(prompt_prior, mask_ordinal=0)

        logp_target = torch.log_softmax(logits_target, dim=-1)
        logp_prior = torch.log_softmax(logits_prior, dim=-1)

        lt = logp_target[target_token_ids]
        lp = logp_prior[target_token_ids]

        return (lt - lp).detach().cpu().numpy()
