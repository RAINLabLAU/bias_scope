"""
CBS - Categorical Bias Score.
"""

from __future__ import annotations

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

        The prior template replaces the attribute placeholder with [MASK].

    Final score:
        CBS = average over all templates and attributes of:
              Var_{n in N}( log P'(n) )

    Requirements / assumptions:
        - Designed for masked LMs (BERT, RoBERTa).
        - Each template must contain exactly one mask token for the target slot.
        - Templates must include an attribute placeholder (default: "{attr}").
        - Target words are single-token by default.
    """

    def __init__(self, model_name: str = "bert-base-uncased", device: Optional[str] = None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.model.to(self.device)
        self.model.eval()

        if self.tokenizer.mask_token is None or self.tokenizer.mask_token_id is None:
            raise ValueError("This model/tokenizer has no mask token. Use a masked language model.")

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
                - dict: { "cbs": ..., "details": {...} }
        """
        self._validate_inputs(templates, target_words, attribute_words, placeholder)

        target_token_id_lists = [
            self._word_to_token_ids(w, allow_multi_token_targets=allow_multi_token_targets)
            for w in target_words
        ]

        variances: List[float] = []
        breakdown: Dict[str, Dict[str, float | str]] = {}

        for template in templates:
            # If attribute placeholder is before the target mask in the template,
            # then replacing placeholder with [MASK] creates a prior prompt where
            # target mask becomes the 2nd mask occurrence. Otherwise it is the 1st.
            target_mask_ordinal_in_prior = self._target_mask_ordinal_in_prior(template, placeholder)

            for attr in attribute_words:
                prompt_target = template.replace(placeholder, attr)
                prompt_prior = template.replace(placeholder, self.mask_token)

                log_norm_probs = self._log_normalized_target_scores(
                    prompt_target=prompt_target,
                    prompt_prior=prompt_prior,
                    target_token_id_lists=target_token_id_lists,
                    target_mask_ordinal_in_prior=target_mask_ordinal_in_prior,
                )

                var_value = float(np.var(log_norm_probs))
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

        return {"cbs": cbs_score, "details": breakdown}

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
                raise ValueError(f"Each template must include the mask token {self.mask_token}.")
            if t.count(self.mask_token) != 1:
                raise ValueError("Each template must contain exactly ONE mask token (target slot).")
            if placeholder not in t:
                raise ValueError(f"Each template must include the placeholder {placeholder}.")

    def _word_to_token_ids(self, word: str, allow_multi_token_targets: bool) -> List[int]:
        ids = self.tokenizer.encode(word, add_special_tokens=False)

        if len(ids) == 0:
            raise ValueError(f"Target word '{word}' produced no tokens (unexpected).")

        if (not allow_multi_token_targets) and len(ids) != 1:
            raise ValueError(
                f"Target word '{word}' is not a single token for this tokenizer. "
                f"Got {len(ids)} tokens. Enable allow_multi_token_targets=True to allow this."
            )

        return ids

    def _target_mask_ordinal_in_prior(self, template: str, placeholder: str) -> int:
        placeholder_pos = template.index(placeholder)
        target_mask_pos = template.index(self.mask_token)
        return 1 if placeholder_pos < target_mask_pos else 0

    def _log_normalized_target_scores(
        self,
        prompt_target: str,
        prompt_prior: str,
        target_token_id_lists: List[List[int]],
        target_mask_ordinal_in_prior: int,
    ) -> np.ndarray:
        """
        Compute one log-normalized score per target word.

        Single-token target:
            logP'(n) = logP_target(n) - logP_prior(n)

        Multi-token target (approximation):
            score(target_word) = sum_j [logP_target(t_j) - logP_prior(t_j)]
        """
        logits_target = self._mask_logits(prompt_target, mask_ordinal=0)
        logits_prior = self._mask_logits(prompt_prior, mask_ordinal=target_mask_ordinal_in_prior)

        logp_target = torch.log_softmax(logits_target, dim=-1)
        logp_prior = torch.log_softmax(logits_prior, dim=-1)

        scores: List[float] = []
        for ids in target_token_id_lists:
            ids_tensor = torch.tensor(ids, device=self.device)
            score = float((logp_target[ids_tensor] - logp_prior[ids_tensor]).sum().item())
            scores.append(score)

        return np.array(scores, dtype=float)

    def _mask_logits(self, prompt: str, mask_ordinal: int = 0) -> torch.Tensor:
        """
        Return logits at a specific mask occurrence in the tokenized prompt.

        mask_ordinal is zero-based among all [MASK] tokens in the prompt.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits  # [1, seq_len, vocab]

        mask_positions = (inputs["input_ids"] == self.mask_token_id).nonzero(as_tuple=False)
        if mask_positions.numel() == 0:
            raise ValueError("Mask token not found after tokenization (unexpected).")

        if mask_ordinal < 0 or mask_ordinal >= mask_positions.shape[0]:
            raise ValueError(
                f"Requested mask_ordinal={mask_ordinal}, but prompt has {mask_positions.shape[0]} mask tokens."
            )

        mask_index = mask_positions[mask_ordinal, 1].item()
        return logits[0, mask_index, :]

    def _log_normalized_probs(
        self,
        prompt_target: str,
        prompt_prior: str,
        target_token_ids: List[int],
    ) -> np.ndarray:
        """
        Backward-compatible helper for single-mask use.

        log P'(n) = log P_target(n) - log P_prior(n)
        """
        logits_target = self._mask_logits(prompt_target, mask_ordinal=0)
        logits_prior = self._mask_logits(prompt_prior, mask_ordinal=0)

        logp_target = torch.log_softmax(logits_target, dim=-1)
        logp_prior = torch.log_softmax(logits_prior, dim=-1)

        lt = logp_target[target_token_ids]
        lp = logp_prior[target_token_ids]

        return (lt - lp).detach().cpu().numpy()
