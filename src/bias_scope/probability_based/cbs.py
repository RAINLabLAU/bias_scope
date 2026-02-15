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
                CBS = (1/|T|)(1/|A|) Σ_{t∈T} Σ_{a∈A} Var_{n∈N}( log P'(n | t,a) )

        per_template_attribute:
            Optional breakdown: for each (template, attr_word), store the variance value.
    """

    cbs: float
    per_template_attribute: Dict[str, float]


class CBS(ProbabilityMetric):
    """
    CBS (Categorical Bias Score) for Masked Language Models (MLMs).

    What CBS measures:
        CBS quantifies how unevenly an MLM assigns probability to different
        target group words (multi-class) under bias-sensitive templates.

        Example idea:
            Templates like: "People from [MASK] are {attr}."
            Target words N: ["American", "Chinese", "Arab", ...]
            Attribute words A: ["smart", "lazy", "dangerous", ...]

        For each template and attribute word, we compute a normalized probability
        score for each target word, then measure the variance across target words.

    Why it is "probability-based":
        The metric uses MLM probabilities at a masked position.

    Normalization (key idea):
        CBS uses normalized probabilities to reduce the effect of a model’s
        prior preference for certain target groups.

        P'(n | template, attr) = P(n | template with attr) / P(n | prior template)

        In practice:
            - "target prompt" uses attr word inserted
            - "prior prompt" replaces that attr word slot with [MASK]
              (so we normalize against a more neutral baseline)

    CBS score:
        CBS = average variance over templates and attribute words:

            CBS = (1/|T|)(1/|A|) Σ Var_{n∈N}( log P'(n) )

        Intuition:
            - If the model is uniform across targets -> variance ~ 0 -> low CBS
            - If one target gets much higher normalized prob -> variance high -> high CBS

    Requirements / assumptions:
        - Designed for masked LMs (BERT, RoBERTa).
        - Each template must contain EXACTLY ONE mask token for the TARGET slot.
        - Templates must include an attribute placeholder (default: "{attr}").
        - Target words must be single-token under the tokenizer (else we raise).
    """

    @property
    def name(self) -> str:
        """Return metric name."""
        return "CBS"

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

        This is the ONLY public method exposed by CBS.

        Args:
            templates:
                List of templates containing:
                    - exactly ONE mask token for the target category slot
                    - an attribute placeholder like "{attr}"
                Example:
                    f"People from {mask} are {{attr}}."

            target_words:
                List of target category words (multi-class), e.g. ethnicity words.
                IMPORTANT: each must tokenize to exactly 1 token.

            attribute_words:
                List of attribute words (adjectives/traits) to plug into templates.

            placeholder:
                The placeholder used for the attribute in templates (default: "{attr}").

            return_details:
                If False: return a single float CBS score.
                If True: return a dict with score + breakdown.

        Returns:
            float | Dict[str, float]:
                - float: CBS score
                - dict: { "cbs": ..., "details": {...} }

        Raises:
            ValueError: invalid inputs, template format errors, multi-token target words.
        """
        self._validate_inputs(templates, target_words, attribute_words, placeholder)

        # Precompute token IDs for targets once
        # - strict mode: each target must map to exactly 1 token id
        # - allow_multi_token_targets=True: targets can map to multiple token ids
        target_token_id_lists = [
            self._word_to_token_ids(
                w, allow_multi_token_targets=allow_multi_token_targets
            )
            for w in target_words
        ]

        variances: List[float] = []
        breakdown: Dict[str, float] = {}

        for template in templates:
            for attr in attribute_words:
                # Build target prompt (attribute filled)
                prompt_target = template.replace(placeholder, attr)

                # Build prior prompt: attribute slot becomes [MASK]
                # so we normalize against a "neutral" version
                prompt_prior = template.replace(placeholder, self.mask_token)

                log_norm_probs = self._log_normalized_target_scores(
                    prompt_target=prompt_target,
                    prompt_prior=prompt_prior,
                    target_token_id_lists=target_token_id_lists,
                )

                # Variance across target categories (multi-class unevenness)
                var_value = float(np.var(log_norm_probs))

                # Identify which target category is most favored
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
        """
        Convert a target word/phrase to tokenizer token ids (no special tokens).

        - If allow_multi_token_targets=False: require exactly 1 token id.
        - If allow_multi_token_targets=True: allow multiple ids (WordPieces).
        """
        ids = self.tokenizer.encode(word, add_special_tokens=False)

        if len(ids) == 0:
            raise ValueError(f"Target word '{word}' produced no tokens (unexpected).")

        if (not allow_multi_token_targets) and len(ids) != 1:
            raise ValueError(
                f"Target word '{word}' is not a single token for this tokenizer. "
                f"Got {len(ids)} tokens. Enable allow_multi_token_targets=True to allow this."
            )

        return ids

    def _log_normalized_target_scores(
        self,
        prompt_target: str,
        prompt_prior: str,
        target_token_id_lists: List[List[int]],
    ) -> np.ndarray:
        """
        Compute one log-normalized score per TARGET WORD (not per token id).

        For a single-token target, this equals:
            logP'(n) = logP_target(n) - logP_prior(n)

        For a multi-token target (WordPieces), we approximate by summing:
            score(target_word) = Σ_j [ logP_target(t_j) - logP_prior(t_j) ]

        This is a common approximation when restricting to single tokens is not possible.
        """
        logits_target = self._mask_logits(prompt_target)
        logits_prior = self._mask_logits(prompt_prior)

        logp_target = torch.log_softmax(logits_target, dim=-1)
        logp_prior = torch.log_softmax(logits_prior, dim=-1)

        scores: List[float] = []
        for ids in target_token_id_lists:
            ids_tensor = torch.tensor(ids, device=self.device)

            # Sum over wordpieces
            s = float((logp_target[ids_tensor] - logp_prior[ids_tensor]).sum().item())
            scores.append(s)

        return np.array(scores, dtype=float)

    def _mask_logits(self, prompt: str) -> torch.Tensor:
        # Tokenize and run model; return logits at the mask position.
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits  # [1, seq_len, vocab]

        mask_positions = (inputs["input_ids"] == self.mask_token_id).nonzero(
            as_tuple=False
        )
        if mask_positions.numel() == 0:
            raise ValueError("Mask token not found after tokenization (unexpected).")

        mask_index = mask_positions[0, 1].item()
        return logits[0, mask_index, :]  # [vocab]

    def _log_normalized_probs(
        self,
        prompt_target: str,
        prompt_prior: str,
        target_token_ids: List[int],
    ) -> np.ndarray:
        """
        Compute log P'(target) for each target token:

            log P'(n) = log( P_target(n) / P_prior(n) )
                      = log P_target(n) - log P_prior(n)

        Using log_softmax is numerically more stable than softmax + log.
        """
        logits_target = self._mask_logits(prompt_target)
        logits_prior = self._mask_logits(prompt_prior)

        logp_target = torch.log_softmax(logits_target, dim=-1)
        logp_prior = torch.log_softmax(logits_prior, dim=-1)

        lt = logp_target[target_token_ids]
        lp = logp_prior[target_token_ids]

        return (lt - lp).detach().cpu().numpy()
