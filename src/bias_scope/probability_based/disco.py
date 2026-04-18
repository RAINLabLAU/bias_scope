"""
DisCo - Top-k distribution shift for Masked Language Models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Set

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from bias_scope.base import ProbabilityMetric


@dataclass
class DisCoResult:
    """
    Container for the output of a single DisCo comparison.

    Fields:
        topk_a:
            Top-k predicted tokens at the [MASK] position for prompt A.

        topk_b:
            Top-k predicted tokens at the [MASK] position for prompt B.

        overlap:
            Tokens that appear in both top-k sets (intersection).

        score:
            DisCo score = |T_A Δ T_B| (symmetric difference size), where:
            - T_A is the set of top-k predictions for prompt A
            - T_B is the set of top-k predictions for prompt B

            Range:
            - 0 -> identical top-k sets
            - 2k -> completely different top-k sets
    """

    topk_a: List[str]
    topk_b: List[str]
    overlap: List[str]
    score: int


class DisCoMetric(ProbabilityMetric):
    """
    DisCo-style top-k sensitivity metric for masked LMs.

    This implementation compares top-k membership at one masked position
    between two prompts that differ only in a sensitive attribute.
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
                "This model/tokenizer has no mask token. "
                "Use a masked language model (e.g., BERT)."
            )

        self.mask_token = self.tokenizer.mask_token
        self.mask_token_id = self.tokenizer.mask_token_id

    def evaluate(
        self,
        template: str,
        attr_a: str,
        attr_b: str,
        k: int = 3,
        placeholder: str = "{attr}",
    ) -> DisCoResult:
        """
        Evaluate DisCo using a single template with an attribute placeholder.
        """
        if placeholder not in template:
            raise ValueError(f"Template must include the placeholder {placeholder}.")
        if k <= 0:
            raise ValueError("k must be a positive integer.")

        prompt_a = template.replace(placeholder, attr_a)
        prompt_b = template.replace(placeholder, attr_b)

        return self._evaluate_pair(prompt_a, prompt_b, k)

    def _evaluate_pair(self, prompt_a: str, prompt_b: str, k: int) -> DisCoResult:
        topk_a = self._top_k_predictions(prompt_a, k)
        topk_b = self._top_k_predictions(prompt_b, k)

        score, overlap = self._disco_score(topk_a, topk_b)

        return DisCoResult(
            topk_a=topk_a,
            topk_b=topk_b,
            overlap=overlap,
            score=score,
        )

    def _top_k_predictions(self, prompt: str, k: int) -> List[str]:
        self._validate_prompt(prompt)

        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits

        mask_positions = (inputs["input_ids"] == self.mask_token_id).nonzero(
            as_tuple=False
        )
        mask_index = mask_positions[0, 1].item()

        mask_logits = logits[0, mask_index, :]
        topk_ids = torch.topk(mask_logits, k=k).indices.tolist()
        return self.tokenizer.convert_ids_to_tokens(topk_ids)

    def _validate_prompt(self, prompt: str) -> None:
        if self.mask_token not in prompt:
            raise ValueError(f"Prompt must contain the mask token {self.mask_token}.")
        if prompt.count(self.mask_token) != 1:
            raise ValueError("Prompt must contain exactly ONE mask token.")

    @staticmethod
    def _disco_score(topk_a: List[str], topk_b: List[str]) -> tuple[int, List[str]]:
        """
        Compute DisCo score from top-k sets.

        Formula:
            T_A = set(topk_a)
            T_B = set(topk_b)
            score = |T_A Δ T_B|
        """
        set_a: Set[str] = set(topk_a)
        set_b: Set[str] = set(topk_b)

        overlap = sorted(list(set_a.intersection(set_b)))
        sym_diff = set_a.symmetric_difference(set_b)

        return len(sym_diff), overlap
