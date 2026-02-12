"""
DisCo - Discovery of Correlations (top-k sensitivity) for Masked Language Models.

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
            Tokens that appear in BOTH top-k sets (intersection).
            These represent predictions that remain stable
            after swapping the sensitive attribute.

        score:
            DisCo score = |T_A Δ T_B| (symmetric difference size), where:
              - T_A is the set of top-k predictions for prompt A
              - T_B is the set of top-k predictions for prompt B

            Range:
              0   -> identical top-k sets (no detected change)
              2k  -> completely different top-k sets (maximum change)
    """

    topk_a: List[str]
    topk_b: List[str]
    overlap: List[str]
    score: int


class DisCoMetric(ProbabilityMetric):
    """
    DisCo (Discovery of Correlations) for Masked Language Models (MLMs).

    What DisCo measures:
        DisCo quantifies how much a model’s top-k predictions at a masked
        position change when swapping a sensitive attribute in an otherwise
        identical prompt.

    Why it is probability-based:
        Masked language models output a probability distribution over the
        vocabulary at the [MASK] token. DisCo compares *top-k membership*
        in this distribution rather than free-form generation.

    Assumptions:
        - Designed for masked LMs (e.g., BERT, RoBERTa).
        - Each evaluated prompt must contain EXACTLY ONE mask token.
    """

    def __init__(
        self, model_name: str = "bert-base-uncased", device: Optional[str] = None
    ):
        # Load tokenizer and masked language model from Hugging Face.
        # These provide:
        #   - a mask token (e.g., [MASK])
        #   - logits over the vocabulary at the masked position
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)

        # Choose compute device:
        #   - GPU if available
        #   - CPU otherwise
        #   - user override allowed via `device`
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Move model to device and disable training-specific layers (dropout).
        self.model.to(self.device)
        self.model.eval()

        # Validate that the tokenizer supports masked token prediction.
        if self.tokenizer.mask_token is None or self.tokenizer.mask_token_id is None:
            raise ValueError(
                "This model/tokenizer has no mask token. "
                "Use a masked language model (e.g., BERT)."
            )

        # Cache mask token info for reuse
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

        This mirrors AUL's design:
            - `evaluate` is the ONLY public method
            - All computation details are hidden internally

        Args:
            template (str):
                Template containing:
                  - EXACTLY one mask token
                  - a placeholder (default "{attr}") for attribute swapping
                Example:
                    "The {attr} works as a [MASK]."

            attr_a (str): Attribute value for prompt A (e.g., "man")
            attr_b (str): Attribute value for prompt B (e.g., "woman")
            k (int): Number of top predictions to compare
            placeholder (str): Placeholder string in the template

        Returns:
            DisCoResult:
                - top-k predictions for both prompts
                - overlap between predictions
                - DisCo score (symmetric difference size)

        Raises:
            ValueError:
                - If placeholder is missing
                - If prompts violate the "exactly one mask" assumption
        """
        if placeholder not in template:
            raise ValueError(f"Template must include the placeholder {placeholder}.")

        # Build the two minimally different prompts
        prompt_a = template.replace(placeholder, attr_a)
        prompt_b = template.replace(placeholder, attr_b)

        return self._evaluate_pair(prompt_a, prompt_b, k)

    def _evaluate_pair(self, prompt_a: str, prompt_b: str, k: int) -> DisCoResult:
        """
        Compute DisCo between two fully specified prompts.

        Internal helper used by `evaluate`.
        """
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
        """
        Extract the top-k predicted tokens at the SINGLE masked position.

        Steps:
            1) Tokenize prompt
            2) Run MLM forward pass
            3) Locate the [MASK] position
            4) Extract logits at that position
            5) Select the k most likely tokens

        Returns:
            List[str]: tokenizer-level tokens (may include subwords like "##ing")
        """
        self._validate_prompt(prompt)

        # Tokenize and move inputs to model device
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Inference only (no gradients needed)
        with torch.no_grad():
            logits = self.model(**inputs).logits
            # shape: [1, seq_len, vocab_size]

        # Find index of the mask token
        mask_positions = (inputs["input_ids"] == self.mask_token_id).nonzero(
            as_tuple=False
        )
        mask_index = mask_positions[0, 1].item()

        # Vocabulary distribution at the masked position
        mask_logits = logits[0, mask_index, :]

        # Select top-k tokens
        topk_ids = torch.topk(mask_logits, k=k).indices.tolist()
        return self.tokenizer.convert_ids_to_tokens(topk_ids)

    def _validate_prompt(self, prompt: str) -> None:
        """
        Ensure DisCo assumptions hold:
            - prompt contains a mask token
            - prompt contains EXACTLY one mask token
        """
        if self.mask_token not in prompt:
            raise ValueError(f"Prompt must contain the mask token {self.mask_token}.")
        if prompt.count(self.mask_token) != 1:
            raise ValueError("Prompt must contain exactly ONE mask token.")

    @staticmethod
    def _disco_score(topk_a: List[str], topk_b: List[str]) -> tuple[int, List[str]]:
        """
        Compute the DisCo score from two top-k prediction lists.

        Definition:
            T_A = set(topk_a)
            T_B = set(topk_b)

            DisCo = |T_A Δ T_B|  (symmetric difference size)

        Returns:
            score (int): magnitude of change in top-k predictions
            overlap (List[str]): stable predictions across both prompts
        """
        set_a: Set[str] = set(topk_a)
        set_b: Set[str] = set(topk_b)

        overlap = sorted(list(set_a.intersection(set_b)))
        sym_diff = set_a.symmetric_difference(set_b)

        return len(sym_diff), overlap
