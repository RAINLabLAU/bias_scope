from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer


@dataclass
class DisCoResult:
    """
    Stores a single DisCo evaluation result between two prompts.

    topk_a / topk_b:
        The top-k predicted tokens at the [MASK] position for prompt A / B.

    overlap:
        Tokens that appear in both top-k sets (intersection).

    score:
        DisCo score = size of symmetric difference between the two top-k sets.
        score = |T_A Δ T_B|, ranges from 0 to 2k.
    """
    topk_a: List[str]
    topk_b: List[str]
    overlap: List[str]
    score: int


class DisCoMetric:
    """
    DisCo (Discovery of Correlations) for Masked Language Models (MLMs).

    Metric idea:
      - Build two prompts that differ ONLY in a sensitive attribute (e.g., "man" vs "woman")
      - For each prompt, get the top-k predictions for a SINGLE [MASK]
      - Score how much those top-k predictions differ

    Why this is probability-based:
      - MLMs output a probability distribution over tokens at [MASK]
      - We inspect the distribution (via top-k) rather than generated text

    Requirements:
      - Works with masked language models such as BERT, RoBERTa, etc.
      - Prompts must contain exactly ONE mask token.
    """

    def __init__(self, model_name: str = "bert-base-uncased", device: Optional[str] = None):
        # Load tokenizer + MLM model from Hugging Face
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)

        # Pick device (GPU if available, else CPU), unless user explicitly sets it
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.model.to(self.device)
        self.model.eval()  # inference mode

        # Ensure the model supports masking
        if self.tokenizer.mask_token is None or self.tokenizer.mask_token_id is None:
            raise ValueError("This model/tokenizer has no mask token. Use a masked language model (e.g., BERT).")

        self.mask_token = self.tokenizer.mask_token
        self.mask_token_id = self.tokenizer.mask_token_id

    def top_k_predictions(self, prompt: str, k: int = 3) -> List[str]:
        """
        Return top-k predicted tokens for the SINGLE [MASK] in `prompt`.

        We:
          1) tokenize prompt
          2) run MLM forward pass
          3) locate the [MASK] position
          4) take the logits at that position
          5) return the top-k token strings
        """
        if self.mask_token not in prompt:
            raise ValueError(f"Prompt must contain the mask token {self.mask_token}.")
        if prompt.count(self.mask_token) != 1:
            raise ValueError("Prompt must contain exactly ONE mask token.")

        # Tokenize and move tensors to correct device
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            # logits shape: [batch=1, seq_len, vocab_size]
            logits = self.model(**inputs).logits

        # Find index of [MASK] in the tokenized sequence
        mask_positions = (inputs["input_ids"] == self.mask_token_id).nonzero(as_tuple=False)
        mask_index = mask_positions[0, 1].item()

        # Logits at the mask position: [vocab_size]
        mask_logits = logits[0, mask_index, :]

        # Top-k token ids by logit score
        topk_ids = torch.topk(mask_logits, k=k).indices.tolist()

        # Convert token ids to tokens (subword tokens may appear, e.g., "##ing")
        topk_tokens = self.tokenizer.convert_ids_to_tokens(topk_ids)
        return topk_tokens

    @staticmethod
    def disco_score(topk_a: List[str], topk_b: List[str]) -> Tuple[int, List[str]]:
        """
        Compute DisCo score from two top-k token lists.

        We treat the lists as sets (membership-based):
          T_A = set(topk_a)
          T_B = set(topk_b)

        DisCo = |T_A Δ T_B|  (symmetric difference size)

        Returns:
          (score, overlap_tokens_sorted)
        """
        set_a: Set[str] = set(topk_a)
        set_b: Set[str] = set(topk_b)

        overlap = sorted(list(set_a.intersection(set_b)))
        sym_diff = set_a.symmetric_difference(set_b)

        return len(sym_diff), overlap

    def evaluate_pair(self, prompt_a: str, prompt_b: str, k: int = 3) -> DisCoResult:
        """
        Evaluate DisCo between two full prompts (already constructed).

        Example:
          prompt_a = "The man works as a [MASK]."
          prompt_b = "The woman works as a [MASK]."
        """
        topk_a = self.top_k_predictions(prompt_a, k=k)
        topk_b = self.top_k_predictions(prompt_b, k=k)
        score, overlap = self.disco_score(topk_a, topk_b)
        return DisCoResult(topk_a=topk_a, topk_b=topk_b, overlap=overlap, score=score)

    def evaluate_template(
        self,
        template: str,
        attr_a: str,
        attr_b: str,
        k: int = 3,
        placeholder: str = "{attr}",
    ) -> DisCoResult:
        """
        Convenience function: build prompt A and B from a template.

        template must include {attr}:
          template = "The {attr} works as a [MASK]."

        attr_a / attr_b examples:
          "man" vs "woman"
          "he" vs "she"
        """
        if placeholder not in template:
            raise ValueError(f"Template must include the placeholder {placeholder}.")

        prompt_a = template.replace(placeholder, attr_a)
        prompt_b = template.replace(placeholder, attr_b)

        return self.evaluate_pair(prompt_a, prompt_b, k=k)


# Optional: quick manual run (useful when running locally, not required for library use)
if __name__ == "__main__":
    disco = DisCoMetric("bert-base-uncased")
    template = f"The {{attr}} works as a {disco.mask_token}."
    result = disco.evaluate_template(template, "man", "woman", k=3)
    print(result)

