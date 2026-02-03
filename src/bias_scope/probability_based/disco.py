from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer


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
            Tokens that appear in BOTH top-k sets (intersection). This is useful
            for interpreting which predictions are stable across attribute swaps.

        score:
            DisCo score = |T_A Δ T_B| (symmetric difference size), where:
              - T_A is the set of top-k predictions for prompt A
              - T_B is the set of top-k predictions for prompt B

            Range:
              0   -> identical top-k sets (no change detected in top-k)
              2k  -> completely different top-k sets (maximum change)
    """

    topk_a: List[str]
    topk_b: List[str]
    overlap: List[str]
    score: int


class DisCoMetric:
    """
    DisCo (Discovery of Correlations) for Masked Language Models (MLMs).

    What DisCo measures:
        DisCo quantifies how much a model’s top-k predictions at a masked position
        change when swapping a sensitive attribute in an otherwise identical prompt.

    Why it is "probability-based":
        MLMs produce a probability distribution over the vocabulary at [MASK].
        DisCo uses this distribution (via top-k predictions) rather than relying on
        free-form generation.

    Typical usage:
        - Build two prompts differing only by an attribute term (e.g., "man" vs "woman")
        - Extract top-k tokens at the mask
        - Score the difference between the two top-k sets

    Requirements / assumptions:
        - Designed for masked LMs (e.g., BERT, RoBERTa).
        - Each prompt must contain EXACTLY ONE mask token.
    """

    def __init__(self, model_name: str = "bert-base-uncased", device: Optional[str] = None):
        # Load tokenizer and masked language model from Hugging Face.
        # These provide:
        #   - a mask token (e.g., [MASK] for BERT)
        #   - logits over the vocabulary at the masked position
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)

        # Choose compute device:
        #   - use GPU if available, otherwise CPU
        #   - allow user override via `device`
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Move model to device and set evaluation mode (disables dropout etc.).
        self.model.to(self.device)
        self.model.eval()

        # Validate the model supports masked token prediction.
        # (Some tokenizers/models do not define a mask token.)
        if self.tokenizer.mask_token is None or self.tokenizer.mask_token_id is None:
            raise ValueError(
                "This model/tokenizer has no mask token. Use a masked language model (e.g., BERT)."
            )

        # Cache mask token string and its token id for reuse.
        self.mask_token = self.tokenizer.mask_token
        self.mask_token_id = self.tokenizer.mask_token_id

    def top_k_predictions(self, prompt: str, k: int = 3) -> List[str]:
        """
        Extract the top-k predicted tokens at the SINGLE masked position in `prompt`.

        This is the core probability-based step in DisCo:
          1) tokenize the prompt
          2) run the MLM forward pass to get logits
          3) locate the [MASK] token index in the tokenized sequence
          4) take logits at that index (vocab distribution)
          5) return the k most likely tokens

        Args:
            prompt (str): Prompt containing exactly one mask token.
            k (int): Number of top tokens to return.

        Returns:
            List[str]: The top-k predicted token strings (may include subword tokens like "##ing").
        """
        # DisCo assumes exactly one masked position. This keeps comparisons controlled and consistent.
        if self.mask_token not in prompt:
            raise ValueError(f"Prompt must contain the mask token {self.mask_token}.")
        if prompt.count(self.mask_token) != 1:
            raise ValueError("Prompt must contain exactly ONE mask token.")

        # Tokenize the prompt into tensors and move them to the same device as the model.
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        # Inference-only: no gradients needed.
        with torch.no_grad():
            # logits shape: [batch_size=1, seq_len, vocab_size]
            logits = self.model(**inputs).logits

        # Locate the position (index) of the mask token in the tokenized input.
        # nonzero gives positions where input_ids == mask_token_id.
        mask_positions = (inputs["input_ids"] == self.mask_token_id).nonzero(as_tuple=False)
        mask_index = mask_positions[0, 1].item()

        # Extract logits at the masked position -> distribution over the entire vocabulary.
        mask_logits = logits[0, mask_index, :]  # shape: [vocab_size]

        # Select the top-k vocabulary ids with highest logit values.
        topk_ids = torch.topk(mask_logits, k=k).indices.tolist()

        # Convert token ids to tokens (note: these are tokenizer tokens, can include subwords).
        return self.tokenizer.convert_ids_to_tokens(topk_ids)

    @staticmethod
    def disco_score(topk_a: List[str], topk_b: List[str]) -> Tuple[int, List[str]]:
        """
        Compute the DisCo score from two top-k token lists.

        DisCo compares membership in the top-k sets (not probabilities themselves):
            T_A = set(topk_a)
            T_B = set(topk_b)

        Score definition:
            DisCo = |T_A Δ T_B|  (size of symmetric difference)

        Intuition:
            - If swapping the attribute barely affects predictions, top-k sets overlap a lot -> low score.
            - If swapping the attribute changes predictions strongly, top-k sets differ -> high score.

        Args:
            topk_a (List[str]): Top-k predicted tokens for prompt A.
            topk_b (List[str]): Top-k predicted tokens for prompt B.

        Returns:
            Tuple[int, List[str]]:
                score: size of symmetric difference
                overlap: sorted list of common tokens (intersection)
        """
        set_a: Set[str] = set(topk_a)
        set_b: Set[str] = set(topk_b)

        overlap = sorted(list(set_a.intersection(set_b)))
        sym_diff = set_a.symmetric_difference(set_b)

        return len(sym_diff), overlap

    def evaluate_pair(self, prompt_a: str, prompt_b: str, k: int = 3) -> DisCoResult:
        """
        Evaluate DisCo between two fully specified prompts.

        Use this when you already constructed the prompts manually.

        Args:
            prompt_a (str): Prompt A with exactly one mask token.
            prompt_b (str): Prompt B with exactly one mask token.
            k (int): Top-k size used for comparison.

        Returns:
            DisCoResult: top-k predictions for each prompt, overlap, and DisCo score.

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
        Convenience helper: build prompt A and B from a single template.

        The template must include a placeholder (default: "{attr}") which is replaced by attr_a/attr_b.

        Args:
            template (str): e.g., "The {attr} works as a [MASK]."
            attr_a (str): attribute value for prompt A (e.g., "man")
            attr_b (str): attribute value for prompt B (e.g., "woman")
            k (int): Top-k size used for comparison.
            placeholder (str): placeholder string to replace in template.

        Returns:
            DisCoResult: computed DisCo result for the generated prompt pair.
        """
        if placeholder not in template:
            raise ValueError(f"Template must include the placeholder {placeholder}.")

        prompt_a = template.replace(placeholder, attr_a)
        prompt_b = template.replace(placeholder, attr_b)

        return self.evaluate_pair(prompt_a, prompt_b, k=k)


# Optional: quick manual run for debugging when running locally.
# This is not required when importing the metric as part of the library.
if __name__ == "__main__":
    disco = DisCoMetric("bert-base-uncased")
    template = f"The {{attr}} works as a {disco.mask_token}."
    result = disco.evaluate_template(template, "man", "woman", k=3)
    print(result)
