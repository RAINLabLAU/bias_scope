from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol

import numpy as np


class TokenPredictionScorer(Protocol):
    """
    Unified scorer protocol for probability-based token prediction metrics.

    Implementations provide token-level probabilities, masked-token candidate
    probabilities, optional attention-aware token probabilities, and sentence
    pseudo-log-probabilities through one shared interface.
    """

    def token_probability(self, tokens: List[str], position: int) -> float:
        """Return P(tokens[position] | tokens with that position masked)."""
        ...

    def token_probability_with_attention(
        self, tokens: List[str], position: int
    ) -> Dict[str, Any]:
        """Return probability and token-level attention for a token position."""
        ...

    def masked_token_probability(self, context: List[str], candidate: str) -> float:
        """Return P(candidate | context), where context contains one mask token."""
        ...

    def logprob(self, tokens: List[str], batch_size: int = 16) -> float:
        """Return a sentence-level pseudo-log-probability."""
        ...


class BertPLLScorer:
    """
    BERT-style masked language model scorer implementing TokenPredictionScorer.

    The scorer mirrors the model_name/device pattern used by CBS and DisCoMetric
    while exposing one reusable adapter for CrowSPairs, AUL, AULA, CAT, ICAT,
    LMB, and LPBS.
    """

    def __init__(
        self, model_name: str = "bert-base-uncased", device: Optional[str] = None
    ):
        try:
            import torch
            from transformers import AutoModelForMaskedLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "BertPLLScorer requires torch and transformers to be installed."
            ) from exc

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)

        if self.tokenizer.mask_token is None or self.tokenizer.mask_token_id is None:
            raise ValueError(
                "Tokenizer has no mask token. Use a masked language model tokenizer."
            )

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.model.to(self.device)
        self.model.eval()

        self.mask_token = self.tokenizer.mask_token
        self.mask_id = self.tokenizer.mask_token_id

    def token_probability(self, tokens: List[str], position: int) -> float:
        if position < 0 or position >= len(tokens):
            raise IndexError(f"position {position} is outside tokens length {len(tokens)}")

        context = list(tokens)
        candidate = context[position]
        context[position] = self.mask_token
        return self.masked_token_probability(context, candidate)

    def token_probability_with_attention(
        self, tokens: List[str], position: int
    ) -> Dict[str, Any]:
        if position < 0 or position >= len(tokens):
            raise IndexError(f"position {position} is outside tokens length {len(tokens)}")

        context = list(tokens)
        candidate = context[position]
        context[position] = self.mask_token
        prob, attention = self._masked_probability_and_attention(context, candidate)
        return {"prob": prob, "attention": attention}

    def masked_token_probability(self, context: List[str], candidate: str) -> float:
        prob, _ = self._masked_probability_and_attention(context, candidate)
        return prob

    def logprob(self, tokens: List[str], batch_size: int = 16) -> float:
        _ = batch_size  # Kept for compatibility with the previous PLL scorer API.
        if len(tokens) == 0:
            raise ValueError("tokens cannot be empty")

        total_logprob = 0.0
        for position in range(len(tokens)):
            prob = self.token_probability(tokens, position)
            if prob <= 0 or prob > 1:
                raise ValueError(
                    f"Invalid probability {prob} for token '{tokens[position]}' "
                    f"at position {position}. Must be in (0, 1]."
                )
            total_logprob += float(np.log(prob))
        return total_logprob

    def _masked_probability_and_attention(
        self, context: List[str], candidate: str
    ) -> tuple[float, np.ndarray]:
        if len(context) == 0:
            raise ValueError("context cannot be empty")

        normalized_context = [
            self.mask_token if token in {"[MASK]", self.mask_token} else token
            for token in context
        ]
        mask_count = sum(token == self.mask_token for token in normalized_context)
        if mask_count != 1:
            raise ValueError(
                f"context must contain exactly one mask token, got {mask_count}"
            )

        candidate_ids = self.tokenizer.encode(candidate, add_special_tokens=False)
        if len(candidate_ids) != 1:
            raise ValueError(
                f"candidate '{candidate}' must tokenize to exactly one token for BertPLLScorer. "
                f"Got {len(candidate_ids)} tokens."
            )

        enc = self.tokenizer(
            normalized_context,
            is_split_into_words=True,
            return_tensors="pt",
        )
        try:
            word_ids = enc.word_ids(batch_index=0)
        except ValueError:
            word_ids = [None] * enc["input_ids"].shape[1]
        inputs = {key: value.to(self.device) for key, value in enc.items()}

        mask_positions = (inputs["input_ids"] == self.mask_id).nonzero(as_tuple=False)
        if mask_positions.numel() == 0:
            raise ValueError("Mask token not found after tokenization.")
        if mask_positions.shape[0] != 1:
            raise ValueError(
                f"Expected exactly one mask token after tokenization, got {mask_positions.shape[0]}."
            )

        batch_index = mask_positions[0, 0].item()
        mask_index = mask_positions[0, 1].item()

        with self._torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)

        logits = outputs.logits[batch_index, mask_index, :]
        log_probs = self._torch.log_softmax(logits, dim=-1)
        candidate_id = candidate_ids[0]
        prob = float(self._torch.exp(log_probs[candidate_id]).item())

        attention = self._token_attention(outputs, word_ids, mask_index, len(context))
        return prob, attention

    def _token_attention(
        self, outputs: Any, word_ids: List[int | None], mask_index: int, n_tokens: int
    ) -> np.ndarray:
        if not getattr(outputs, "attentions", None):
            return np.ones(n_tokens, dtype=float) / n_tokens

        last_layer = outputs.attentions[-1][0]
        token_attention = last_layer.mean(dim=0)[mask_index].detach().cpu().numpy()

        weights = np.zeros(n_tokens, dtype=float)
        counts = np.zeros(n_tokens, dtype=float)
        for token_index, word_index in enumerate(word_ids):
            if word_index is None or word_index >= n_tokens:
                continue
            weights[word_index] += float(token_attention[token_index])
            counts[word_index] += 1.0

        nonzero = counts > 0
        weights[nonzero] /= counts[nonzero]
        weight_sum = weights.sum()
        if weight_sum <= 0:
            return np.ones(n_tokens, dtype=float) / n_tokens
        return weights / weight_sum