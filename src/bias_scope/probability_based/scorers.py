from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any, Dict, List, Literal, Optional, Protocol, Tuple

import numpy as np


AttentionScheme = Literal["last_layer_diag", "all_layers_avg"]


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
        self,
        model_name: str = "bert-base-uncased",
        device: Optional[str] = None,
        *,
        attention_scheme: AttentionScheme = "last_layer_diag",
    ):
        try:
            import torch
            from transformers import AutoModelForMaskedLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "BertPLLScorer requires torch and transformers to be installed."
            ) from exc

        if attention_scheme not in ("last_layer_diag", "all_layers_avg"):
            raise ValueError(
                f"attention_scheme must be 'last_layer_diag' or 'all_layers_avg', "
                f"got {attention_scheme!r}"
            )

        self._torch = torch
        self.attention_scheme = attention_scheme
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # eager attention is required for output_attentions to work reliably
        self.model = AutoModelForMaskedLM.from_pretrained(
            model_name, attn_implementation="eager"
        )

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
        if len(candidate_ids) == 0:
            raise ValueError(
                f"candidate '{candidate}' tokenized to zero pieces; cannot score."
            )

        # Expand the single [MASK] slot into N mask slots in the whitespace-level
        # context so the WordPiece tokenizer produces exactly N mask WordPieces
        # aligned with candidate_ids (one per subword). Multi-piece candidates
        # are then scored via Salazar-style chain-rule masking.
        n_pieces = len(candidate_ids)
        expanded_context: List[str] = []
        expanded_mask_ws_idx: Optional[int] = None
        for ws_i, tok in enumerate(normalized_context):
            if tok == self.mask_token:
                expanded_context.extend([self.mask_token] * n_pieces)
                expanded_mask_ws_idx = ws_i
            else:
                expanded_context.append(tok)

        enc = self.tokenizer(
            expanded_context,
            is_split_into_words=True,
            return_tensors="pt",
        )
        try:
            word_ids = enc.word_ids(batch_index=0)
        except ValueError:
            word_ids = [None] * enc["input_ids"].shape[1]

        input_ids = enc["input_ids"][0].tolist()
        mask_positions = [i for i, tid in enumerate(input_ids) if tid == self.mask_id]
        if len(mask_positions) != n_pieces:
            raise ValueError(
                f"Expected {n_pieces} mask tokens after tokenization, got "
                f"{len(mask_positions)}. This can happen if the tokenizer "
                f"re-normalizes the mask token."
            )

        if n_pieces == 1:
            # Fast path: single forward pass, direct log P(candidate).
            input_tensor = enc["input_ids"].to(self.device)
            attn_mask = enc.get("attention_mask")
            attn_mask = attn_mask.to(self.device) if attn_mask is not None else None
            with self._torch.no_grad():
                outputs = self.model(
                    input_ids=input_tensor,
                    attention_mask=attn_mask,
                    output_attentions=True,
                )
            mask_idx = mask_positions[0]
            log_probs = self._torch.log_softmax(
                outputs.logits[0, mask_idx, :], dim=-1
            )
            prob = float(self._torch.exp(log_probs[candidate_ids[0]]).item())
            attention = self._token_attention(
                outputs, word_ids, mask_idx, len(context)
            )
            return prob, attention

        # Multi-piece chain-rule: reveal one candidate subword at a time and
        # accumulate log P(subword_i | context, revealed subwords). Reference:
        # Salazar et al. 2019 (pseudo-log-likelihood scoring).
        log_prob_total = 0.0
        working_ids = list(input_ids)
        last_outputs = None
        last_mask_idx = mask_positions[0]
        attn_mask = enc.get("attention_mask")
        attn_mask_dev = attn_mask.to(self.device) if attn_mask is not None else None
        for step, mask_idx in enumerate(mask_positions):
            input_tensor = self._torch.tensor([working_ids], device=self.device)
            with self._torch.no_grad():
                outputs = self.model(
                    input_ids=input_tensor,
                    attention_mask=attn_mask_dev,
                    output_attentions=True,
                )
            log_probs = self._torch.log_softmax(
                outputs.logits[0, mask_idx, :], dim=-1
            )
            log_prob_total += float(log_probs[candidate_ids[step]].item())
            working_ids[mask_idx] = candidate_ids[step]
            last_outputs = outputs
            last_mask_idx = mask_idx

        prob = float(np.exp(log_prob_total))
        prob = max(min(prob, 1.0), 1e-30)
        attention = self._token_attention(
            last_outputs, word_ids, last_mask_idx, len(context)
        )
        return prob, attention

    def _token_attention(
        self, outputs: Any, word_ids: List[int | None], mask_index: int, n_tokens: int
    ) -> np.ndarray:
        if not getattr(outputs, "attentions", None):
            return np.ones(n_tokens, dtype=float) / n_tokens

        if self.attention_scheme == "all_layers_avg":
            # Kaneko github style: mean over all layers × heads × from-positions.
            stacked = self._torch.cat(
                [layer[0] for layer in outputs.attentions], dim=0
            )  # (L*H, T, T)
            per_wordpiece = stacked.mean(dim=0).mean(dim=0).detach().cpu().numpy()  # (T,)
        else:
            # last_layer_diag (existing default): attention paid BY the mask
            # position TO each other position, averaged over heads.
            last_layer = outputs.attentions[-1][0]
            per_wordpiece = last_layer.mean(dim=0)[mask_index].detach().cpu().numpy()

        weights = np.zeros(n_tokens, dtype=float)
        counts = np.zeros(n_tokens, dtype=float)
        for token_index, word_index in enumerate(word_ids):
            if word_index is None or word_index >= n_tokens:
                continue
            weights[word_index] += float(per_wordpiece[token_index])
            counts[word_index] += 1.0

        nonzero = counts > 0
        weights[nonzero] /= counts[nonzero]
        weight_sum = weights.sum()
        if weight_sum <= 0:
            return np.ones(n_tokens, dtype=float) / n_tokens
        return weights / weight_sum


class WordPieceBertScorer:
    """
    WordPiece-level BERT scorer for paper-faithful reproductions of
    Nangia 2020 (CrowS-Pairs) and Kaneko & Bollegala 2022 (AUL, AULA).

    Unlike :class:`BertPLLScorer`, which operates on whitespace tokens,
    this scorer works directly on WordPiece IDs — so it can score any
    natural sentence pair regardless of how many subwords the target
    words split into, and it aligns the two sentences at the subword
    level rather than requiring same whitespace-token length.

    Usage
    -----
    The three probability metrics (``CrowSPairs``, ``AUL``, ``AULA``)
    accept this scorer via ``mode="wordpiece"`` on their constructor.
    """

    def __init__(
        self, model_name: str = "bert-base-uncased", device: Optional[str] = None
    ):
        try:
            import torch
            from transformers import AutoModelForMaskedLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "WordPieceBertScorer requires torch and transformers to be installed."
            ) from exc

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # eager attention is required for output_attentions
        self.model = AutoModelForMaskedLM.from_pretrained(
            model_name, attn_implementation="eager"
        )
        if self.tokenizer.mask_token_id is None:
            raise ValueError(
                "Tokenizer has no mask token. Use a masked language model tokenizer."
            )
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(self.device).eval()
        self.mask_id = self.tokenizer.mask_token_id

    def encode(self, sentence: str) -> List[int]:
        """Encode a sentence to WordPiece IDs, including special tokens."""
        return self.tokenizer.encode(sentence)

    @staticmethod
    def align_unmodified(
        ids_a: List[int], ids_b: List[int]
    ) -> Tuple[List[int], List[int]]:
        """difflib opcode-based alignment on subword IDs.

        Returns two parallel position lists in ids_a / ids_b such that
        ids_a[pos_a[k]] == ids_b[pos_b[k]] for every k — these are the
        UNMODIFIED subwords per Nangia 2020's CrowS-Pairs protocol.
        """
        sm = SequenceMatcher(None, ids_a, ids_b, autojunk=False)
        pos_a: List[int] = []
        pos_b: List[int] = []
        for op, i1, i2, j1, j2 in sm.get_opcodes():
            if op == "equal":
                for k in range(i2 - i1):
                    pos_a.append(i1 + k)
                    pos_b.append(j1 + k)
        return pos_a, pos_b

    def pll_over_positions(
        self, input_ids: List[int], positions: List[int]
    ) -> float:
        """Sum of log P(input_ids[p] | rest) for p in positions.

        Each position is masked in turn; the model is called once per
        position. Matches Nangia 2020 pseudo-log-likelihood.
        """
        if not positions:
            return 0.0
        import torch  # local for typing; already imported in __init__
        _ = torch  # keep for readability

        total = 0.0
        for pos in positions:
            masked = list(input_ids)
            masked[pos] = self.mask_id
            input_tensor = self._torch.tensor([masked], device=self.device)
            with self._torch.no_grad():
                out = self.model(input_ids=input_tensor)
            log_probs = self._torch.log_softmax(out.logits[0, pos], dim=-1)
            total += float(log_probs[input_ids[pos]].item())
        return total

    def aul_aula(self, input_ids: List[int]) -> Tuple[float, float]:
        """Return (AUL, AULA) per Kaneko & Bollegala 2022.

        AUL:  mean of log P(input_ids[i] | full sentence, unmasked forward),
              taken over interior positions (i.e. excluding [CLS] and [SEP]).
        AULA: same as AUL, weighted by attention received at each interior
              position, aggregated over all layers × heads × from-positions.
        """
        T = len(input_ids)
        if T < 3:
            return 0.0, 0.0
        with self._torch.no_grad():
            out = self.model(
                input_ids=self._torch.tensor([input_ids], device=self.device),
                output_attentions=True,
            )
        log_probs = self._torch.log_softmax(out.logits[0], dim=-1)

        interior = range(1, T - 1)
        token_lp = self._torch.stack(
            [log_probs[i, input_ids[i]] for i in interior]
        )

        # Mean over all layers × heads → (T, T) attention[from, to].
        stacked = self._torch.cat([layer[0] for layer in out.attentions], dim=0)
        attn_mean = stacked.mean(dim=0)
        # Attention RECEIVED by each token = mean over from-positions.
        attn_received = attn_mean.mean(dim=0)[1:-1]

        aul = float(token_lp.mean().item())
        aula = float((token_lp * attn_received).mean().item())
        return aul, aula

    # --- TokenPredictionScorer protocol compatibility ---

    def token_probability(self, tokens: List[str], position: int) -> float:
        """P(whitespace token[position] | rest) — used when this scorer is
        plugged into ``CrowSPairs.evaluate(predict_masked_token=scorer)`` or
        ``AUL.evaluate(predict_token_given_sentence=scorer)`` for the
        whitespace-based algorithm paths."""
        enc = self.tokenizer(tokens, is_split_into_words=True, return_tensors="pt")
        word_ids = enc.word_ids(batch_index=0)
        input_ids = enc.input_ids[0].tolist()
        target_idx = [i for i, wid in enumerate(word_ids) if wid == position]
        if not target_idx:
            raise ValueError(f"No WordPiece span for whitespace position {position}")

        lp_total = self.pll_over_positions(input_ids, target_idx)
        prob = float(np.exp(lp_total))
        return max(min(prob, 1.0), 1e-30)

    def token_probability_with_attention(
        self, tokens: List[str], position: int
    ) -> Dict[str, Any]:
        """As above plus a per-whitespace-token attention vector."""
        enc = self.tokenizer(tokens, is_split_into_words=True, return_tensors="pt")
        word_ids = enc.word_ids(batch_index=0)
        input_ids = enc.input_ids[0].tolist()
        target_idx = [i for i, wid in enumerate(word_ids) if wid == position]
        if not target_idx:
            raise ValueError(f"No WordPiece span for whitespace position {position}")

        lp_total = self.pll_over_positions(input_ids, target_idx)
        prob = max(min(float(np.exp(lp_total)), 1.0), 1e-30)

        with self._torch.no_grad():
            out = self.model(
                input_ids=self._torch.tensor([input_ids], device=self.device),
                output_attentions=True,
            )
        stacked = self._torch.cat([layer[0] for layer in out.attentions], dim=0)
        attn_received = stacked.mean(dim=0).mean(dim=0).detach().cpu().numpy()

        n_ws = len(tokens)
        agg = np.zeros(n_ws, dtype=float)
        counts = np.zeros(n_ws, dtype=float)
        for j, wid in enumerate(word_ids):
            if wid is None or wid >= n_ws:
                continue
            agg[wid] += float(attn_received[j])
            counts[wid] += 1.0
        nz = counts > 0
        agg[nz] /= counts[nz]
        s = agg.sum()
        if s > 0:
            agg /= s
        return {"prob": prob, "attention": agg}