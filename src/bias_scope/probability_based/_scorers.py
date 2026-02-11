import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer


class BertPLLScorer:
    """
    Compute pseudo-log-likelihood (PLL) for a sentence using a masked LM (e.g., BERT).

    PLL(sentence) = sum_{i in tokens} log P(token_i | sentence with token_i masked)

    This is a common way to get "sentence log-prob" from MLMs, which don't define a
    true left-to-right likelihood like GPT.
    """

    def __init__(
        self, model_name: str = "bert-base-uncased", device: str | None = None
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)

        if self.tokenizer.mask_token_id is None:
            raise ValueError(
                "Tokenizer has no mask token. Use an MLM tokenizer (BERT/RoBERTa)."
            )

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.model.to(self.device)
        self.model.eval()

        self.mask_id = self.tokenizer.mask_token_id

    def logprob(self, tokens: list[str], batch_size: int = 16) -> float:
        """
        Args:
            tokens: token list (your library format), e.g. ["The","doctor","is","kind","."]
            batch_size: masks per forward pass (bigger = faster, more memory)

        Returns:
            float: PLL score (sum of log probs across masked positions)
        """
        # Join tokens into a sentence. BERT tokenization will then produce wordpieces.
        text = " ".join(tokens)

        enc = self.tokenizer(text, return_tensors="pt")
        input_ids = enc["input_ids"][0]  # shape [seq_len]
        attention_mask = enc["attention_mask"][0]

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        # We skip [CLS] and [SEP] (special tokens) at positions 0 and last.
        # Also skip any padding (not typical here, but safe).
        seq_len = input_ids.size(0)
        positions = []
        for i in range(seq_len):
            if attention_mask[i].item() == 0:
                continue
            if i == 0 or i == seq_len - 1:
                continue
            positions.append(i)

        if not positions:
            raise ValueError("Sentence too short after tokenization to compute PLL.")

        total_logprob = 0.0

        # Process positions in batches: create many masked versions, run model once per batch.
        with torch.no_grad():
            for start in range(0, len(positions), batch_size):
                batch_pos = positions[start : start + batch_size]

                batch_input_ids = input_ids.repeat(len(batch_pos), 1)  # [B, seq_len]
                batch_attention = attention_mask.repeat(
                    len(batch_pos), 1
                )  # [B, seq_len]

                # Save original token ids at each masked position
                orig_token_ids = batch_input_ids[
                    torch.arange(len(batch_pos)),
                    torch.tensor(batch_pos, device=self.device),
                ]

                # Mask the selected positions
                batch_input_ids[
                    torch.arange(len(batch_pos)),
                    torch.tensor(batch_pos, device=self.device),
                ] = self.mask_id

                outputs = self.model(
                    input_ids=batch_input_ids, attention_mask=batch_attention
                )
                logits = outputs.logits  # [B, seq_len, vocab]

                log_probs = torch.log_softmax(logits, dim=-1)

                # Extract log P(original_token | masked sentence) at each masked position
                idx = torch.arange(len(batch_pos), device=self.device)
                pos_tensor = torch.tensor(batch_pos, device=self.device)
                batch_logp = log_probs[idx, pos_tensor, orig_token_ids]  # [B]

                total_logprob += float(batch_logp.sum().item())

        # PLL score: sum of log probs (usually negative)
        return total_logprob
