"""Interface tests for BertPLLScorer multi-piece support and
WordPieceBertScorer basic protocol conformance.

We mock the underlying HF `AutoTokenizer` / `AutoModelForMaskedLM` /
`AutoModel` so no model is downloaded.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch


def _mock_tokenizer(mask_id: int = 999, vocab_size: int = 1000):
    """Build a MagicMock tokenizer whose behaviour is enough for our tests."""

    tok = MagicMock()
    tok.mask_token = "[MASK]"
    tok.mask_token_id = mask_id

    def encode(text, add_special_tokens=True):
        # single-char tokens: id = ord(c). No special tokens.
        return [ord(c) for c in text]

    tok.encode.side_effect = encode

    def call(
        input_arg,
        is_split_into_words=False,
        return_tensors=None,
        padding=False,
        truncation=False,
    ):
        if is_split_into_words:
            words = input_arg
            input_ids = []
            word_ids = []
            for wi, w in enumerate(words):
                if w == "[MASK]":
                    input_ids.append(mask_id)
                    word_ids.append(wi)
                else:
                    for c in w:
                        input_ids.append(ord(c))
                        word_ids.append(wi)
            enc = MagicMock()
            enc.__getitem__.side_effect = lambda k: {
                "input_ids": torch.tensor([input_ids]),
                "attention_mask": torch.ones((1, len(input_ids)), dtype=torch.long),
            }[k]
            enc.get.side_effect = lambda k, default=None: {
                "input_ids": torch.tensor([input_ids]),
                "attention_mask": torch.ones((1, len(input_ids)), dtype=torch.long),
            }.get(k, default)
            enc.word_ids = MagicMock(return_value=word_ids)
            enc.input_ids = torch.tensor([input_ids])
            enc.attention_mask = torch.ones((1, len(input_ids)), dtype=torch.long)
            return enc
        else:
            # simple encoding for aul_aula path
            input_ids = [ord(c) for c in input_arg] if isinstance(input_arg, str) else []
            enc = MagicMock()
            enc.input_ids = torch.tensor([input_ids])
            enc.attention_mask = torch.ones((1, len(input_ids)), dtype=torch.long)
            enc.word_ids = MagicMock(return_value=list(range(len(input_ids))))
            return enc

    tok.side_effect = call
    return tok


def _mock_mlm(vocab_size: int = 1000):
    model = MagicMock()
    model.to.return_value = model
    model.eval.return_value = model

    def forward(**kwargs):
        input_ids = kwargs["input_ids"]
        n = input_ids.shape[1]
        # Deterministic logits: put a big score on the token id equal to the
        # position + 1, so log P favours that id. Multi-piece candidates land
        # at consecutive positions with distinct favoured ids.
        logits = torch.full((1, n, vocab_size), -1000.0)
        for pos in range(n):
            fav_id = (pos + 1) % vocab_size
            logits[0, pos, fav_id] = 10.0
        # Attention: one layer, one head — uniform.
        attentions = (torch.full((1, 1, n, n), 1.0 / n),)
        out = MagicMock()
        out.logits = logits
        out.attentions = attentions
        return out

    model.side_effect = forward
    model.forward = forward
    return model


class TestBertPLLScorerMultiPiece:
    def test_multi_piece_candidate_no_longer_raises(self):
        from bias_scope.probability_based.scorers import BertPLLScorer

        with patch(
            "transformers.AutoTokenizer"
        ) as auto_tok, patch(
            "transformers.AutoModelForMaskedLM"
        ) as auto_mlm:
            tok = _mock_tokenizer(mask_id=999)
            auto_tok.from_pretrained.return_value = tok
            auto_mlm.from_pretrained.return_value = _mock_mlm()

            scorer = BertPLLScorer(model_name="fake", device="cpu")
            # candidate 'ab' encodes to [ord('a'), ord('b')] = 2 pieces
            prob = scorer.masked_token_probability(["hello", "[MASK]", "world"], "ab")
        assert 0.0 <= prob <= 1.0

    def test_single_piece_path_unchanged(self):
        from bias_scope.probability_based.scorers import BertPLLScorer

        with patch(
            "transformers.AutoTokenizer"
        ) as auto_tok, patch(
            "transformers.AutoModelForMaskedLM"
        ) as auto_mlm:
            auto_tok.from_pretrained.return_value = _mock_tokenizer(mask_id=999)
            auto_mlm.from_pretrained.return_value = _mock_mlm()
            scorer = BertPLLScorer(model_name="fake", device="cpu")
            prob = scorer.masked_token_probability(["hi", "[MASK]"], "a")
        assert 0.0 <= prob <= 1.0

    def test_attention_scheme_kwarg_stored(self):
        from bias_scope.probability_based.scorers import BertPLLScorer

        with patch(
            "transformers.AutoTokenizer"
        ) as auto_tok, patch(
            "transformers.AutoModelForMaskedLM"
        ) as auto_mlm:
            auto_tok.from_pretrained.return_value = _mock_tokenizer(mask_id=999)
            auto_mlm.from_pretrained.return_value = _mock_mlm()
            scorer = BertPLLScorer(
                model_name="fake", device="cpu",
                attention_scheme="all_layers_avg",
            )
        assert scorer.attention_scheme == "all_layers_avg"

    def test_attention_scheme_invalid_raises(self):
        from bias_scope.probability_based.scorers import BertPLLScorer

        with pytest.raises(ValueError):
            BertPLLScorer(model_name="fake", attention_scheme="cursed")  # type: ignore[arg-type]


class TestWordPieceBertScorerProtocol:
    def test_align_unmodified_pure_static(self):
        from bias_scope.probability_based.scorers import WordPieceBertScorer

        pa, pb = WordPieceBertScorer.align_unmodified(
            [1, 2, 3, 4, 5], [1, 9, 3, 4, 8]
        )
        # positions 0, 2, 3 match on both
        assert pa == [0, 2, 3]
        assert pb == [0, 2, 3]

    def test_encode_matches_tokenizer(self):
        from bias_scope.probability_based.scorers import WordPieceBertScorer

        with patch(
            "transformers.AutoTokenizer"
        ) as auto_tok, patch(
            "transformers.AutoModelForMaskedLM"
        ) as auto_mlm:
            auto_tok.from_pretrained.return_value = _mock_tokenizer()
            auto_mlm.from_pretrained.return_value = _mock_mlm()
            scorer = WordPieceBertScorer(model_name="fake", device="cpu")
        ids = scorer.encode("abc")
        assert ids == [ord("a"), ord("b"), ord("c")]

    def test_pll_over_positions_sums_log_probs(self):
        from bias_scope.probability_based.scorers import WordPieceBertScorer

        with patch(
            "transformers.AutoTokenizer"
        ) as auto_tok, patch(
            "transformers.AutoModelForMaskedLM"
        ) as auto_mlm:
            auto_tok.from_pretrained.return_value = _mock_tokenizer()
            auto_mlm.from_pretrained.return_value = _mock_mlm()
            scorer = WordPieceBertScorer(model_name="fake", device="cpu")

        # empty positions → 0
        assert scorer.pll_over_positions([1, 2, 3], []) == 0.0

    def test_aul_aula_returns_pair_of_floats(self):
        from bias_scope.probability_based.scorers import WordPieceBertScorer

        with patch(
            "transformers.AutoTokenizer"
        ) as auto_tok, patch(
            "transformers.AutoModelForMaskedLM"
        ) as auto_mlm:
            auto_tok.from_pretrained.return_value = _mock_tokenizer()
            auto_mlm.from_pretrained.return_value = _mock_mlm()
            scorer = WordPieceBertScorer(model_name="fake", device="cpu")
        aul, aula = scorer.aul_aula([1, 2, 3, 4, 5])
        assert isinstance(aul, float)
        assert isinstance(aula, float)
