"""Interface tests for the pooling='cls' path in the embedding encoder.

Mocks `AutoModel` and `AutoTokenizer` so no BERT model is downloaded.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from bias_scope.embeddings_based import CEAT, SEAT, WEAT
from bias_scope.embeddings_based.encoder import embed


class _FakeEncoding(dict):
    def to(self, _device):
        return self


def _fake_encoder(hidden_size: int = 8):
    """Return (tokenizer_mock, model_mock) that produce deterministic CLS
    vectors keyed off the batch size."""
    import torch

    tokenizer = MagicMock()

    def tok_call(texts, padding=True, truncation=True, return_tensors="pt"):
        n = len(texts)
        return _FakeEncoding(
            input_ids=torch.zeros((n, 4), dtype=torch.long),
            attention_mask=torch.ones((n, 4), dtype=torch.long),
        )

    tokenizer.side_effect = tok_call

    model = MagicMock()
    model.parameters.return_value = iter([torch.zeros(1)])
    # `to()` returns self for chaining
    model.to.return_value = model
    model.eval.return_value = model

    def call(**kwargs):
        n = kwargs["input_ids"].shape[0]
        # CLS row is [i, i, ..., i] so different inputs get different vectors.
        cls = torch.arange(n).float().unsqueeze(-1).repeat(1, hidden_size)
        hidden = torch.zeros((n, 4, hidden_size))
        hidden[:, 0, :] = cls
        out = MagicMock()
        out.last_hidden_state = hidden
        return out

    model.side_effect = call
    return tokenizer, model


class TestEmbedClsPooling:
    def test_cls_pooling_returns_first_token_of_last_hidden_state(self):
        tok, mdl = _fake_encoder(hidden_size=4)
        # Clear the lru_cache so our patch takes effect for this call
        from bias_scope.embeddings_based.encoder import _load_cls_encoder

        _load_cls_encoder.cache_clear()
        with patch("transformers.AutoTokenizer") as auto_tok, patch(
            "transformers.AutoModel"
        ) as auto_mdl:
            auto_tok.from_pretrained.return_value = tok
            auto_mdl.from_pretrained.return_value = mdl
            out = embed(["a", "b", "c"], model_name="fake", pooling="cls")
        assert out.shape == (3, 4)
        # Row i should equal [i, i, i, i]
        assert np.allclose(out[0], 0.0)
        assert np.allclose(out[1], 1.0)
        assert np.allclose(out[2], 2.0)

    def test_invalid_pooling_raises(self):
        with pytest.raises(ValueError):
            embed(["a"], pooling="bogus")

    def test_pooling_mean_still_uses_sentence_transformers(self):
        # If pooling='mean' (default), the sentence-transformers path is used —
        # patch it and confirm the CLS loader is NOT touched.
        st_model = MagicMock()
        st_model.encode.return_value = np.zeros((2, 4))
        from bias_scope.embeddings_based.encoder import _load_sentence_transformer

        _load_sentence_transformer.cache_clear()
        with patch(
            "bias_scope.embeddings_based.encoder._load_sentence_transformer",
            return_value=st_model,
        ) as ld_st, patch(
            "bias_scope.embeddings_based.encoder._embed_cls"
        ) as embed_cls:
            out = embed(["x", "y"], pooling="mean")
            assert ld_st.called
            assert not embed_cls.called
        assert out.shape == (2, 4)


class TestSeatCeatPoolingPropagation:
    def test_seat_forwards_pooling_kwarg(self):
        # Patch WEAT.evaluate inside SEAT to observe the kwarg.
        with patch("bias_scope.embeddings_based.seat.WEAT") as WEAT_cls:
            weat_inst = MagicMock()
            weat_inst.evaluate.return_value = 0.42
            WEAT_cls.return_value = weat_inst
            s = SEAT(model_name="fake", pooling="cls")
            score = s.evaluate(
                (np.zeros((2, 4)), np.zeros((2, 4))),
                (np.zeros((2, 4)), np.zeros((2, 4))),
            )
        WEAT_cls.assert_called_once()
        assert WEAT_cls.call_args.kwargs["pooling"] == "cls"
        weat_inst.evaluate.assert_called_once()
        assert weat_inst.evaluate.call_args.kwargs["pooling"] == "cls"
        assert score == 0.42

    def test_seat_default_pooling_is_mean(self):
        with patch("bias_scope.embeddings_based.seat.WEAT") as WEAT_cls:
            weat_inst = MagicMock()
            weat_inst.evaluate.return_value = 0.0
            WEAT_cls.return_value = weat_inst
            SEAT().evaluate(
                (np.zeros((2, 4)), np.zeros((2, 4))),
                (np.zeros((2, 4)), np.zeros((2, 4))),
            )
        assert WEAT_cls.call_args.kwargs["pooling"] == "mean"

    def test_ceat_stores_pooling(self):
        c = CEAT(pooling="cls")
        assert c.pooling == "cls"
        assert CEAT().pooling == "mean"

    def test_weat_stores_pooling(self):
        w = WEAT(pooling="cls")
        assert w.pooling == "cls"
        assert WEAT().pooling == "mean"
