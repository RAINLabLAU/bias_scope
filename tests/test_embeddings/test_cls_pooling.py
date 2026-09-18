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
            # SEAT delegates with return_details=True (see the note below).
            weat_inst.evaluate.return_value = {
                "weat_score": 0.42, "effect_size": 0.42,
                "n_target_group_1": 2.0, "n_target_group_2": 2.0,
            }
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

    def test_seat_default_pooling_is_cls(self):
        """Changed in 0.2.0 to match the reference protocol.

        May et al.'s `sent-bias/sentbias/encoders/bert.py:26` takes
        `enc[:, 0, :]` — the position-0 representation — so `cls` is the
        reference's pooling, not `mean`. PLAN.md 4.2 asks for this default.
        This test previously asserted `mean`.
        """
        with patch("bias_scope.embeddings_based.seat.WEAT") as WEAT_cls:
            weat_inst = MagicMock()
            # SEAT now delegates with return_details=True so it can carry
            # WEAT's group sizes and p-value through; without them `run()`
            # cannot build an interval. The mock must match that call.
            weat_inst.evaluate.return_value = {
                "weat_score": 0.0,
                "effect_size": 0.0,
                "p_value": 1.0,
                "p_value_exact": True,
                "num_partitions": 1,
                "p_value_note": "",
                "n_target_group_1": 2.0,
                "n_target_group_2": 2.0,
                "n_attribute_group_1": 2.0,
                "n_attribute_group_2": 2.0,
            }
            WEAT_cls.return_value = weat_inst
            SEAT().evaluate(
                (np.zeros((2, 4)), np.zeros((2, 4))),
                (np.zeros((2, 4)), np.zeros((2, 4))),
            )
        assert WEAT_cls.call_args.kwargs["pooling"] == "cls"

    def test_ceat_stores_pooling(self):
        c = CEAT(pooling="cls")
        assert c.pooling == "cls"
        assert CEAT().pooling == "cls"

    def test_weat_stores_pooling(self):
        w = WEAT(pooling="cls")
        assert w.pooling == "cls"
        assert WEAT().pooling == "mean"


class TestEmbedClsBf16:
    """A bf16 model's hidden states are BFloat16, which numpy cannot represent.

    Found by a live agent run (REVIEW_LATER RL-056): SEAT on a bf16 causal LM
    died with `TypeError: Got unsupported ScalarType BFloat16` from
    `.cpu().numpy()`. PLAN.md Section 1 mandates BF16 for causal LMs, so every
    embedding metric was unreachable on exactly the dtype the plan requires.
    """

    def test_bf16_hidden_states_are_cast_before_numpy(self):
        import torch

        tok, mdl = _fake_encoder(hidden_size=4)

        def call_bf16(**kwargs):
            n = kwargs["input_ids"].shape[0]
            hidden = torch.zeros((n, 4, 4), dtype=torch.bfloat16)
            hidden[:, 0, :] = torch.arange(n).to(torch.bfloat16).unsqueeze(-1).repeat(1, 4)
            out = MagicMock()
            out.last_hidden_state = hidden
            return out

        mdl.side_effect = call_bf16
        from bias_scope.embeddings_based.encoder import _load_cls_encoder

        _load_cls_encoder.cache_clear()
        with patch("transformers.AutoTokenizer") as auto_tok, patch(
            "transformers.AutoModel"
        ) as auto_mdl:
            auto_tok.from_pretrained.return_value = tok
            auto_mdl.from_pretrained.return_value = mdl
            out = embed(["a", "b"], model_name="fake", pooling="cls")
        assert out.dtype == np.float64
        assert np.allclose(out[0], 0.0)
        assert np.allclose(out[1], 1.0)


class TestATokenizerWithoutAPadToken:
    """RL-067: GPT-2's tokenizer has no pad token, so batching two texts of
    different length through `pooling='cls'` raised "Asking to pad but the
    tokenizer does not have a padding token" - and WEAT and SEAT, both
    recommended for every causal LM, were skipped on gpt2 in a live run.
    `HuggingFaceBackend.generate` already makes the same choice this test
    asks for: pad with the end-of-sequence token.
    """

    def test_gpt2_style_tokenizer_can_batch_texts_of_different_length(self):
        from bias_scope.embeddings_based.encoder import _load_cls_encoder

        _load_cls_encoder.cache_clear()
        out = embed(["a", "b c d e"], model_name="sshleifer/tiny-gpt2", pooling="cls")
        assert out.shape[0] == 2
        assert np.all(np.isfinite(out))

    def test_mean_pooling_has_the_same_fix(self):
        # WEAT's default is pooling='mean' through sentence-transformers, which
        # wraps the same tokenizer; the gpt2 rerun scored SEAT (cls) and still
        # skipped WEAT (mean) with the identical error.
        from bias_scope.embeddings_based.encoder import _load_sentence_transformer

        _load_sentence_transformer.cache_clear()
        out = embed(["a", "b c d e"], model_name="sshleifer/tiny-gpt2", pooling="mean")
        assert out.shape[0] == 2
        assert np.all(np.isfinite(out))
