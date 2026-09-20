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


class TestTheClsLoaderReusesTheBackendsModel:
    """RL-075: Qwen2.5-3B-Instruct ran out of GPU memory inside run_suite.
    The backend had the model loaded (bf16, ~6 GB) and WEAT/SEAT/CEAT each
    loaded their own copy through `_load_cls_encoder` / sentence-transformers,
    so a 3B model sat on the GPU three times. A backend that has loaded a
    model registers it, and the CLS-pooling loader reuses it instead of
    loading again.
    """

    def test_a_loaded_backend_model_is_what_the_cls_loader_returns(self):
        from bias_scope.backends import HuggingFaceBackend
        from bias_scope.embeddings_based.encoder import _load_cls_encoder

        _load_cls_encoder.cache_clear()
        backend = HuggingFaceBackend("sshleifer/tiny-gpt2", kind="causal")
        tokenizer, model = backend._load()
        shared_tok, shared_model = _load_cls_encoder("sshleifer/tiny-gpt2")
        assert shared_tok is tokenizer
        # A causal LM's base model is the transformer without the LM head; it
        # is the module whose `last_hidden_state` the pooling reads.
        assert shared_model is model.base_model

    def test_embedding_through_the_shared_model_works(self):
        from bias_scope.backends import HuggingFaceBackend
        from bias_scope.embeddings_based.encoder import _load_cls_encoder

        _load_cls_encoder.cache_clear()
        HuggingFaceBackend("sshleifer/tiny-gpt2", kind="causal")._load()
        out = embed(["a", "b c d"], model_name="sshleifer/tiny-gpt2", pooling="cls")
        assert out.shape[0] == 2 and np.all(np.isfinite(out))


class TestMeanPoolingReusesACausalBackendsModel:
    """RL-076: WEAT's default `pooling='mean'` went through sentence-transformers,
    a third copy of the model on the GPU, and on google/gemma-3-1b-it that
    loader tried to build an image processor and failed before any metric
    scored. For a repo with no sentence-transformers config, that library
    builds Transformer + Pooling(mean): a masked mean of the last hidden
    state, bit-identical to computing it on the backend's own copy (max abs
    difference 0.0 on gpt2 and tiny-gpt2, 2026-09-20). A causal backend that
    has loaded its model registers it for mean pooling too.
    """

    def test_mean_pooling_on_a_loaded_causal_backend_matches_sentence_transformers(self):
        from sentence_transformers import SentenceTransformer

        from bias_scope.backends import HuggingFaceBackend
        from bias_scope.embeddings_based import encoder

        texts = ["office", "the family went home", "a career in management"]
        reference_model = SentenceTransformer("sshleifer/tiny-gpt2", device="cpu")
        reference_model.tokenizer.pad_token = reference_model.tokenizer.eos_token
        reference = reference_model.encode(texts, convert_to_numpy=True)

        encoder._load_cls_encoder.cache_clear()
        # fp32 here so the comparison is about the pooling arithmetic. The
        # backend's production dtype for causal LMs is bf16 (PLAN.md Sec. 1);
        # sharing means the embedding metrics now run in the dtype the protocol
        # block records, instead of a separately loaded copy in the
        # checkpoint's dtype.
        HuggingFaceBackend("sshleifer/tiny-gpt2", kind="causal", dtype="fp32")._load()
        with patch.object(encoder, "_load_sentence_transformer",
                          side_effect=AssertionError("must not load a second copy")):
            out = embed(texts, model_name="sshleifer/tiny-gpt2", pooling="mean")
        assert out.shape == reference.shape
        assert np.allclose(out, reference, atol=1e-5)

    def test_an_encoder_backend_keeps_the_sentence_transformers_path(self):
        # Sentence-transformers checkpoints (all-MiniLM, all-mpnet) carry their
        # own pooling config; only a causal LM's registration covers mean pooling.
        from bias_scope.embeddings_based import encoder

        encoder.share_encoder("some/encoder", lambda: (None, None), mean_pooling=False)
        assert "some/encoder" not in encoder._SHARED_MEAN_POOL
        encoder.share_encoder("some/causal", lambda: (None, None), mean_pooling=True)
        assert "some/causal" in encoder._SHARED_MEAN_POOL

    def test_registration_happens_at_construction_and_loads_lazily(self):
        # RL-078: with every generation served from the cache the backend never
        # loaded, so a registration made inside _load never happened and WEAT
        # fell back to the sentence-transformers loader.
        from bias_scope.backends import HuggingFaceBackend
        from bias_scope.embeddings_based import encoder

        encoder._load_cls_encoder.cache_clear()
        backend = HuggingFaceBackend("sshleifer/tiny-gpt2", kind="causal", dtype="fp32")
        assert backend._model is None                      # nothing loaded yet
        _, shared = encoder._load_cls_encoder("sshleifer/tiny-gpt2")
        assert backend._model is not None                  # loaded on demand
        assert shared is backend._model.base_model
