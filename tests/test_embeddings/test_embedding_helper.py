"""Tests for public embedding helper convenience paths."""

import numpy as np
import pytest

import bias_scope.embeddings_based.encoder as encoder
from bias_scope.embeddings_based import CEAT, SEAT, WEAT, SentenceBiasScore, embed


class FakeSentenceTransformer:
    def __init__(self, model_name):
        self.model_name = model_name

    def encode(
        self,
        texts,
        convert_to_numpy=True,
        normalize_embeddings=False,
        batch_size=32,
    ):
        vectors = []
        for text in texts:
            lower = text.lower()
            if lower in {"man", "male", "he", "career", "office", "toxic"}:
                vec = np.array([1.0, 0.0])
            elif lower in {"woman", "female", "she", "family", "home", "safe"}:
                vec = np.array([0.0, 1.0])
            elif lower in {"science", "math"}:
                vec = np.array([0.9, 0.1])
            else:
                vec = np.array([0.1, 0.9])
            vectors.append(vec)
        result = np.array(vectors, dtype=float)
        if normalize_embeddings:
            result = result / np.linalg.norm(result, axis=1, keepdims=True)
        return result


@pytest.fixture
def fake_encoder(monkeypatch):
    """Fake both pooling paths.

    SEAT and CEAT default to `pooling="cls"` since 0.2.0 (the reference
    protocol), which goes through `_embed_cls` rather than
    sentence-transformers, so the fixture has to cover both or the tests
    silently stop exercising the default.
    """
    encoder._load_sentence_transformer.cache_clear()
    monkeypatch.setattr(
        encoder,
        "_load_sentence_transformer",
        lambda model_name: FakeSentenceTransformer(model_name),
    )

    def fake_embed_cls(texts, model_name, batch_size, normalize_embeddings):
        # Same deterministic vectors the fake sentence-transformer produces, so
        # a test can compare the two pooling modes if it wants to.
        return FakeSentenceTransformer(model_name).encode(
            list(texts), normalize_embeddings=normalize_embeddings
        )

    monkeypatch.setattr(encoder, "_embed_cls", fake_embed_cls)


def test_public_embed_helper(fake_encoder):
    vectors = embed(["man", "woman"], model_name="fake-model")

    assert isinstance(vectors, np.ndarray)
    assert vectors.shape == (2, 2)
    assert np.allclose(vectors[0], [1.0, 0.0])


def test_weat_accepts_text_inputs(fake_encoder):
    score = WEAT().evaluate(
        (["man", "career"], ["woman", "family"]),
        (["science", "math"], ["home", "safe"]),
        model_name="fake-model",
    )

    assert isinstance(score, float)
    assert score > 0


def test_seat_accepts_text_inputs(fake_encoder):
    score = SEAT().evaluate(
        (["man", "career"], ["woman", "family"]),
        (["science", "math"], ["home", "safe"]),
        model_name="fake-model",
    )

    assert isinstance(score, float)


def test_ceat_rejects_text_inputs(fake_encoder):
    with pytest.raises(ValueError, match="contextual token embeddings"):
        CEAT().evaluate(
            (["man", "career", "office"], ["woman", "family", "home"]),
            (["science", "math", "career"], ["home", "safe", "family"]),
            n_samples=2,
            model_name="fake-model",
        )


def test_sentence_bias_score_rejects_text_inputs(fake_encoder):
    with pytest.raises(TypeError, match="Raw text inputs are noncanonical"):
        SentenceBiasScore().evaluate(
            ["woman", "man"],
            np.array([1.0, 0.0]),
            np.array([0.5, 0.5]),
            np.array([False, False]),
        )


def test_sentence_bias_score_accepts_precomputed_embeddings(fake_encoder):
    female_bias, male_bias = SentenceBiasScore().evaluate(
        np.array([[0.0, 1.0], [1.0, 0.0]]),
        np.array([1.0, 0.0]),
        np.array([0.5, 0.5]),
        np.array([False, False]),
    )

    assert isinstance(female_bias, float)
    assert isinstance(male_bias, float)
