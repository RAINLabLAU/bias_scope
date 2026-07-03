"""Public embedding utilities for embedding-based bias metrics."""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from typing import Any, Literal

import numpy as np

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

Pooling = Literal["mean", "cls"]


@lru_cache(maxsize=4)
def _load_sentence_transformer(model_name: str) -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "embed() requires sentence-transformers to be installed. "
            "Install the embedding dependencies or pass precomputed embeddings."
        ) from exc

    return SentenceTransformer(model_name)


@lru_cache(maxsize=4)
def _load_cls_encoder(model_name: str) -> tuple[Any, Any]:
    """Load AutoModel + AutoTokenizer for `[CLS]`-token pooling. Cached."""
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "pooling='cls' requires torch and transformers to be installed."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    return tokenizer, model


def _embed_cls(
    texts: list[str], model_name: str, batch_size: int, normalize_embeddings: bool
) -> np.ndarray:
    import torch

    tokenizer, model = _load_cls_encoder(model_name)
    device = next(model.parameters()).device
    out_chunks: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        enc = tokenizer(
            chunk, padding=True, truncation=True, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            hs = model(**enc).last_hidden_state
        cls_vecs = hs[:, 0, :].cpu().numpy()
        out_chunks.append(cls_vecs)
    embeddings = np.concatenate(out_chunks, axis=0)
    if normalize_embeddings:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        embeddings = embeddings / norms
    return np.asarray(embeddings, dtype=float)


def embed(
    texts: Sequence[str],
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    *,
    normalize_embeddings: bool = False,
    batch_size: int = 32,
    pooling: Pooling = "mean",
) -> np.ndarray:
    """
    Embed text strings with a SentenceTransformer model.

    Args:
        texts (Sequence[str]): Texts to embed.
        model_name (str): SentenceTransformer/Hugging Face model name.
        normalize_embeddings (bool): Whether to L2-normalize embeddings.
        batch_size (int): Encoding batch size.

    Returns:
        np.ndarray: Embedding matrix with shape (n_texts, embedding_dim).

    Raises:
        TypeError: If texts is not a sequence of strings.
        ValueError: If texts is empty or contains empty strings.
        ImportError: If sentence-transformers is not installed.
    """
    if not isinstance(texts, Sequence) or isinstance(texts, (str, bytes)):
        raise TypeError("texts must be a Sequence of strings")

    text_list = list(texts)
    if len(text_list) == 0:
        raise ValueError("texts cannot be empty")

    for i, text in enumerate(text_list):
        if not isinstance(text, str):
            raise TypeError(f"texts[{i}] must be a string, got {type(text).__name__}")
        if text == "":
            raise ValueError(f"texts[{i}] cannot be empty")

    if pooling not in ("mean", "cls"):
        raise ValueError(f"pooling must be 'mean' or 'cls', got {pooling!r}")

    if pooling == "cls":
        return _embed_cls(
            text_list,
            model_name=model_name,
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings,
        )

    model = _load_sentence_transformer(model_name)
    embeddings = model.encode(
        text_list,
        convert_to_numpy=True,
        normalize_embeddings=normalize_embeddings,
        batch_size=batch_size,
    )
    return np.asarray(embeddings, dtype=float)


def _is_text_sequence(value: object) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes, np.ndarray))
        and all(isinstance(item, str) for item in value)
    )


def _resolve_embeddings(
    values: object,
    *,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    normalize_embeddings: bool = False,
    batch_size: int = 32,
    pooling: Pooling = "mean",
) -> object:
    if _is_text_sequence(values):
        return embed(
            values,
            model_name=model_name,
            normalize_embeddings=normalize_embeddings,
            batch_size=batch_size,
            pooling=pooling,
        )
    return values


def _resolve_embedding_pair(
    embedding_pair: tuple[object, object],
    *,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    normalize_embeddings: bool = False,
    batch_size: int = 32,
    pooling: Pooling = "mean",
) -> tuple[object, object]:
    return (
        _resolve_embeddings(
            embedding_pair[0],
            model_name=model_name,
            normalize_embeddings=normalize_embeddings,
            batch_size=batch_size,
            pooling=pooling,
        ),
        _resolve_embeddings(
            embedding_pair[1],
            model_name=model_name,
            normalize_embeddings=normalize_embeddings,
            batch_size=batch_size,
            pooling=pooling,
        ),
    )
