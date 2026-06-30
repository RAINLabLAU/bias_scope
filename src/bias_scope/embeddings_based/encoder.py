"""Public embedding utilities for embedding-based bias metrics."""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from typing import Any

import numpy as np

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


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


def embed(
    texts: Sequence[str],
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    *,
    normalize_embeddings: bool = False,
    batch_size: int = 32,
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
) -> object:
    if _is_text_sequence(values):
        return embed(
            values,
            model_name=model_name,
            normalize_embeddings=normalize_embeddings,
            batch_size=batch_size,
        )
    return values


def _resolve_embedding_pair(
    embedding_pair: tuple[object, object],
    *,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    normalize_embeddings: bool = False,
    batch_size: int = 32,
) -> tuple[object, object]:
    return (
        _resolve_embeddings(
            embedding_pair[0],
            model_name=model_name,
            normalize_embeddings=normalize_embeddings,
            batch_size=batch_size,
        ),
        _resolve_embeddings(
            embedding_pair[1],
            model_name=model_name,
            normalize_embeddings=normalize_embeddings,
            batch_size=batch_size,
        ),
    )
