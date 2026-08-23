"""Utility functions for bias detection metrics."""

import hashlib
import json
import os
import random
from typing import Any, Dict, Union

import numpy as np

try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    torch = None


def to_numpy(arr: Union[np.ndarray, list]) -> np.ndarray:
    """
    Convert input to numpy array.

    Handles PyTorch tensors, lists, and numpy arrays.

    Parameters
    ----------
    arr : array-like
        Input array (numpy array, PyTorch tensor, or list)

    Returns
    -------
    np.ndarray
        Numpy array

    Examples
    --------
    >>> import torch
    >>> tensor = torch.randn(3, 5)
    >>> arr = to_numpy(tensor)
    >>> isinstance(arr, np.ndarray)
    True
    """
    if _TORCH_AVAILABLE and isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()
    elif isinstance(arr, list):
        return np.array(arr)
    elif isinstance(arr, np.ndarray):
        return arr
    else:
        # Try to convert to numpy
        return np.array(arr)


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.

    Formula: cos(θ) = (A · B) / (||A|| × ||B||)

    Parameters
    ----------
    vec1 : np.ndarray
        First vector
    vec2 : np.ndarray
        Second vector

    Returns
    -------
    float
        Cosine similarity in range [-1, 1]
        1 = identical direction
        0 = orthogonal
        -1 = opposite direction

    Examples
    --------
    >>> vec1 = np.array([1.0, 0.0, 0.0])
    >>> vec2 = np.array([1.0, 0.0, 0.0])
    >>> cosine_similarity(vec1, vec2)
    1.0

    >>> vec1 = np.array([1.0, 0.0])
    >>> vec2 = np.array([0.0, 1.0])
    >>> cosine_similarity(vec1, vec2)
    0.0
    """
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    # Avoid division by zero
    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(dot_product / (norm1 * norm2))


def seed_everything(seed: int = 42) -> int:
    """
    Seed every random source the library can reach.

    Sets ``PYTHONHASHSEED``, :mod:`random`, :mod:`numpy`, and — when torch is
    installed — ``torch.manual_seed`` and ``torch.cuda.manual_seed_all``.
    Call this once at the start of a run; every result written to ``results/``
    records the seed used in its ``protocol`` block.

    Parameters
    ----------
    seed : int
        Seed value. Default 42, the library-wide default.

    Returns
    -------
    int
        The seed that was applied, so callers can record it.

    Examples
    --------
    >>> seed_everything(0)
    0
    >>> import random
    >>> a = random.random()
    >>> seed_everything(0)
    0
    >>> a == random.random()
    True
    """
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError(f"seed must be an int, got {type(seed).__name__}")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    if _TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    return seed


def protocol_hash(protocol: Dict[str, Any]) -> str:
    """
    Stable short hash of a protocol dict.

    Formula: first 12 hex characters of the SHA-256 of the protocol serialised
    as canonical JSON (keys sorted, no insignificant whitespace). Two protocols
    that differ only in key order hash the same; any difference in a value
    changes the hash.

    Parameters
    ----------
    protocol : dict
        JSON-serialisable protocol block. Values that are not natively
        JSON-serialisable are rendered with ``str``.

    Returns
    -------
    str
        12 lowercase hex characters.

    Examples
    --------
    >>> protocol_hash({"a": 1, "b": 2}) == protocol_hash({"b": 2, "a": 1})
    True
    >>> len(protocol_hash({"a": 1}))
    12
    """
    if not isinstance(protocol, dict):
        raise ValueError(
            f"protocol must be a dict, got {type(protocol).__name__}"
        )

    canonical = json.dumps(
        protocol, sort_keys=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]
