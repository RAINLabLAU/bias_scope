"""Utility functions for bias detection metrics."""

from typing import Union

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
