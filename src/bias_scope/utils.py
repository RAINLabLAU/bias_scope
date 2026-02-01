import numpy as np
import torch


def cosine_similarity(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.

    Args:
        x (np.ndarray): first vector
        y (np.ndarray): second vector

    Returns:
        float: cosine similarity between x and y
    """
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


def to_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    """
    Convert torch.Tensor to np.ndarray if necessary.

    Args:
        x (torch.Tensor | np.ndarray): input tensor

    Returns:
        np.ndarray: converted tensor
    """
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    if not isinstance(x, np.ndarray):
        return np.array(x)
    return x
