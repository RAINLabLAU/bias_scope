import numpy as np


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
