import numpy as np
import torch


def weat(
    target_embeddings1: torch.Tensor | np.ndarray,
    target_embeddings2: torch.Tensor | np.ndarray,
    attribute_embeddings1: torch.Tensor | np.ndarray,
    attribute_embeddings2: torch.Tensor | np.ndarray,
) -> float:
    """Computes the Word Embedding Association Test (WEAT) score.

    Args:
        target_embeddings1 (torch.Tensor | np.ndarray): target embeddings for the first target group
        target_embeddings2 (torch.Tensor | np.ndarray): target embeddings for the second target group
        attribute_embeddings1 (torch.Tensor | np.ndarray): attribute embeddings for the first attribute group
        attribute_embeddings2 (torch.Tensor | np.ndarray): attribute embeddings for the second attribute group

    Returns:
        float: weat score
    """
    pass

