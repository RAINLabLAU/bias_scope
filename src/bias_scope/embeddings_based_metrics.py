import numpy as np
import torch

from bias_scope.utils import cosine_similarity


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

    if isinstance(target_embeddings1, torch.Tensor):
        target_embeddings1 = target_embeddings1.cpu().numpy()
    if isinstance(target_embeddings2, torch.Tensor):
        target_embeddings2 = target_embeddings2.cpu().numpy()
    if isinstance(attribute_embeddings1, torch.Tensor):
        attribute_embeddings1 = attribute_embeddings1.cpu().numpy()
    if isinstance(attribute_embeddings2, torch.Tensor):
        attribute_embeddings2 = attribute_embeddings2.cpu().numpy()

    def __similarity_measure(
        w: np.ndarray, a_embeddings1: np.ndarray, a_embeddings2: np.ndarray
    ) -> float:
        """
        Compute the similarity measure between the target word and the attribute words.

        Args:
            w (np.ndarray): target word embedding
            a_embeddings1 (np.ndarray): attribute embeddings for the first attribute group
            a_embeddings2 (np.ndarray): attribute embeddings for the second attribute group

        Returns:
            float: similarity measure between the target word and the attribute words
        """
        cos_w1 = [cosine_similarity(w, a) for a in a_embeddings1]
        cos_w2 = [cosine_similarity(w, a) for a in a_embeddings2]

        return np.mean(cos_w1) - np.mean(cos_w2)

    cos_target1 = [
        __similarity_measure(w, attribute_embeddings1, attribute_embeddings2)
        for w in target_embeddings1
    ]
    cos_target2 = [
        __similarity_measure(w, attribute_embeddings1, attribute_embeddings2)
        for w in target_embeddings2
    ]

    union_attribute_embeddings = np.union1d(
        (target_embeddings1.flatten(), target_embeddings2.flatten())
    )
    union_attribute_embeddings = union_attribute_embeddings(
        -1, target_embeddings1.shape[1]
    )

    cos_target_union = [
        __similarity_measure(w, attribute_embeddings1, attribute_embeddings2)
        for w in union_attribute_embeddings
    ]

    return np.mean(cos_target1) - np.mean(cos_target2) / np.std(cos_target_union)
