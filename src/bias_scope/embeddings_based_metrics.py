from typing import Tuple

import numpy as np
import torch

from bias_scope.utils import cosine_similarity, to_numpy


def weat(
    target_embeddings: Tuple[torch.Tensor | np.ndarray, torch.Tensor | np.ndarray],
    attribute_embeddings: Tuple[torch.Tensor | np.ndarray, torch.Tensor | np.ndarray],
) -> float:
    """
    Computes the Word Embedding Association Test (WEAT) score.

    Args:
        target_embeddings (Tuple[torch.Tensor | np.ndarray, torch.Tensor | np.ndarray]): target word embeddings
        attribute_embeddings (Tuple[torch.Tensor | np.ndarray, torch.Tensor | np.ndarray]): attribute word embeddings

    Returns:
        float: weat score
    """

    if len(target_embeddings) != 2 or len(attribute_embeddings) != 2:
        raise ValueError(
            "The target_embeddings and attribute_embeddings must have two elements each."
        )

    target_embeddings1, target_embeddings2 = target_embeddings
    attribute_embeddings1, attribute_embeddings2 = attribute_embeddings

    target_embeddings1 = to_numpy(target_embeddings1)
    target_embeddings2 = to_numpy(target_embeddings2)
    attribute_embeddings1 = to_numpy(attribute_embeddings1)
    attribute_embeddings2 = to_numpy(attribute_embeddings2)

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
