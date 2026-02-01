from typing import Tuple

import numpy as np
import torch

from bias_scope.utils import cosine_similarity, to_numpy


def weat(
    target_word_embeddings: Tuple[torch.Tensor | np.ndarray, torch.Tensor | np.ndarray],
    attribute_word_embeddings: Tuple[
        torch.Tensor | np.ndarray, torch.Tensor | np.ndarray
    ],
) -> float:
    """
    Compute Word Embedding Association Test (WEAT) effect size.

    Parameters
    ----------
    target_word_embeddings : Tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor]
        Two sets of target word embeddings to compare
    attribute_word_embeddings : Tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor]
        Two sets of attribute word embeddings

    Returns
    -------
    float
        Effect size measuring differential association

    Raises
    ------
    ValueError
        If tuples don't contain exactly 2 elements each

    Notes
    -----
    Based on Caliskan et al. (2017). Effect size formula:
        d = (mean_X - mean_Y) / std(X ∪ Y)
    """

    if len(target_word_embeddings) != 2 or len(attribute_word_embeddings) != 2:
        raise ValueError(
            "The target_embeddings and attribute_embeddings must have two elements each."
        )

    target_word_embeddings1, target_word_embeddings2 = target_word_embeddings
    attribute_word_embeddings1, attribute_word_embeddings2 = attribute_word_embeddings

    target_word_embeddings1 = to_numpy(target_word_embeddings1)
    target_word_embeddings2 = to_numpy(target_word_embeddings2)
    attribute_word_embeddings1 = to_numpy(attribute_word_embeddings1)
    attribute_word_embeddings2 = to_numpy(attribute_word_embeddings2)

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
        __similarity_measure(w, attribute_word_embeddings1, attribute_word_embeddings2)
        for w in target_word_embeddings1
    ]
    cos_target2 = [
        __similarity_measure(w, attribute_word_embeddings1, attribute_word_embeddings2)
        for w in target_word_embeddings2
    ]

    union_target_embeddings = np.concatenate(
        [target_word_embeddings1, target_word_embeddings2]
    )

    cos_target_union = [
        __similarity_measure(w, attribute_word_embeddings1, attribute_word_embeddings2)
        for w in union_target_embeddings
    ]

    return float((np.mean(cos_target1) - np.mean(cos_target2)) / np.std(
        cos_target_union, ddof=1
    ))


def seat(
    target_sentence_embeddings: Tuple[torch.Tensor | np.ndarray, torch.Tensor | np.ndarray],
    attributes_sentence_embeddings: Tuple[torch.Tensor | np.ndarray, torch.Tensor | np.ndarray],
) -> float:
    """
    Compute Sentence Embedding Association Test (SEAT) score.

    Parameters
    ----------
    target_sentence_embeddings : Tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor]
        Two sets of target sentence embeddings
    attributes_sentence_embeddings : Tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor]
        Two sets of attribute sentence embeddings

    Returns
    -------
    float
        SEAT score (same calculation as WEAT)
    """

    return weat(target_sentence_embeddings, attributes_sentence_embeddings)


def sentence_bias(
    word_embeddings: np.ndarray | torch.Tensor,
    gender_direction: np.ndarray | torch.Tensor,
    word_importance: np.ndarray | torch.Tensor,
    gender_words_mask: np.ndarray | torch.Tensor = None,
) -> Tuple[float, float]:
    """
    Compute gender bias score for a sentence.
    
    Measures stereotypical gender associations by computing weighted cosine
    similarities between word embeddings and a gender direction vector.
    
    Parameters
    ----------
    word_embeddings : np.ndarray or torch.Tensor
        Word embedding vectors, shape (num_words, embedding_dim)
    gender_direction : np.ndarray or torch.Tensor
        Gender direction vector from PCA, shape (embedding_dim,)
        Positive values indicate feminine, negative indicate masculine
    word_importance : np.ndarray or torch.Tensor
        Semantic importance weight for each word, shape (num_words,)
        Typically derived from max-pooling in sentence encoder
    gender_words_mask : np.ndarray or torch.Tensor, optional
        Boolean mask indicating gendered words to exclude, shape (num_words,)
        True indicates word should be excluded (e.g., "she", "he", "mother")
    
    Returns
    -------
    Tuple[float, float]
        (female_bias, male_bias) where female_bias is sum of positive
        weighted similarities and male_bias is sum of negative weighted similarities
    
    Raises
    ------
    ValueError
        If array shapes are incompatible
    
    Notes
    -----
    Formula from Dolci et al. (2023):
        BiasScore_F = sum(cos(word, g) * importance) for cos > 0
        BiasScore_M = sum(cos(word, g) * importance) for cos < 0
    where g is the gender direction and gendered words are excluded.
    """
    # Convert to numpy
    word_embeddings = to_numpy(word_embeddings)
    gender_direction = to_numpy(gender_direction)
    word_importance = to_numpy(word_importance)
    
    # Input validation
    num_words = word_embeddings.shape[0]
    embedding_dim = word_embeddings.shape[1]
    
    if gender_direction.shape[0] != embedding_dim:
        raise ValueError(
            f"Gender direction dimension {gender_direction.shape[0]} "
            f"does not match embedding dimension {embedding_dim}"
        )
    
    if word_importance.shape[0] != num_words:
        raise ValueError(
            f"Importance array length {word_importance.shape[0]} "
            f"does not match number of words {num_words}"
        )
    
    # Normalize gender direction
    gender_direction = gender_direction / np.linalg.norm(gender_direction)
    
    # Compute cosine similarity for each word
    word_biases = np.array([
        cosine_similarity(word_emb, gender_direction) 
        for word_emb in word_embeddings
    ])
    
    # Exclude gendered words if mask provided
    if gender_words_mask is not None:
        gender_words_mask = to_numpy(gender_words_mask)
        if gender_words_mask.shape[0] != num_words:
            raise ValueError(
                f"Mask length {gender_words_mask.shape[0]} "
                f"does not match number of words {num_words}"
            )
        word_biases = word_biases * (~gender_words_mask)
    
    # Weight by importance
    weighted_biases = word_biases * word_importance
    
    # Separate female and male bias
    female_bias = float(np.sum(weighted_biases[weighted_biases > 0]))
    male_bias = float(np.sum(weighted_biases[weighted_biases < 0]))
    
    return female_bias, male_bias
