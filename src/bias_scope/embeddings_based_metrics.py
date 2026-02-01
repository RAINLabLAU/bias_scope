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

    Examples
    --------
    Basic usage with word embeddings:

    >>> import numpy as np
    >>> # Male/Female names vs Career/Family words
    >>> male_names = np.random.randn(5, 300)  # 300-dim embeddings
    >>> female_names = np.random.randn(5, 300)
    >>> career_words = np.random.randn(5, 300)
    >>> family_words = np.random.randn(5, 300)
    >>> 
    >>> effect_size = weat(
    ...     (male_names, female_names),
    ...     (career_words, family_words)
    ... )
    >>> print(f"Gender-career bias effect size: {effect_size:.3f}")
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

    # Check for empty arrays
    if len(target_word_embeddings1) == 0 or len(target_word_embeddings2) == 0:
        raise ValueError(
            "Target embeddings cannot be empty. "
            f"Got {len(target_word_embeddings1)} and {len(target_word_embeddings2)} embeddings."
        )

    if len(attribute_word_embeddings1) == 0 or len(attribute_word_embeddings2) == 0:
        raise ValueError(
            "Attribute embeddings cannot be empty. "
            f"Got {len(attribute_word_embeddings1)} and {len(attribute_word_embeddings2)} embeddings."
        )

    # Check dimension consistency
    embedding_dims = [
        target_word_embeddings1.shape[1],
        target_word_embeddings2.shape[1],
        attribute_word_embeddings1.shape[1],
        attribute_word_embeddings2.shape[1]
    ]

    if len(set(embedding_dims)) != 1:
        raise ValueError(
            f"All embeddings must have the same dimensionality. "
            f"Got dimensions: target1={embedding_dims[0]}, target2={embedding_dims[1]}, "
            f"attr1={embedding_dims[2]}, attr2={embedding_dims[3]}"
        )

    # Check for NaN/Inf
    for name, arr in [
        ("target_word_embeddings1", target_word_embeddings1),
        ("target_word_embeddings2", target_word_embeddings2),
        ("attribute_word_embeddings1", attribute_word_embeddings1),
        ("attribute_word_embeddings2", attribute_word_embeddings2)
    ]:
        if np.isnan(arr).any():
            raise ValueError(f"{name} contains NaN values")
        if np.isinf(arr).any():
            raise ValueError(f"{name} contains Inf values")

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

    std_union = np.std(cos_target_union, ddof=1)

    # Handle zero/near-zero standard deviation
    if std_union < 1e-10:  # Numerical threshold
        raise ValueError(
            "Standard deviation of association scores is zero or near-zero. "
            "This occurs when all target embeddings have identical associations "
            "with the attribute embeddings. Cannot compute effect size."
        )

    # Handle single element case (ddof=1 makes std undefined)
    if len(cos_target_union) < 2:
        raise ValueError(
            "Need at least 2 total target embeddings to compute effect size. "
            f"Got {len(cos_target_union)} embeddings total."
        )

    return float((np.mean(cos_target1) - np.mean(cos_target2)) / std_union)


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

    Examples
    --------
    Compute bias for a sentence:

    >>> import numpy as np
    >>> # 5 words, 300-dimensional embeddings
    >>> words = np.random.randn(5, 300)
    >>> gender_dir = np.random.randn(300)  # From PCA
    >>> importance = np.array([0.15, 0.25, 0.20, 0.30, 0.10])
    >>> 
    >>> # Mark first word as gendered (e.g., "she")
    >>> mask = np.array([True, False, False, False, False])
    >>> 
    >>> female_bias, male_bias = sentence_bias(
    ...     words, gender_dir, importance, mask
    ... )
    >>> print(f"Female bias: {female_bias:.4f}")
    >>> print(f"Male bias: {male_bias:.4f}")
    """
    # Convert to numpy
    word_embeddings = to_numpy(word_embeddings)
    gender_direction = to_numpy(gender_direction)
    word_importance = to_numpy(word_importance)

    # Validate no NaN/Inf values
    if np.isnan(word_embeddings).any():
        raise ValueError("word_embeddings contains NaN values")
    if np.isinf(word_embeddings).any():
        raise ValueError("word_embeddings contains Inf values")

    if np.isnan(gender_direction).any():
        raise ValueError("gender_direction contains NaN values")
    if np.isinf(gender_direction).any():
        raise ValueError("gender_direction contains Inf values")

    if np.isnan(word_importance).any():
        raise ValueError("word_importance contains NaN values")
    if np.isinf(word_importance).any():
        raise ValueError("word_importance contains Inf values")
    
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

    # Validate importance values are non-negative
    if (word_importance < 0).any():
        raise ValueError(
            "Word importance values must be non-negative. "
            f"Found {np.sum(word_importance < 0)} negative values. "
            f"Min value: {np.min(word_importance)}"
        )
    
    # Normalize gender direction
    gender_norm = np.linalg.norm(gender_direction)

    if gender_norm < 1e-10:  # Numerical threshold
        raise ValueError(
            "Gender direction vector has zero or near-zero magnitude. "
            "Cannot normalize. Please provide a non-zero gender direction vector."
        )

    gender_direction = gender_direction / gender_norm
    
    # Compute cosine similarity for each word
    word_biases = np.array([
        cosine_similarity(word_emb, gender_direction) 
        for word_emb in word_embeddings
    ])
    
    # Exclude gendered words if mask provided
    if gender_words_mask is not None:
        gender_words_mask = to_numpy(gender_words_mask)
        
        # Validate mask length
        if gender_words_mask.shape[0] != num_words:
            raise ValueError(
                f"Mask length {gender_words_mask.shape[0]} "
                f"does not match number of words {num_words}"
            )
        
        # Validate mask is boolean type
        if gender_words_mask.dtype != bool:
            raise TypeError(
                f"gender_words_mask must be boolean array. "
                f"Got dtype: {gender_words_mask.dtype}. "
                f"Convert using .astype(bool) if needed."
            )
        
        # Validate not all masked
        if gender_words_mask.all():
            # This is valid - return zero bias
            return 0.0, 0.0

        word_biases = word_biases * (~gender_words_mask)
    
    # Weight by importance
    weighted_biases = word_biases * word_importance
    
    # Separate female and male bias
    female_bias = float(np.sum(weighted_biases[weighted_biases > 0]))
    male_bias = float(np.sum(weighted_biases[weighted_biases < 0]))
    
    return female_bias, male_bias
