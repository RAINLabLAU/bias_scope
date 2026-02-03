"""Contextualized Embedding Association Test (CEAT)."""

from typing import Tuple, Optional, Dict
import numpy as np
import torch

from bias_scope.base import EmbeddingMetric
from bias_scope.utils import to_numpy
from bias_scope.embeddings.weat import WEAT
from bias_scope.embeddings._helpers import (
    _validate_tuple_length,
    _validate_embedding_dimensions,
    _compute_random_effects_weights
)


class CEAT(EmbeddingMetric):
    """
    Contextualized Embedding Association Test.
    
    Extends WEAT to contextualized embeddings by calculating a distribution
    of WEAT effect sizes across randomly sampled subsets, then aggregating
    them using a random-effects model. This captures both the mean bias and
    how bias varies across different contexts.
    
    Unlike SEAT which averages over contexts, CEAT measures the full
    distribution of bias, revealing context-dependent associations.
    
    Reference
    ---------
    Guo, W., & Caliskan, A. (2021). Detecting Emergent Intersectional Biases:
    Contextualized Word Embeddings Contain a Distribution of Human-like Biases.
    AIES '21, pp. 122-133. https://doi.org/10.1145/3461702.3462536
    
    Examples
    --------
    >>> from bias_scope.embeddings import CEAT
    >>> import numpy as np
    >>> 
    >>> # Test with BERT contextualized embeddings
    >>> ceat = CEAT()
    >>> 
    >>> # Many contextualized embeddings (different sentences)
    >>> male_embeddings = np.random.randn(50, 768)
    >>> female_embeddings = np.random.randn(50, 768)
    >>> career_embeddings = np.random.randn(40, 768)
    >>> family_embeddings = np.random.randn(40, 768)
    >>> 
    >>> result = ceat.compute(
    ...     (male_embeddings, female_embeddings),
    ...     (career_embeddings, family_embeddings),
    ...     n_samples=100,
    ...     random_seed=42
    ... )
    >>> 
    >>> print(f"CEAT score: {result['ceat_score']:.3f}")
    >>> print(f"Mean WEAT: {result['weat_mean']:.3f}")
    >>> print(f"WEAT variance: {result['weat_variance']:.3f}")
    """
    
    @property
    def name(self) -> str:
        """Return metric name."""
        return "CEAT"
    
    @property
    def reference(self) -> str:
        """Return paper citation."""
        return (
            "Guo, W., & Caliskan, A. (2021). Detecting Emergent Intersectional "
            "Biases: Contextualized Word Embeddings Contain a Distribution of "
            "Human-like Biases. AIES '21, pp. 122-133."
        )
    
    @property
    def complexity(self) -> str:
        """Return complexity rating."""
        return "hard"
    
    def compute(
        self,
        target_embeddings: Tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor],
        attribute_embeddings: Tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor],
        n_samples: int = 100,
        sample_size: Optional[int] = None,
        random_seed: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Compute CEAT score with distribution of WEAT effect sizes.
        
        Parameters
        ----------
        target_embeddings : Tuple of 2 arrays
            (target_group1, target_group2) contextualized embeddings.
            Each array shape: (n_contexts, embedding_dim)
            Need sufficient contexts for sampling (recommended: 30+)
        attribute_embeddings : Tuple of 2 arrays
            (attribute_group1, attribute_group2) contextualized embeddings.
            Each array shape: (n_contexts, embedding_dim)
            Need sufficient contexts for sampling (recommended: 30+)
        n_samples : int, default=100
            Number of random samples to draw for distribution.
            More samples = more stable results but slower.
            Recommended: 50-200
        sample_size : int, optional
            Number of embeddings per group in each sample.
            If None, uses min(10, smallest_group_size)
        random_seed : int, optional
            Random seed for reproducibility. If provided, results will
            be deterministic.
        
        Returns
        -------
        Dict[str, float]
            Dictionary containing:
            - 'ceat_score': Weighted average of WEAT scores (main metric)
            - 'weat_mean': Simple mean of WEAT scores
            - 'weat_std': Standard deviation of WEAT scores
            - 'weat_variance': Variance of WEAT scores
            - 'n_samples': Number of samples actually used
        
        Raises
        ------
        ValueError
            If inputs are invalid:
            - Tuples don't have exactly 2 elements
            - Any array is empty
            - Arrays have mismatched dimensions
            - Insufficient data for sampling
            - n_samples <= 0
            - Arrays contain NaN or Inf values
        
        Notes
        -----
        CEAT formula:
            CEAT = (Σ vᵢ × WEATᵢ) / (Σ vᵢ)
        
        Where:
            - WEATᵢ = WEAT score for sample i
            - vᵢ = inverse-variance weight from random-effects model
            - N = n_samples
        
        Random-effects model accounts for heterogeneity across contexts,
        giving more weight to samples with lower variance.
        
        Examples
        --------
        >>> import numpy as np
        >>> from bias_scope.embeddings import CEAT
        >>> 
        >>> ceat = CEAT()
        >>> 
        >>> # Generate contextualized embeddings (50 contexts each)
        >>> male = np.random.randn(50, 768)
        >>> female = np.random.randn(50, 768)
        >>> career = np.random.randn(40, 768)
        >>> family = np.random.randn(40, 768)
        >>> 
        >>> # Compute with 100 random samples
        >>> result = ceat.compute(
        ...     (male, female),
        ...     (career, family),
        ...     n_samples=100,
        ...     sample_size=10,
        ...     random_seed=42
        ... )
        >>> 
        >>> # Main bias score
        >>> print(f"CEAT: {result['ceat_score']:.3f}")
        >>> 
        >>> # Variance shows context-dependency
        >>> print(f"Variance: {result['weat_variance']:.3f}")
        """
        # Validate inputs
        _validate_tuple_length(target_embeddings, "target_embeddings")
        _validate_tuple_length(attribute_embeddings, "attribute_embeddings")
        
        if n_samples <= 0:
            raise ValueError(f"n_samples must be positive. Got {n_samples}")
        
        # Unpack and convert to numpy
        target1, target2 = target_embeddings
        attr1, attr2 = attribute_embeddings
        
        target1 = to_numpy(target1)
        target2 = to_numpy(target2)
        attr1 = to_numpy(attr1)
        attr2 = to_numpy(attr2)
        
        # Validate embeddings
        self._validate_embeddings(target1, "target_embeddings[0]")
        self._validate_embeddings(target2, "target_embeddings[1]")
        self._validate_embeddings(attr1, "attribute_embeddings[0]")
        self._validate_embeddings(attr2, "attribute_embeddings[1]")
        
        # Validate dimensions
        _validate_embedding_dimensions([target1, target2, attr1, attr2])
        
        # Determine sample size
        if sample_size is None:
            min_group_size = min(len(target1), len(target2), len(attr1), len(attr2))
            sample_size = min(10, min_group_size)
        
        # Validate sufficient data for sampling
        self._validate_sufficient_data(
            [target1, target2, attr1, attr2],
            sample_size
        )
        
        # Set random seed if provided
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Compute WEAT scores for random samples
        weat_scores = self._compute_weat_distribution(
            target1, target2, attr1, attr2,
            n_samples, sample_size
        )
        
        # Compute random-effects weights
        weights = _compute_random_effects_weights(weat_scores)
        
        # Compute final CEAT score (weighted average)
        ceat_score = float(np.sum(weights * np.array(weat_scores)))
        
        # Return comprehensive results
        return {
            'ceat_score': ceat_score,
            'weat_mean': float(np.mean(weat_scores)),
            'weat_std': float(np.std(weat_scores, ddof=1)),
            'weat_variance': float(np.var(weat_scores, ddof=1)),
            'n_samples': n_samples
        }
    
    def _validate_sufficient_data(
        self,
        arrays: list,
        sample_size: int
    ) -> None:
        """
        Validate sufficient data for sampling (PRIVATE).
        
        Parameters
        ----------
        arrays : list of np.ndarray
            All embedding arrays
        sample_size : int
            Required sample size
            
        Raises
        ------
        ValueError
            If any array has fewer embeddings than sample_size
        """
        names = [
            "target_embeddings[0]",
            "target_embeddings[1]",
            "attribute_embeddings[0]",
            "attribute_embeddings[1]"
        ]
        
        for arr, name in zip(arrays, names):
            if len(arr) < sample_size:
                raise ValueError(
                    f"{name} has only {len(arr)} embeddings but "
                    f"sample_size={sample_size}. Need at least {sample_size} "
                    f"embeddings per group for sampling."
                )
    
    def _compute_weat_distribution(
        self,
        target1: np.ndarray,
        target2: np.ndarray,
        attr1: np.ndarray,
        attr2: np.ndarray,
        n_samples: int,
        sample_size: int
    ) -> list:
        """
        Compute distribution of WEAT scores via random sampling (PRIVATE).
        
        Parameters
        ----------
        target1, target2 : np.ndarray
            Target group embeddings
        attr1, attr2 : np.ndarray
            Attribute group embeddings
        n_samples : int
            Number of random samples
        sample_size : int
            Size of each sample
        
        Returns
        -------
        list of float
            WEAT scores for each random sample
        """
        weat = WEAT()
        weat_scores = []
        
        for i in range(n_samples):
            # Random sample without replacement from each group
            t1_sample = target1[np.random.choice(len(target1), sample_size, replace=False)]
            t2_sample = target2[np.random.choice(len(target2), sample_size, replace=False)]
            a1_sample = attr1[np.random.choice(len(attr1), sample_size, replace=False)]
            a2_sample = attr2[np.random.choice(len(attr2), sample_size, replace=False)]
            
            # Compute WEAT for this sample
            weat_score = weat.compute(
                (t1_sample, t2_sample),
                (a1_sample, a2_sample)
            )
            weat_scores.append(weat_score)
        
        return weat_scores
