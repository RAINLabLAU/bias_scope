"""Sentence Encoder Association Test (SEAT)."""

from typing import Tuple
import numpy as np
import torch

from bias_scope.base import EmbeddingMetric
from bias_scope.embeddings.weat import WEAT


class SEAT(EmbeddingMetric):
    """
    Sentence Encoder Association Test.
    
    Adapts WEAT to contextualized sentence embeddings. Uses the same
    effect size calculation as WEAT, but operates on sentence-level
    representations instead of static word embeddings.
    
    SEAT generates sentence embeddings using templates and measures
    bias through the same differential association test as WEAT.
    
    Reference
    ---------
    May, C., Wang, A., Bordia, S., Bowman, S. R., & Rudinger, R. (2019).
    On measuring social biases in sentence encoders. NAACL-HLT 2019.
    
    Examples
    --------
    >>> from bias_scope.embeddings import SEAT
    >>> import numpy as np
    >>> 
    >>> # Test with BERT sentence embeddings (768-dimensional)
    >>> seat = SEAT()
    >>> 
    >>> # Sentences encoded with "This is [WORD]" template
    >>> male_sentences = np.random.randn(10, 768)
    >>> female_sentences = np.random.randn(10, 768)
    >>> career_sentences = np.random.randn(8, 768)
    >>> family_sentences = np.random.randn(8, 768)
    >>> 
    >>> score = seat.compute(
    ...     (male_sentences, female_sentences),
    ...     (career_sentences, family_sentences)
    ... )
    >>> print(f"Gender-career bias (SEAT): {score:.3f}")
    """
    
    @property
    def name(self) -> str:
        """Return metric name."""
        return "SEAT"
    
    @property
    def reference(self) -> str:
        """Return paper citation."""
        return (
            "May, C., Wang, A., Bordia, S., Bowman, S. R., & Rudinger, R. (2019). "
            "On measuring social biases in sentence encoders. NAACL-HLT 2019."
        )
    
    @property
    def complexity(self) -> str:
        """Return complexity rating."""
        return "easy"
    
    def compute(
        self,
        target_embeddings: Tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor],
        attribute_embeddings: Tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor],
    ) -> float:
        """
        Compute SEAT score.
        
        SEAT uses the identical calculation as WEAT, simply applied to
        sentence embeddings instead of word embeddings. This method
        delegates to WEAT internally.
        
        Parameters
        ----------
        target_embeddings : Tuple of 2 arrays
            (target_group1, target_group2) sentence embeddings.
            Each array shape: (n_sentences, embedding_dim)
            Example: Sentences with male vs female terms
        attribute_embeddings : Tuple of 2 arrays
            (attribute_group1, attribute_group2) sentence embeddings.
            Each array shape: (n_sentences, embedding_dim)
            Example: Sentences with career vs family words
        
        Returns
        -------
        float
            SEAT effect size (identical to WEAT calculation).
            Interpretation same as WEAT.
        
        Raises
        ------
        ValueError
            Same validation as WEAT (delegates to WEAT internally)
        
        Notes
        -----
        SEAT = WEAT applied to sentence embeddings from encoders like BERT.
        
        Typical workflow:
        1. Create sentences using templates (e.g., "This is [WORD]")
        2. Encode with sentence encoder (BERT, RoBERTa, etc.)
        3. Pass sentence embeddings to SEAT
        
        Examples
        --------
        >>> import numpy as np
        >>> from bias_scope.embeddings import SEAT
        >>> 
        >>> seat = SEAT()
        >>> 
        >>> # BERT [CLS] token embeddings (768-dim)
        >>> male_sent = np.random.randn(5, 768)
        >>> female_sent = np.random.randn(5, 768)
        >>> career_sent = np.random.randn(5, 768)
        >>> family_sent = np.random.randn(5, 768)
        >>> 
        >>> score = seat.compute(
        ...     (male_sent, female_sent),
        ...     (career_sent, family_sent)
        ... )
        """
        # SEAT is just WEAT applied to sentence embeddings
        weat_instance = WEAT()
        return weat_instance.compute(target_embeddings, attribute_embeddings)
