"""
Private helper classes and functions for generated text metrics.

These are internal implementation details and should NOT be imported by users.
The underscore prefix in the filename signals this is a private module.
"""

from typing import List
from bias_scope.base import GeneratedTextMetric
from bias_scope.generated_text.perspective_api import PerspectiveAPIClient


class ToxicityMetric(GeneratedTextMetric):
    """
    Base class for toxicity-based metrics (PRIVATE).
    
    Provides shared Perspective API integration for TF, TP, and EMT.
    All toxicity metrics inherit from this class to reuse the API client.
    """
    
    def __init__(self, api_key: str):
        """
        Initialize ToxicityMetric.
        
        Args:
            api_key (str): Google Cloud API key for Perspective API
        """
        self.perspective = PerspectiveAPIClient(api_key)
    
    def _score_texts(self, texts: List[str]) -> List[float]:
        """
        Get toxicity scores for a batch of texts (PRIVATE).
        
        Args:
            texts (List[str]): texts to score
            
        Returns:
            List[float]: toxicity scores
        """
        return self.perspective.score_batch(texts)
