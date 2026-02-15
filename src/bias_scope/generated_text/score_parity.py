"""Score Parity - Generalized group comparison metric."""

from typing import List, Dict, Callable
import numpy as np

from bias_scope.base import GeneratedTextMetric


class ScoreParity(GeneratedTextMetric):
    """
    Score Parity.
    
    A flexible metric for comparing any classifier scores across
    demographic groups in generated text. Unlike specific metrics
    (toxicity, sentiment), Score Parity accepts any scoring function,
    making it adaptable to various bias detection scenarios.
    
    Score Parity computes the difference in average scores between
    two groups. The classifier can measure any attribute: toxicity,
    sentiment, formality, complexity, or custom properties. This
    generality makes it useful for exploring new bias dimensions
    without implementing new metrics.
    
    For example, you could use Score Parity to compare:
    - Formality: Are texts about Group A more formal?
    - Complexity: Are texts about Group B simpler?
    - Domain-specific attributes: Medical accuracy, legal precision, etc.
    
    Reference
    ---------
    Borkan, D., Dixon, L., Sorensen, J., Thain, N., & Vasserman, L. (2019).
    Nuanced Metrics for Measuring Unintended Bias with Real Data for Text
    Classification. WWW 2019.
    
    Examples
    --------
    >>> from bias_scope.generated_text import ScoreParity
    >>> 
    >>> # Define custom classifier (e.g., formality scorer)
    >>> def formality_scorer(texts):
    ...     # Returns formality score 0-1 for each text
    ...     return [0.8, 0.6, 0.9]  # Example scores
    >>> 
    >>> # Initialize with classifier
    >>> parity = ScoreParity(classifier=formality_scorer)
    >>> 
    >>> # Generated texts for two groups
    >>> group_a = [["The man is a doctor"], ["He is intelligent"]]
    >>> group_b = [["The woman is a nurse"], ["She is caring"]]
    >>> 
    >>> # Compute score parity
    >>> scores = parity.evaluate(group_a, group_b)
    >>> print(scores)
    >>> # {'difference': 0.15, 'group_a_mean': 0.75, 'group_b_mean': 0.60}
    """
    
    def __init__(self, classifier: Callable[[List[str]], List[float]]):
        """
        Initialize Score Parity metric.

        Args:
            classifier (Callable[[List[str]], List[float]]): Function that takes
                a list of texts and returns a list of scores (one per text).
                Scores should be in [0, 1].

        Raises:
            ValueError: If classifier is not callable
        """
        if not callable(classifier):
            raise ValueError("classifier must be a callable function")
        
        self.classifier = classifier
    
    @property
    def name(self) -> str:
        """Return metric name."""
        return "Score Parity"
    
    def evaluate(
        self,
        group_a_texts: List[List[str]],
        group_b_texts: List[List[str]]
    ) -> Dict[str, float]:
        """

        Args:
            group_a_texts (List[List[str]]): Generated texts for group A.
                Shape: (n_prompts, n_texts_per_prompt)
                
                Example:
                    [
                        ["The man is a doctor", "He is smart"],
                        ["The man is strong", "He works hard"]
                    ]
            group_b_texts (List[List[str]]): Generated texts for group B.
                Same structure as group_a_texts.
        
        Returns:
            Dict[str, float]: Statistical differences:
                - 'difference': mean_A - mean_B
                - 'effect_size': Cohen's d
                - 'group_a_mean': mean score for A
                - 'group_b_mean': mean score for B
                - 'group_a_std': standard deviation for A
                - 'group_b_std': standard deviation for B
        
        Raises:
            ValueError: If inputs are invalid
        
        Notes:
            **Algorithm:**
            1. Flatten text lists for each group
            2. Compute scores for all texts using the injected classifier
            3. Calculate means and standard deviations
            4. Compute effect size (Cohen's d):
               d = (mean_A - mean_B) / pooled_std
            
            **Interpretation:**
            - difference != 0: Bias exists (disparate impact)
            - effect_size:
              - 0.2: Small effect
              - 0.5: Medium effect
              - 0.8: Large effect
        
        Examples:
            >>> # Classifier that returns length of text
            >>> def len_score(texts): return [len(t) for t in texts]
            >>> 
            >>> parity = ScoreParity(classifier=len_score)
            >>> 
            >>> # Group A generates longer texts
            >>> sc = parity.evaluate([["Long text"]], [["Short"]])
            >>> print(sc['difference'])  # > 0
        """
        # Validate inputs
        self._validate_generated_texts(group_a_texts, "group_a_texts")
        self._validate_generated_texts(group_b_texts, "group_b_texts")
        
        # Flatten texts
        flat_a = [t for sublist in group_a_texts for t in sublist]
        flat_b = [t for sublist in group_b_texts for t in sublist]
        
        # Get scores
        scores_a = self.classifier(flat_a)
        scores_b = self.classifier(flat_b)
        
        # Validate scores
        self._validate_classifier_scores(scores_a, "group_a_scores")
        self._validate_classifier_scores(scores_b, "group_b_scores")
        
        # Statistics
        mean_a = np.mean(scores_a)
        mean_b = np.mean(scores_b)
        std_a = np.std(scores_a, ddof=1)
        std_b = np.std(scores_b, ddof=1)
        
        # Difference
        diff = mean_a - mean_b
        
        # Effect size (Cohen's d)
        n_a = len(scores_a)
        n_b = len(scores_b)
        
        if n_a + n_b > 2:
            # Pooled standard deviation
            pooled_var = ((n_a - 1) * std_a**2 + (n_b - 1) * std_b**2) / (n_a + n_b - 2)
            pooled_std = np.sqrt(pooled_var)
            
            if pooled_std > 1e-10:
                effect_size = diff / pooled_std
            else:
                effect_size = 0.0
        else:
            effect_size = 0.0
            
        return {
            'difference': float(diff),
            'effect_size': float(effect_size),
            'group_a_mean': float(mean_a),
            'group_b_mean': float(mean_b),
            'group_a_std': float(std_a),
            'group_b_std': float(std_b)
        }
