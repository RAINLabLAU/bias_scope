"""Regard Score - Sentiment-based bias detection."""

from typing import List, Dict
import numpy as np
from transformers import pipeline

from bias_scope.base import GeneratedTextMetric


class RegardScore(GeneratedTextMetric):
    """
    Regard Score.
    
    Measures bias in generated text by analyzing sentiment differences
    across demographic groups. The metric uses a sentiment classifier
    to score how positively or negatively a language model describes
    different groups, revealing implicit biases in text generation.
    
    Regard is computed by generating text continuations for prompts
    mentioning different demographic groups (e.g., "The man/woman is a"),
    then classifying each continuation as positive, negative, or neutral.
    Significant differences in sentiment distributions indicate bias.
    
    Unlike toxicity-based metrics that focus on harmful content, Regard
    captures subtle biases in how groups are portrayed. For example, a
    model might describe men with positive professional terms while
    describing women with appearance-focused language, even without
    explicit toxicity.
    
    Reference
    ---------
    Sheng, E., Chang, K. W., Natarajan, P., & Peng, N. (2019).
    The Woman Worked as a Babysitter: On Biases in Language Generation.
    EMNLP 2019.
    
    Examples
    --------
    >>> from bias_scope.generated_text import RegardScore
    >>> 
    >>> # Initialize (automatically loads sentiment classifier)
    >>> regard = RegardScore()
    >>> 
    >>> # Generated texts for two groups
    >>> group_a = [
    ...     ["The man is a doctor", "He is intelligent"],
    ...     ["The man is successful", "He is strong"]
    ... ]
    >>> group_b = [
    ...     ["The woman is a nurse", "She is pretty"],
    ...     ["The woman is caring", "She is kind"]
    ... ]
    >>> 
    >>> # Compute regard scores
    >>> scores = regard.evaluate(group_a, group_b)
    >>> print(scores)
    >>> # {'positive_diff': 0.15, 'negative_diff': -0.10, 'neutral_diff': -0.05}
    """
    
    def __init__(self, model_name: str = "siebert/sentiment-roberta-large-english"):
        """
        Initialize Regard Score metric.
        
        Parameters
        ----------
        model_name : str, default="siebert/sentiment-roberta-large-english"
            HuggingFace model for sentiment classification.
            Default is a high-quality RoBERTa model fine-tuned on sentiment.
        """
        self.sentiment_classifier = pipeline(
            "sentiment-analysis",
            model=model_name,
            return_all_scores=True
        )
    
    @property
    def name(self) -> str:
        """Return metric name."""
        return "Regard Score"
    
    @property
    def reference(self) -> str:
        """Return paper citation."""
        return (
            "Sheng, E., Chang, K. W., Natarajan, P., & Peng, N. (2019). "
            "The Woman Worked as a Babysitter: On Biases in Language Generation. "
            "EMNLP 2019."
        )
    
    @property
    def complexity(self) -> str:
        """Return complexity rating."""
        return "medium"
    
    def evaluate(
        self,
        group_a_texts: List[List[str]],
        group_b_texts: List[List[str]]
    ) -> Dict[str, float]:
        """
        Compute regard score differences between two groups.
        
        Analyzes sentiment distributions for texts about two demographic
        groups and computes the difference in positive, negative, and
        neutral regard.
        
        Parameters
        ----------
        group_a_texts : List[List[str]]
            Generated texts for group A (e.g., male-related prompts).
            Shape: (n_prompts, n_texts_per_prompt)
            
            Example:
                [
                    ["The man is a doctor", "He is smart"],
                    ["The man is strong", "He works hard"]
                ]
        group_b_texts : List[List[str]]
            Generated texts for group B (e.g., female-related prompts).
            Same structure as group_a_texts.
        
        Returns
        -------
        Dict[str, float]
            Dictionary with sentiment differences:
            - 'positive_diff': P(positive|A) - P(positive|B)
            - 'negative_diff': P(negative|A) - P(negative|B)
            - 'neutral_diff': P(neutral|A) - P(neutral|B)
            - 'group_a_positive': Fraction of positive texts for group A
            - 'group_a_negative': Fraction of negative texts for group A
            - 'group_a_neutral': Fraction of neutral texts for group A
            - 'group_b_positive': Fraction of positive texts for group B
            - 'group_b_negative': Fraction of negative texts for group B
            - 'group_b_neutral': Fraction of neutral texts for group B
        Evaluate Regard Score differences.
        
        Args:
            group_a_texts (List[List[str]]): Texts generated for group A (e.g., "The man...")
            group_b_texts (List[List[str]]): Texts generated for group B (e.g., "The woman...")
        
        Returns:
            Dict[str, float]: Differences in sentiment ratios (A - B)
                - 'positive_difference': pos_A - pos_B
                - 'negative_difference': neg_A - neg_B
                - 'neutral_difference': neu_A - neu_B
                - 'group_a_positive': pos_A
                - 'group_b_positive': pos_B
                ... and so on for all sentiments
        
        Raises:
            ValueError: If inputs are invalid
        
        Notes:
            **Algorithm:**
            1. Flatten text lists for each group
            2. Classify sentiment of every text (Positive, Negative, Neutral)
            3. Compute distribution of sentiments for each group
            4. Calculate difference: distribution_A - distribution_B
            
            **Interpretation:**
            - positive_difference > 0: Group A has more positive texts
            - negative_difference > 0: Group A has more negative texts
        
        Examples:
            >>> regard = RegardScore()
            >>> 
            >>> group_a = [["He is great"]]
            >>> group_b = [["She is bad"]]
            >>> 
            >>> scores = regard.evaluate(group_a, group_b)
            >>> print(scores['positive_difference'])  # > 0
            >>> print(scores['negative_difference'])  # < 0
        """
        # Validate inputs
        self._validate_generated_texts(group_a_texts, "group_a_texts")
        self._validate_generated_texts(group_b_texts, "group_b_texts")
        
        # Flatten texts
        flat_a = [t for sublist in group_a_texts for t in sublist]
        flat_b = [t for sublist in group_b_texts for t in sublist]
        
        # Get sentiments
        sentiments_a = self._score_sentiments(flat_a)
        sentiments_b = self._score_sentiments(flat_b)
        
        # Compute distributions
        dist_a = self._compute_distribution(sentiments_a)
        dist_b = self._compute_distribution(sentiments_b)
        
        # Calculate differences and combine results
        results = {}
        
        # Differences
        for label in ['positive', 'negative', 'neutral']:
            diff = dist_a.get(label, 0.0) - dist_b.get(label, 0.0)
            results[f'{label}_difference'] = float(diff)
        
        # Add absolute scores for details
        for label, score in dist_a.items():
            results[f'group_a_{label}'] = float(score)
        for label, score in dist_b.items():
            results[f'group_b_{label}'] = float(score)
            
        return results
    
    def _score_sentiments(self, texts: List[str]) -> List[str]:
        """
        Classify sentiment for a batch of texts (PRIVATE).
        
        Args:
            texts (List[str]): Texts to classify
            
        Returns:
            List[str]: Labels ('positive', 'negative', 'neutral')
        """
        if not texts:
            return []
            
        # HuggingFace pipeline returns list of dicts: [{'label': 'POSITIVE', 'score': 0.9}, ...]
        # Note: Return structure depends on model, but we map to standard 3 classes
        results = self.sentiment_classifier(texts)
        
        labels = []
        for res in results:
            # Handle different return formats (some return list of scores, some single dict)
            if isinstance(res, list):
                # return_all_scores=True case, find max score
                top_class = max(res, key=lambda x: x['score'])
                label = top_class['label']
            else:
                label = res['label']
            
            labels.append(self._normalize_label(label))
            
        return labels
    
    def _normalize_label(self, label: str) -> str:
        """
        Normalize sentiment label to standard set (PRIVATE).
        
        Args:
            label (str): Raw label from model (e.g., "LABEL_0", "POS")
            
        Returns:
            str: 'positive', 'negative', or 'neutral'
        """
        label = label.upper()
        
        if 'POS' in label or 'LABEL_2' in label:
            return 'positive'
        elif 'NEG' in label or 'LABEL_0' in label:
            return 'negative'
        else:
            return 'neutral'
            
    def _compute_distribution(self, labels: List[str]) -> Dict[str, float]:
        """
        Compute percentage distribution of labels (PRIVATE).
        
        Args:
            labels (List[str]): List of sentiment labels
            
        Returns:
            Dict[str, float]: Dictionary mapping label to fraction (0-1)
        """
        if not labels:
            return {}
            
        total = len(labels)
        counts = {}
        
        for label in labels:
            counts[label] = counts.get(label, 0) + 1
            
        return {k: v / total for k, v in counts.items()}
