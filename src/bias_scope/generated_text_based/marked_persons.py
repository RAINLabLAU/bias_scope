"""Marked Persons - Log-odds analysis for marked vs unmarked personas."""

from typing import Sequence, Callable, Optional, Dict, List
import numpy as np
from collections import Counter

from bias_scope.base import GeneratedTextMetric
from bias_scope.generated_text_based._helpers import (
    tokenize,
    compute_log_odds_with_prior
)


class MarkedPersons(GeneratedTextMetric):
    """
    Marked Persons.
    
    Identifies distinctive vocabulary between marked (demographic-specific)
    and unmarked (baseline) personas using log-odds ratio with informative
    Dirichlet prior (Monroe et al., 2008).
    
    Computes z-scores to identify terms that are significantly more associated
    with marked vs unmarked contexts, revealing linguistic markedness patterns.
    
    Reference
    ---------
    Monroe, B. L., Colaresi, M. P., & Quinn, K. M. (2008). Fightin' Words:
    Lexical Feature Selection and Evaluation for Identifying the Content of
    Political Conflict. Political Analysis, 16(4), 372-403.
    
    Conceptually grounded in marked persona analysis (Cheng et al., 2023).
    
    Examples
    --------
    >>> from bias_scope.generated_text_based import MarkedPersons
    >>> 
    >>> mp = MarkedPersons()
    >>> 
    >>> marked_texts = [
    ...     "She is nurturing and empathetic.",
    ...     "She loves caring for others."
    ... ]
    >>> 
    >>> unmarked_texts = [
    ...     "The person is analytical and logical.",
    ...     "They excel at problem-solving."
    ... ]
    >>> 
    >>> result = mp.evaluate(
    ...     marked_generations=marked_texts,
    ...     unmarked_generations=unmarked_texts,
    ...     min_count=1
    ... )
    >>> 
    >>> print("Top marked terms:")
    >>> for term_info in result['top_marked_terms'][:5]:
    ...     print(f"  {term_info['term']}: z={term_info['z']:.2f}")
    """
    
    @property
    def name(self) -> str:
        """Return metric name."""
        return "MarkedPersons"
    
    def evaluate(
        self,
        marked_generations: Sequence[str],
        unmarked_generations: Sequence[str],
        *,
        prior_alpha: float = 0.01,
        min_count: int = 5,
        return_top_k: int = 50,
        tokenizer: Optional[Callable[[str], List[str]]] = None
    ) -> Dict:
        """
        Evaluate Marked Persons metric.
        
        Args:
            marked_generations (Sequence[str]): Texts with marked demographic context
            unmarked_generations (Sequence[str]): Baseline texts without demographic marking
            prior_alpha (float): Prior strength parameter (default: 0.01)
            min_count (int): Minimum total occurrences to include term (default: 5)
            return_top_k (int): Number of top terms to return (default: 50)
            tokenizer (Optional[Callable]): Custom tokenizer function (default: built-in)
        
        Returns:
            Dict: Results including top marked/unmarked terms and statistics
        
        Raises:
            ValueError: If inputs are invalid
        
        Notes:
            **Log-Odds with Informative Prior:**
            Uses Monroe et al. (2008) method:
            
            delta(w) = log((c_m(w) + a_w) / (n_m - c_m(w) + a_¬w))
                     - log((c_u(w) + a_w) / (n_u - c_u(w) + a_¬w))
            
            var(w) = 1/(c_m(w) + a_w) + 1/(c_u(w) + a_w)
            
            z(w) = delta(w) / sqrt(var(w))
            
            Where:
            - c_m(w): count in marked corpus
            - c_u(w): count in unmarked corpus  
            - n_m, n_u: total tokens
            - a_w: prior_alpha * c_bg(w) (background count)
            - a_total: prior_alpha * sum(c_bg)
            - a_¬w: a_total - a_w
            
            **Return Dictionary Structure:**
            {
                'metric': 'MarkedPersons',
                'category': 'generated_text',
                'prior_alpha': float,
                'min_count': int,
                'return_top_k': int,
                'top_marked_terms': [
                    {'term': str, 'z': float, 'delta': float, 
                     'c_marked': int, 'c_unmarked': int}
                ],
                'top_unmarked_terms': [...],
                'terms': {
                    '<term>': {'z': float, 'delta': float, 
                              'c_marked': int, 'c_unmarked': int}
                },
                'summary': {
                    'vocab_considered': int,
                    'marked_total_tokens': int,
                    'unmarked_total_tokens': int
                }
            }
        """
        # Validate inputs
        marked_list = self._validate_texts(marked_generations, "marked_generations")
        unmarked_list = self._validate_texts(unmarked_generations, "unmarked_generations")
        
        if prior_alpha <= 0:
            raise ValueError(f"prior_alpha must be > 0, got {prior_alpha}")
        
        if min_count < 1:
            raise ValueError(f"min_count must be >= 1, got {min_count}")
        
        if return_top_k < 1:
            raise ValueError(f"return_top_k must be >= 1, got {return_top_k}")
        
        if tokenizer is not None:
            self._validate_callable(tokenizer, "tokenizer")
            tokenize_fn = tokenizer
        else:
            tokenize_fn = tokenize
        
        # Tokenize and count
        marked_tokens = []
        for text in marked_list:
            tokens = tokenize_fn(text)
            marked_tokens.extend(tokens)
        
        unmarked_tokens = []
        for text in unmarked_list:
            tokens = tokenize_fn(text)
            unmarked_tokens.extend(tokens)
        
        # Count frequencies
        marked_counts = Counter(marked_tokens)
        unmarked_counts = Counter(unmarked_tokens)
        
        n_marked = len(marked_tokens)
        n_unmarked = len(unmarked_tokens)
        
        # Combine to get background counts
        all_vocab = set(marked_counts.keys()) | set(unmarked_counts.keys())
        background_counts = Counter()
        for word in all_vocab:
            background_counts[word] = marked_counts.get(word, 0) + unmarked_counts.get(word, 0)
        
        # Compute prior mass
        total_bg_count = sum(background_counts.values())
        a_total = prior_alpha * total_bg_count
        
        # Compute log-odds for each term
        term_scores = {}
        
        for word in all_vocab:
            c_marked = marked_counts.get(word, 0)
            c_unmarked = unmarked_counts.get(word, 0)
            c_bg = background_counts[word]
            
            # Filter by min_count
            if c_marked + c_unmarked < min_count:
                continue
            
            # Compute prior for this word
            a_w = prior_alpha * c_bg
            
            # Compute log-odds with prior
            delta, variance, z_score = compute_log_odds_with_prior(
                c_marked, c_unmarked, n_marked, n_unmarked, a_w, a_total
            )
            
            term_scores[word] = {
                'z': z_score,
                'delta': delta,
                'c_marked': c_marked,
                'c_unmarked': c_unmarked
            }
        
        # Sort by z-score
        sorted_terms = sorted(term_scores.items(), key=lambda x: x[1]['z'], reverse=True)
        
        # Extract top marked (positive z) and top unmarked (negative z)
        top_marked = []
        for word, scores in sorted_terms[:return_top_k]:
            top_marked.append({
                'term': word,
                'z': float(scores['z']),
                'delta': float(scores['delta']),
                'c_marked': int(scores['c_marked']),
                'c_unmarked': int(scores['c_unmarked'])
            })
        
        top_unmarked = []
        for word, scores in sorted_terms[-return_top_k:][::-1]:
            top_unmarked.append({
                'term': word,
                'z': float(scores['z']),
                'delta': float(scores['delta']),
                'c_marked': int(scores['c_marked']),
                'c_unmarked': int(scores['c_unmarked'])
            })
        
        # Convert term_scores for return
        terms_dict = {}
        for word, scores in term_scores.items():
            terms_dict[word] = {
                'z': float(scores['z']),
                'delta': float(scores['delta']),
                'c_marked': int(scores['c_marked']),
                'c_unmarked': int(scores['c_unmarked'])
            }
        
        # Return results
        return {
            'metric': self.name,
            'category': self.category,
            'prior_alpha': prior_alpha,
            'min_count': min_count,
            'return_top_k': return_top_k,
            'top_marked_terms': top_marked,
            'top_unmarked_terms': top_unmarked,
            'terms': terms_dict,
            'summary': {
                'vocab_considered': len(term_scores),
                'marked_total_tokens': n_marked,
                'unmarked_total_tokens': n_unmarked
            }
        }


