"""Stereotypical Associations - Rule-based stereotype detection."""

from typing import Sequence, Dict, Literal, List
import re

from bias_scope.base import GeneratedTextMetric
from bias_scope.generated_text_based._helpers import (
    tokenize,
    normalize_lexicon,
    find_token_positions
)


class StereotypicalAssociations(GeneratedTextMetric):
    """
    Stereotypical Associations.
    
    Detects stereotypical associations in generated text using rule-based
    pattern matching. Supports both token-window matching (group terms
    near attribute terms) and regex pattern matching.
    
    Reference
    ---------
    Text-based stereotype auditing with rule-based detection. Compatible
    with modern stereotype benchmarking frameworks.
    
    Examples
    --------
    >>> from bias_scope.generated_text_based import StereotypicalAssociations
    >>> 
    >>> sa = StereotypicalAssociations()
    >>> 
    >>> generations = [
    ...     "Women are bad at math.",
    ...     "Men are strong leaders.",
    ...     "The doctor is highly skilled."
    ... ]
    >>> 
    >>> rules = [
    ...     {
    ...         'name': 'women_math_negative',
    ...         'group_terms': ['woman', 'women', 'she'],
    ...         'attribute_terms': ['bad', 'poor', 'weak'],
    ...         'polarity': 'negative'
    ...     },
    ...     {
    ...         'name': 'men_leadership',
    ...         'group_terms': ['man', 'men', 'he'],
    ...         'attribute_terms': ['leader', 'strong', 'powerful']
    ...     }
    ... ]
    >>> 
    >>> result = sa.evaluate(
    ...     generations=generations,
    ...     stereotype_rules=rules,
    ...     context_window=5
    ... )
    >>> 
    >>> print(f"Hit rate: {result['overall']['any_hit_rate_per_1k']:.1f} per 1k")
    """
    
    def evaluate(
        self,
        generations: Sequence[str],
        stereotype_rules: Sequence[dict],
        *,
        context_window: int = 10,
        matcher: Literal["token_window", "regex"] = "token_window",
        case_insensitive: bool = True,
    ) -> Dict:
        """
        Evaluate Stereotypical Associations.
        
        Args:
            generations (Sequence[str]): Generated texts to analyze
            stereotype_rules (Sequence[dict]): List of stereotype detection rules
            context_window (int): Window size for token_window matcher (default: 10)
            matcher (Literal["token_window", "regex"]): Matching strategy (default: "token_window")
            case_insensitive (bool): Case-insensitive matching (default: True)
        
        Returns:
            Dict: Results including per-rule hits and overall statistics
        
        Raises:
            ValueError: If inputs are invalid
        
        Notes:
            **Rule Schema:**
            Each rule dict must include:
            - 'name': str (unique identifier)
            
            For token_window matcher:
            - 'group_terms': Sequence[str] (demographic terms)
            - 'attribute_terms': Sequence[str] (stereotype attributes)
            
            For regex matcher:
            - 'pattern': str (regex pattern)
            
            Optional fields:
            - 'polarity': 'any'|'negative'|'positive'
            - 'notes': str (description)
            
            **Matching Behavior:**
            - token_window: Finds group terms and attribute terms, checks if
              any are within ±context_window tokens
            - regex: Matches regex pattern against text
            
            **Return Dictionary Structure:**
            {
                'metric': 'StereotypicalAssociations',
                'category': 'generated_text',
                'matcher': str,
                'context_window': int,
                'rules': [
                    {'name': str, 'hits': int, 'rate_per_1k': float}
                ],
                'overall': {
                    'any_hit_generations': int,
                    'any_hit_rate_per_1k': float
                },
                'per_generation': [
                    {'any_hit': bool, 'hits': List[str]}
                ]
            }
        """
        # Validate inputs
        generations_list = self._validate_texts(generations, "generations")
        
        if not isinstance(stereotype_rules, Sequence):
            raise TypeError("stereotype_rules must be a Sequence (list)")
        
        if len(stereotype_rules) == 0:
            raise ValueError("stereotype_rules cannot be empty")
        
        if context_window < 1:
            raise ValueError(f"context_window must be >= 1, got {context_window}")
        
        if matcher not in {"token_window", "regex"}:
            raise ValueError(f"matcher must be 'token_window' or 'regex', got '{matcher}'")
        
        # Validate and process rules
        processed_rules = []
        for i, rule in enumerate(stereotype_rules):
            if not isinstance(rule, dict):
                raise TypeError(f"stereotype_rules[{i}] must be a dict")
            
            if 'name' not in rule:
                raise ValueError(f"stereotype_rules[{i}] missing required key 'name'")
            
            rule_name = rule['name']
            
            if matcher == "token_window":
                # Validate required keys
                if 'group_terms' not in rule:
                    raise ValueError(
                        f"Rule '{rule_name}' missing 'group_terms' for token_window matcher"
                    )
                if 'attribute_terms' not in rule:
                    raise ValueError(
                        f"Rule '{rule_name}' missing 'attribute_terms' for token_window matcher"
                    )
                
                # Normalize lexicons
                group_terms = self._validate_texts(
                    rule['group_terms'], 
                    f"rule '{rule_name}' group_terms"
                )
                attribute_terms = self._validate_texts(
                    rule['attribute_terms'],
                    f"rule '{rule_name}' attribute_terms"
                )
                
                # Normalize lexicons based on case sensitivity
                if case_insensitive:
                    group_lex = normalize_lexicon(group_terms)
                    attr_lex = normalize_lexicon(attribute_terms)
                else:
                    # Case-sensitive: preserve case
                    group_lex = set(group_terms)
                    attr_lex = set(attribute_terms)
                
                processed_rules.append({
                    'name': rule_name,
                    'type': 'token_window',
                    'group_lex': group_lex,
                    'attr_lex': attr_lex
                })
            
            elif matcher == "regex":
                # Validate required keys
                if 'pattern' not in rule:
                    raise ValueError(
                        f"Rule '{rule_name}' missing 'pattern' for regex matcher"
                    )
                
                pattern = rule['pattern']
                
                # Try to compile regex
                try:
                    flags = re.IGNORECASE if case_insensitive else 0
                    compiled_pattern = re.compile(pattern, flags)
                except re.error as e:
                    raise ValueError(
                        f"Rule '{rule_name}' has invalid regex pattern: {e}"
                    ) from e
                
                processed_rules.append({
                    'name': rule_name,
                    'type': 'regex',
                    'pattern': compiled_pattern
                })
        
        # Apply rules to generations
        rule_hits = {rule['name']: 0 for rule in processed_rules}
        per_generation = []
        any_hit_count = 0
        
        for text in generations_list:
            generation_hits = []
            generation_any_hit = False
            
            # Tokenize for token_window matcher
            if matcher == "token_window":
                if case_insensitive:
                    tokens = tokenize(text)  # Lowercase tokenization
                else:
                    # Case-sensitive tokenization: preserve case
                    tokens = re.findall(r"[a-zA-Z0-9']+", text)
            else:
                # For regex, we'll work with original text
                text_to_match = text.lower() if case_insensitive else text
            
            # Apply each rule
            for rule in processed_rules:
                hit = False
                
                if rule['type'] == 'token_window':
                    # Find group and attribute positions
                    group_positions = find_token_positions(tokens, rule['group_lex'])
                    attr_positions = find_token_positions(tokens, rule['attr_lex'])
                    
                    # Check if any group is within window of any attribute
                    for g_pos in group_positions:
                        for a_pos in attr_positions:
                            if abs(g_pos - a_pos) <= context_window:
                                hit = True
                                break
                        if hit:
                            break
                
                elif rule['type'] == 'regex':
                    # Check regex match
                    if rule['pattern'].search(text_to_match):
                        hit = True
                
                if hit:
                    rule_hits[rule['name']] += 1
                    generation_hits.append(rule['name'])
                    generation_any_hit = True
            
            per_generation.append({
                'any_hit': generation_any_hit,
                'hits': generation_hits
            })
            
            if generation_any_hit:
                any_hit_count += 1
        
        # Compute rates per 1k
        n_generations = len(generations_list)
        scale_factor = 1000.0 / n_generations if n_generations > 0 else 0.0
        
        rules_results = []
        for rule in processed_rules:
            rule_name = rule['name']
            hits = rule_hits[rule_name]
            rate_per_1k = float(hits * scale_factor)
            
            rules_results.append({
                'name': rule_name,
                'hits': hits,
                'rate_per_1k': rate_per_1k
            })
        
        any_hit_rate = float(any_hit_count * scale_factor)
        
        # Return results
        return {
            'metric': 'StereotypicalAssociations',
            'category': self.category,
            'matcher': matcher,
            'context_window': context_window,
            'rules': rules_results,
            'overall': {
                'any_hit_generations': any_hit_count,
                'any_hit_rate_per_1k': any_hit_rate
            },
            'per_generation': per_generation
        }


