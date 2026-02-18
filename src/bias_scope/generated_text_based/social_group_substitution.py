"""Social Group Substitution - Counterfactual fairness metric."""

from typing import Sequence, Mapping, Callable, Dict, Optional, Literal, Union
import numpy as np
from itertools import combinations
import inspect

from bias_scope.base import GeneratedTextMetric


class SocialGroupSubstitution(GeneratedTextMetric):
    """
    Social Group Substitution.
    
    Measures bias by comparing model outputs across counterfactual prompts
    where only demographic group terms are substituted. Uses counterfactual
    fairness framework to quantify disparate treatment.
    
    Computes individual unfairness (prompt-level variation) and group
    disparity (aggregate differences across demographic groups).
    
    Reference
    ---------
    Huang, P.-S., Zhang, H., Jiang, R., Stanforth, R., Welbl, J., Rae, J., ...
    & Kohli, P. (2020). Reducing Sentiment Bias in Language Models via
    Counterfactual Evaluation. Findings of EMNLP 2020.
    
    Examples
    --------
    >>> from bias_scope.generated_text_based import SocialGroupSubstitution
    >>> 
    >>> sgs = SocialGroupSubstitution()
    >>> 
    >>> # Define prompts with placeholders
    >>> prompts = [
    ...     "The {gender} is a talented {profession}.",
    ...     "{gender} are often good at {skill}."
    ... ]
    >>> 
    >>> # Substitution values
    >>> substitutions = {
    ...     'gender': ['man', 'woman'],
    ...     'profession': ['engineer', 'nurse'],
    ...     'skill': ['math', 'communication']
    ... }
    >>> 
    >>> # Generation function
    >>> def generate(prompt):
    ...     return f"Generated: {prompt}"
    >>> 
    >>> # Scoring function (e.g., sentiment analyzer)
    >>> def score(text):
    ...     return 0.5  # sentiment score
    >>> 
    >>> result = sgs.evaluate(
    ...     prompts=prompts,
    ...     substitutions=substitutions,
    ...     generate_fn=generate,
    ...     score_fn=score
    ... )
    >>> 
    >>> print(f"Individual Unfairness: {result['individual_unfairness_overall']:.3f}")
    >>> print(f"Group Disparity: {result['group_disparity']['_overall']:.3f}")
    """
    
    @property
    def name(self) -> str:
        """Return metric name."""
        return "SocialGroupSubstitution"
    
    def evaluate(
        self,
        prompts: Sequence[str],
        substitutions: Mapping[str, Sequence[str]],
        generate_fn: Callable,
        score_fn: Callable[[str], float],
        *,
        num_samples: int = 1,
        decoding_kwargs: Optional[dict] = None,
        aggregation: Literal["mean", "median"] = "mean",
    ) -> Dict:
        """
        Evaluate Social Group Substitution metric.
        
        Args:
            prompts (Sequence[str]): Template prompts with placeholders like {group}
            substitutions (Mapping[str, Sequence[str]]): Placeholder -> values mapping
            generate_fn (Callable): Text generation function
            score_fn (Callable[[str], float]): Scoring function (sentiment, toxicity, etc.)
            num_samples (int): Number of generations per prompt (default: 1)
            decoding_kwargs (Optional[dict]): Kwargs to pass to generate_fn if supported
            aggregation (Literal["mean", "median"]): How to aggregate scores (default: "mean")
        
        Returns:
            Dict: Comprehensive results including scores, unfairness metrics, and metadata
        
        Raises:
            ValueError: If inputs are invalid
            TypeError: If functions are not callable or prompts are not sequences
        
        Notes:
            **Input Structure:**
            - prompts: List of template strings with placeholders
              Example: ["The {gender} is a {profession}."]
            - substitutions: Dict mapping placeholder names to values
              Example: {'gender': ['man', 'woman'], 'profession': ['doctor', 'nurse']}
            - generate_fn: Function that generates text
              Signature: str -> str OR Sequence[str] -> Sequence[str]
            - score_fn: Function that scores generated text
              Signature: str -> float
            
            **Return Dictionary Structure:**
            {
                'metric': 'SocialGroupSubstitution',
                'category': 'generated_text',
                'num_prompts': int,
                'num_samples': int,
                'placeholders': List[str],
                'aggregation': str,
                'scores': {
                    '<placeholder>': {
                        '<value>': [float per prompt]
                    }
                },
                'individual_unfairness': {
                    '<placeholder>': [float per prompt]
                },
                'individual_unfairness_overall': float,
                'group_means': {
                    '<placeholder>': {'<value>': float}
                },
                'group_disparity': {
                    '<placeholder>': float,
                    '_overall': float
                },
                'metadata': {
                    'supports_batched_generate_fn': bool,
                    'decoding_kwargs_used': dict
                }
            }
        """
        # Validate inputs
        prompts_list = self._validate_texts(prompts, "prompts")
        self._validate_callable(generate_fn, "generate_fn")
        self._validate_callable(score_fn, "score_fn")
        
        if not isinstance(substitutions, Mapping):
            raise TypeError("substitutions must be a Mapping (dict)")
        
        if len(substitutions) == 0:
            raise ValueError("substitutions cannot be empty")
        
        # Validate substitutions
        for key, values in substitutions.items():
            if not isinstance(key, str):
                raise TypeError(f"substitution key must be string, got {type(key).__name__}")
            
            values_list = self._validate_texts(values, f"substitutions['{key}']")
            
            if len(values_list) < 2:
                raise ValueError(
                    f"substitutions['{key}'] must have at least 2 values for comparison, "
                    f"got {len(values_list)}"
                )
        
        # Validate num_samples
        if num_samples < 1:
            raise ValueError(f"num_samples must be >= 1, got {num_samples}")
        
        # Validate aggregation
        if aggregation not in {"mean", "median"}:
            raise ValueError(f"aggregation must be 'mean' or 'median', got '{aggregation}'")
        
        # Validate that placeholder keys appear in prompts
        for key in substitutions.keys():
            placeholder = "{" + key + "}"
            if not any(placeholder in prompt for prompt in prompts_list):
                raise ValueError(
                    f"Placeholder '{{{key}}}' not found in any prompt"
                )
        
        # Validate prompt formatting
        for i, prompt in enumerate(prompts_list):
            # Try formatting with first value of each substitution
            test_kwargs = {k: list(v)[0] for k, v in substitutions.items()}
            try:
                prompt.format(**test_kwargs)
            except (KeyError, ValueError) as e:
                raise ValueError(
                    f"prompts[{i}] cannot be formatted with substitutions: {e}"
                ) from e
        
        # Check if generate_fn supports batching
        supports_batch = self._check_batch_support(generate_fn)
        
        # Set up decoding kwargs
        if decoding_kwargs is None:
            decoding_kwargs = {}
        
        # Collect all scores: scores[placeholder][value][prompt_idx] = aggregated_score
        all_scores = {key: {val: [] for val in substitutions[key]} 
                      for key in substitutions.keys()}
        
        # Generate and score for each prompt
        for prompt_idx, prompt_template in enumerate(prompts_list):
            # For each placeholder
            for placeholder_key in substitutions.keys():
                # For each substitution value
                for sub_value in substitutions[placeholder_key]:
                    # Create counterfactual prompt
                    cf_kwargs = {placeholder_key: sub_value}
                    # Fill other placeholders with first value (or leave unfilled if not present)
                    for other_key in substitutions.keys():
                        if other_key != placeholder_key and "{" + other_key + "}" in prompt_template:
                            cf_kwargs[other_key] = list(substitutions[other_key])[0]
                    
                    cf_prompt = prompt_template.format(**cf_kwargs)
                    
                    # Generate num_samples outputs
                    sample_scores = []
                    for _ in range(num_samples):
                        # Generate
                        if supports_batch:
                            # Try batch generation
                            generated = generate_fn([cf_prompt])
                            if isinstance(generated, (list, tuple)) and len(generated) > 0:
                                output = generated[0]
                            else:
                                output = str(generated)
                        else:
                            output = generate_fn(cf_prompt)
                        
                        # Score
                        score = score_fn(output)
                        validated_score = self._validate_finite_float(
                            score, 
                            f"score_fn output for prompt {prompt_idx}"
                        )
                        sample_scores.append(validated_score)
                    
                    # Aggregate samples
                    if aggregation == "mean":
                        agg_score = float(np.mean(sample_scores))
                    else:  # median
                        agg_score = float(np.median(sample_scores))
                    
                    all_scores[placeholder_key][sub_value].append(agg_score)
        
        # Compute metrics
        num_prompts = len(prompts_list)
        
        # Individual unfairness per (prompt, placeholder)
        individual_unfairness = {}
        for placeholder_key in substitutions.keys():
            if_per_prompt = []
            
            for prompt_idx in range(num_prompts):
                # Get scores for this prompt and placeholder across all values
                scores_for_prompt = [
                    all_scores[placeholder_key][val][prompt_idx]
                    for val in substitutions[placeholder_key]
                ]
                
                # Compute pairwise absolute differences
                pairwise_diffs = [
                    abs(s1 - s2)
                    for s1, s2 in combinations(scores_for_prompt, 2)
                ]
                
                # Average
                if_prompt = float(np.mean(pairwise_diffs))
                if_per_prompt.append(if_prompt)
            
            individual_unfairness[placeholder_key] = if_per_prompt
        
        # Overall individual unfairness
        all_if_values = [val for vals in individual_unfairness.values() for val in vals]
        if_overall = float(np.mean(all_if_values))
        
        # Group means
        group_means = {}
        for placeholder_key in substitutions.keys():
            group_means[placeholder_key] = {}
            for sub_value in substitutions[placeholder_key]:
                # Average across prompts
                mean_score = float(np.mean(all_scores[placeholder_key][sub_value]))
                group_means[placeholder_key][sub_value] = mean_score
        
        # Group disparity
        group_disparity = {}
        for placeholder_key in substitutions.keys():
            means = list(group_means[placeholder_key].values())
            gd = float(max(means) - min(means))
            group_disparity[placeholder_key] = gd
        
        # Overall group disparity
        gd_overall = float(np.mean(list(group_disparity.values())))
        group_disparity['_overall'] = gd_overall
        
        # Return comprehensive results
        return {
            'metric': self.name,
            'category': self.category,
            'num_prompts': num_prompts,
            'num_samples': num_samples,
            'placeholders': list(substitutions.keys()),
            'aggregation': aggregation,
            'scores': all_scores,
            'individual_unfairness': individual_unfairness,
            'individual_unfairness_overall': if_overall,
            'group_means': group_means,
            'group_disparity': group_disparity,
            'metadata': {
                'supports_batched_generate_fn': supports_batch,
                'decoding_kwargs_used': decoding_kwargs
            }
        }
    
    def _check_batch_support(self, fn: Callable) -> bool:
        """
        Check if function might support batch input (PRIVATE helper).
        
        Args:
            fn: Function to check
        
        Returns:
            bool: True if might support batching
        
        Notes:
            This is a heuristic check. We'll try to determine if the function
            accepts a sequence input based on its signature or behavior.
        """
        # Try to inspect signature
        try:
            sig = inspect.signature(fn)
            params = list(sig.parameters.values())
            if len(params) > 0:
                # If first parameter has sequence annotation, likely supports batch
                first_param = params[0]
                if first_param.annotation != inspect.Parameter.empty:
                    annotation_str = str(first_param.annotation)
                    if any(t in annotation_str for t in ['Sequence', 'List', 'list']):
                        return True
        except:
            pass
        
        # Default: assume single-input
        return False


