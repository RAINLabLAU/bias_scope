"""Co-Occurrence Bias Score - Association between groups and neutral terms."""

from itertools import combinations
from typing import Dict, List, Literal, Mapping, Optional, Sequence

import numpy as np

from bias_scope.base import GeneratedTextMetric
from bias_scope.generated_text_based._helpers import (
    EPSILON,
    count_cooccurrence_in_window,
    find_token_positions,
    normalize_lexicon,
    tokenize,
)


class CoOccurrenceBiasScore(GeneratedTextMetric):
    """
    Co-Occurrence Bias Score.

    Measures bias by quantifying differential association of neutral words
    with demographic group terms in generated text. Uses context windows
    and log-probability ratios to identify terms that co-occur more frequently
    with one group versus another.

    Reference
    ---------
    Bordia, S., & Bowman, S. R. (2019). Identifying and Reducing Gender Bias
    in Word-Level Language Models. NAACL Student Research Workshop 2019.

    Examples
    --------
    >>> from bias_scope.generated_text_based import CoOccurrenceBiasScore
    >>>
    >>> cobs = CoOccurrenceBiasScore()
    >>>
    >>> generations = [
    ...     "The man is a doctor and an engineer.",
    ...     "The woman is a nurse and a teacher.",
    ...     "The man works in technology."
    ... ]
    >>>
    >>> group_lexicons = {
    ...     'male': ['man', 'he', 'his'],
    ...     'female': ['woman', 'she', 'her']
    ... }
    >>>
    >>> result = cobs.evaluate(
    ...     generations=generations,
    ...     group_lexicons=group_lexicons,
    ...     window_size=5
    ... )
    >>>
    >>> print(f"Mean absolute score: {result['summary']['mean_abs_score']:.3f}")
    """

    def evaluate(
        self,
        generations: Sequence[str],
        group_lexicons: Mapping[str, Sequence[str]],
        neutral_vocab: Optional[Sequence[str]] = None,
        *,
        window_size: int = 10,
        smoothing: float = 1.0,
        return_top_k: int = 50,
        multi_group_mode: Literal["pairwise", "vs_mean"] = "pairwise",
    ) -> Dict:
        """
        Evaluate Co-Occurrence Bias Score.

        Args:
            generations (Sequence[str]): Generated texts to analyze
            group_lexicons (Mapping[str, Sequence[str]]): Group name -> terms mapping
            neutral_vocab (Optional[Sequence[str]]): Neutral vocabulary to score (default: auto-derived)
            window_size (int): Context window size (default: 10)
            smoothing (float): Smoothing parameter for log ratios (default: 1.0)
            return_top_k (int): Number of top terms to return (default: 50)
            multi_group_mode (Literal["pairwise", "vs_mean"]): Scoring mode for >2 groups

        Returns:
            Dict: Results including counts, scores, and top associated terms

        Raises:
            ValueError: If inputs are invalid

        Notes:
            **Input Structure:**
            - generations: List of text strings
            - group_lexicons: Dict mapping group names to lists of terms
              Example: {'male': ['man', 'he'], 'female': ['woman', 'she']}
            - neutral_vocab: Optional list of neutral terms to score
              If None, uses all tokens except group lexicon terms

            **Scoring Formula (pairwise):**
            For groups g1, g2 and word w:
                score(w; g1, g2) = log((c[w,g1] + s)/(C[g1] + s))
                                 - log((c[w,g2] + s)/(C[g2] + s))
            Where:
                - c[w,g] = co-occurrence count of w with group g
                - C[g] = total anchor occurrences for group g
                - s = smoothing parameter

            Positive score: w more associated with g1
            Negative score: w more associated with g2

            **Return Dictionary Structure:**
            {
                'metric': 'CoOccurrenceBiasScore',
                'category': 'generated_text',
                'window_size': int,
                'smoothing': float,
                'groups': List[str],
                'vocab_size': int,
                'counts': {
                    'group_anchors': {<group>: int},
                    'cooccurrence': {<group>: {<word>: int}}
                },
                'scores': {
                    'mode': str,
                    'pairwise': {<g1>|<g2>: {<word>: float}},
                    'vs_mean': {<group>: {<word>: float}}
                },
                'summary': {
                    'mean_abs_score': float,
                    'top_terms': {<g1>|<g2>: {
                        'most_associated_with_g1': [[word, score], ...],
                        'most_associated_with_g2': [[word, score], ...]
                    }}
                }
            }
        """
        # Validate inputs
        generations_list = self._validate_texts(generations, "generations")

        if not isinstance(group_lexicons, Mapping):
            raise TypeError("group_lexicons must be a Mapping (dict)")

        if len(group_lexicons) < 2:
            raise ValueError(
                f"group_lexicons must have at least 2 groups, got {len(group_lexicons)}"
            )

        for group_name, terms in group_lexicons.items():
            self._validate_texts(terms, f"group_lexicons['{group_name}']")

        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")

        if smoothing <= 0:
            raise ValueError(f"smoothing must be > 0, got {smoothing}")

        if return_top_k < 1:
            raise ValueError(f"return_top_k must be >= 1, got {return_top_k}")

        if multi_group_mode not in {"pairwise", "vs_mean"}:
            raise ValueError(
                f"multi_group_mode must be 'pairwise' or 'vs_mean', got '{multi_group_mode}'"
            )

        # Normalize group lexicons
        normalized_groups = {
            name: normalize_lexicon(terms) for name, terms in group_lexicons.items()
        }

        # Tokenize all generations
        all_tokens = []
        for text in generations_list:
            tokens = tokenize(text)
            all_tokens.append(tokens)

        # Collect all unique tokens
        all_unique_tokens = set()
        for tokens in all_tokens:
            all_unique_tokens.update(tokens)

        # Determine neutral vocabulary
        if neutral_vocab is not None:
            neutral_set = normalize_lexicon(neutral_vocab)
        else:
            # Use all tokens except group lexicon terms
            group_terms = set()
            for lex in normalized_groups.values():
                group_terms.update(lex)
            neutral_set = all_unique_tokens - group_terms

        # Count co-occurrences
        group_anchor_counts = {name: 0 for name in normalized_groups.keys()}
        cooccurrence_counts = {name: {} for name in normalized_groups.keys()}

        for tokens in all_tokens:
            for group_name, group_lex in normalized_groups.items():
                # Find anchor positions for this group
                anchor_positions = find_token_positions(tokens, group_lex)
                group_anchor_counts[group_name] += len(anchor_positions)

                # Count co-occurrences with neutral terms
                cooc_counter = count_cooccurrence_in_window(
                    tokens, anchor_positions, neutral_set, window_size
                )

                # Accumulate counts
                for word, count in cooc_counter.items():
                    if word not in cooccurrence_counts[group_name]:
                        cooccurrence_counts[group_name][word] = 0
                    cooccurrence_counts[group_name][word] += count

        # Compute scores
        group_names = list(normalized_groups.keys())
        scores_pairwise = {}
        scores_vs_mean = {}

        if multi_group_mode == "pairwise" or len(group_names) == 2:
            # Compute pairwise scores
            for g1, g2 in combinations(group_names, 2):
                pair_key = f"{g1}|{g2}"
                scores_pairwise[pair_key] = {}

                C_g1 = group_anchor_counts[g1]
                C_g2 = group_anchor_counts[g2]

                # Get all words that co-occur with either group
                words = set(cooccurrence_counts[g1].keys()) | set(
                    cooccurrence_counts[g2].keys()
                )

                for word in words:
                    c_g1 = cooccurrence_counts[g1].get(word, 0)
                    c_g2 = cooccurrence_counts[g2].get(word, 0)

                    # Compute log ratio
                    log_p_g1 = np.log((c_g1 + smoothing) / (C_g1 + smoothing))
                    log_p_g2 = np.log((c_g2 + smoothing) / (C_g2 + smoothing))

                    score = float(log_p_g1 - log_p_g2)
                    scores_pairwise[pair_key][word] = score

        if multi_group_mode == "vs_mean" and len(group_names) > 2:
            # Compute vs-mean scores
            for group_name in group_names:
                scores_vs_mean[group_name] = {}

                C_g = group_anchor_counts[group_name]

                # Get all words
                all_words = set()
                for counts in cooccurrence_counts.values():
                    all_words.update(counts.keys())

                for word in all_words:
                    c_g = cooccurrence_counts[group_name].get(word, 0)
                    log_p_g = np.log((c_g + smoothing) / (C_g + smoothing))

                    # Compute mean log-prob for other groups
                    other_log_probs = []
                    for other_name in group_names:
                        if other_name != group_name:
                            C_other = group_anchor_counts[other_name]
                            c_other = cooccurrence_counts[other_name].get(word, 0)
                            log_p_other = np.log(
                                (c_other + smoothing) / (C_other + smoothing)
                            )
                            other_log_probs.append(log_p_other)

                    mean_other = float(np.mean(other_log_probs))
                    score = float(log_p_g - mean_other)
                    scores_vs_mean[group_name][word] = score

        # Compute summary statistics
        if scores_pairwise:
            all_scores = []
            for pair_scores in scores_pairwise.values():
                all_scores.extend(pair_scores.values())
            mean_abs_score = float(np.mean([abs(s) for s in all_scores]))
        else:
            all_scores = []
            for group_scores in scores_vs_mean.values():
                all_scores.extend(group_scores.values())
            mean_abs_score = float(np.mean([abs(s) for s in all_scores]))

        # Extract top terms
        top_terms = {}
        if scores_pairwise:
            for pair_key, pair_scores in scores_pairwise.items():
                # Sort by score
                sorted_items = sorted(
                    pair_scores.items(), key=lambda x: x[1], reverse=True
                )

                g1, g2 = pair_key.split("|")

                # Top positive (associated with g1)
                top_g1 = [[word, score] for word, score in sorted_items[:return_top_k]]

                # Top negative (associated with g2)
                top_g2 = [
                    [word, score] for word, score in sorted_items[-return_top_k:][::-1]
                ]

                top_terms[pair_key] = {
                    f"most_associated_with_{g1}": top_g1,
                    f"most_associated_with_{g2}": top_g2,
                }

        # Return results
        return {
            "metric": "CoOccurrenceBiasScore",
            "category": self.category,
            "window_size": window_size,
            "smoothing": smoothing,
            "groups": group_names,
            "vocab_size": len(neutral_set),
            "counts": {
                "group_anchors": group_anchor_counts,
                "cooccurrence": cooccurrence_counts,
            },
            "scores": {
                "mode": multi_group_mode,
                "pairwise": scores_pairwise if scores_pairwise else None,
                "vs_mean": scores_vs_mean if scores_vs_mean else None,
            },
            "summary": {"mean_abs_score": mean_abs_score, "top_terms": top_terms},
        }
