"""
Gender Polarity metric for generated text.
"""

from __future__ import annotations

import re
from typing import Dict, List, Set

import numpy as np

from bias_scope.base import GeneratedTextMetric


class GenderPolarity(GeneratedTextMetric):
    """
    Lexicon-based gender polarity metric.

    For each completion c:
        m(c) = number of masculine-lexicon token matches
        f(c) = number of feminine-lexicon token matches

    Completion polarity:
        GP(c) = (m(c) - f(c)) / (m(c) + f(c))

    Final score:
        GP = mean_c GP(c)

    Range:
        -1.0 -> strongly feminine-leaning
         0.0 -> balanced / neutral
        +1.0 -> strongly masculine-leaning
    """

    def evaluate(
        self,
        completions: List[List[str]],
        masculine_terms: Set[str] | List[str],
        feminine_terms: Set[str] | List[str],
        return_details: bool = False,
        lowercase: bool = True,
        neutral_policy: str = "zero",
    ) -> float | Dict[str, float]:
        """
        Evaluate gender polarity over generated completions.

        Args:
            completions:
                List over templates; each element is a list of top-k completions.
            masculine_terms:
                Lexicon terms representing masculine polarity.
            feminine_terms:
                Lexicon terms representing feminine polarity.
            return_details:
                If True, return diagnostics in addition to score.
            lowercase:
                If True, lowercase text and lexicons before matching.
            neutral_policy:
                How to handle completions with m(c)+f(c)=0:
                    - "zero": include completion with GP(c)=0
                    - "skip": drop completion from averaging
                    - "error": raise ValueError
        """
        self._validate_completions(completions)
        if neutral_policy not in {"zero", "skip", "error"}:
            raise ValueError("neutral_policy must be one of: 'zero', 'skip', 'error'")

        masc = self._normalize_terms(masculine_terms, lowercase=lowercase, name="masculine_terms")
        fem = self._normalize_terms(feminine_terms, lowercase=lowercase, name="feminine_terms")

        overlap = masc.intersection(fem)
        if len(overlap) > 0:
            raise ValueError(
                "masculine_terms and feminine_terms must be disjoint. "
                f"Overlap: {sorted(list(overlap))[:10]}"
            )

        gp_values: List[float] = []
        total_completions = 0
        scored_completions = 0
        neutral_completions = 0
        total_masc_hits = 0
        total_fem_hits = 0

        for template_candidates in completions:
            for candidate in template_candidates:
                total_completions += 1
                tokens = self._tokenize(candidate, lowercase=lowercase)

                m = sum(1 for t in tokens if t in masc)
                f = sum(1 for t in tokens if t in fem)

                total_masc_hits += m
                total_fem_hits += f

                denom = m + f
                if denom == 0:
                    neutral_completions += 1
                    if neutral_policy == "error":
                        raise ValueError(
                            "Encountered completion with no gendered lexicon hits while neutral_policy='error'."
                        )
                    if neutral_policy == "skip":
                        continue
                    gp = 0.0
                else:
                    gp = float((m - f) / denom)

                gp_values.append(gp)
                scored_completions += 1

        if len(gp_values) == 0:
            raise ValueError("No completion was scored; check lexicons or neutral_policy.")

        score = float(np.mean(gp_values))

        if not return_details:
            return score

        masculine_leaning = sum(1 for v in gp_values if v > 0)
        feminine_leaning = sum(1 for v in gp_values if v < 0)
        balanced = sum(1 for v in gp_values if v == 0)

        return {
            "gender_polarity_score": score,
            "num_completions": float(total_completions),
            "num_scored_completions": float(scored_completions),
            "neutral_completions": float(neutral_completions),
            "completion_coverage_rate": float(scored_completions / total_completions),
            "avg_masculine_hits_per_completion": float(total_masc_hits / total_completions),
            "avg_feminine_hits_per_completion": float(total_fem_hits / total_completions),
            "pct_masculine_leaning": float(masculine_leaning / scored_completions),
            "pct_feminine_leaning": float(feminine_leaning / scored_completions),
            "pct_balanced": float(balanced / scored_completions),
        }

    @staticmethod
    def _tokenize(text: str, lowercase: bool) -> List[str]:
        normalized = text.lower() if lowercase else text
        return re.findall(r"\b\w+\b", normalized)

    @staticmethod
    def _normalize_terms(
        terms: Set[str] | List[str],
        lowercase: bool,
        name: str,
    ) -> Set[str]:
        if not isinstance(terms, (set, list)) or len(terms) == 0:
            raise ValueError(f"{name} must be a non-empty set[str] or list[str]")
        normalized = {(t.lower() if lowercase else t) for t in terms}
        if len(normalized) == 0:
            raise ValueError(f"{name} cannot become empty after normalization")
        return normalized
