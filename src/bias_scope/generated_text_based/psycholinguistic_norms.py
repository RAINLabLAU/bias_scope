"""
Psycholinguistic Norms metric for generated text.
"""

from __future__ import annotations

import re
from typing import Dict, List

import numpy as np

from bias_scope.base import GeneratedTextMetric


class PsycholinguisticNorms(GeneratedTextMetric):
    """
    Lexicon-based psycholinguistic norms metric for generated completions.

    This metric aggregates word-level psycholinguistic ratings
    (e.g., valence/arousal/dominance) over model outputs.

    Let:
        - T = set of templates
        - K = number of completions per template
        - d = a psycholinguistic dimension (e.g., valence)
        - S_d(w) = norm score of word w on dimension d

    For a completion c, completion-level score is:
        C_d(c) = mean_{w in c intersect L_d} S_d(w)

    Final metric per dimension:
        PN_d = mean_{t in T} mean_{c in topK(t)} C_d(c)

    Completions with no lexicon-covered words are skipped by default.
    """

    def evaluate(
        self,
        completions: List[List[str]],
        norms_lexicon: Dict[str, Dict[str, float]],
        dimensions: List[str] | None = None,
        return_details: bool = False,
        lowercase: bool = True,
        skip_uncovered_completions: bool = True,
    ) -> Dict[str, float]:
        """
        Evaluate psycholinguistic norms over generated completions.

        Args:
            completions:
                List over templates; each element is a list of top-k generated
                completions for that template.
            norms_lexicon:
                Mapping from token -> dimension scores.
                Example:
                    {
                        "kind": {"valence": 7.8, "arousal": 3.1, "dominance": 6.2},
                        "thief": {"valence": 2.2, "arousal": 5.9, "dominance": 3.5},
                    }
            dimensions:
                Dimensions to aggregate.
                If None, uses sorted keys from the first lexicon entry.
            return_details:
                If True, include coverage and per-template stats.
            lowercase:
                If True, lowercase tokens before lexicon lookup.
            skip_uncovered_completions:
                If True, completions with no lexicon-covered tokens are skipped.
                If False, raise ValueError when such a completion is encountered.
        """
        self._validate_completions(completions)
        lexicon = self._validate_and_normalize_lexicon(norms_lexicon, lowercase=lowercase)

        if dimensions is None:
            first_word = next(iter(lexicon))
            dimensions = sorted(list(lexicon[first_word].keys()))
        else:
            if len(dimensions) == 0:
                raise ValueError("dimensions cannot be empty")
            missing_dimensions = [d for d in dimensions if d not in next(iter(lexicon.values())).keys()]
            if len(missing_dimensions) > 0:
                raise ValueError(
                    f"Requested dimensions not found in lexicon schema: {missing_dimensions}"
                )

        # Per-dimension completion-level scores
        per_dim_completion_scores: Dict[str, List[float]] = {d: [] for d in dimensions}
        # Per-template completion means for optional diagnostics
        per_template_dim_means: List[Dict[str, float]] = []

        total_completions = 0
        covered_completions = 0
        total_tokens = 0
        covered_tokens = 0

        for template_candidates in completions:
            template_scores: Dict[str, List[float]] = {d: [] for d in dimensions}

            for candidate in template_candidates:
                total_completions += 1
                tokens = self._tokenize(candidate, lowercase=lowercase)
                total_tokens += len(tokens)

                hit_vectors: List[Dict[str, float]] = []
                for token in tokens:
                    row = lexicon.get(token)
                    if row is not None:
                        covered_tokens += 1
                        hit_vectors.append(row)

                if len(hit_vectors) == 0:
                    if skip_uncovered_completions:
                        continue
                    raise ValueError(
                        "Encountered completion with no lexicon-covered tokens while "
                        "skip_uncovered_completions=False."
                    )

                covered_completions += 1

                for d in dimensions:
                    c_score = float(np.mean([v[d] for v in hit_vectors]))
                    per_dim_completion_scores[d].append(c_score)
                    template_scores[d].append(c_score)

            # Record per-template means only from covered completions
            if all(len(template_scores[d]) > 0 for d in dimensions):
                per_template_dim_means.append(
                    {f"template_mean::{d}": float(np.mean(template_scores[d])) for d in dimensions}
                )

        if covered_completions == 0:
            raise ValueError("No completion had lexicon-covered tokens; cannot compute metric.")

        result: Dict[str, float] = {
            f"pn::{d}": float(np.mean(scores))
            for d, scores in per_dim_completion_scores.items()
        }

        if not return_details:
            return result

        result["num_templates"] = float(len(completions))
        result["k"] = float(len(completions[0]))
        result["num_completions"] = float(total_completions)
        result["num_scored_completions"] = float(covered_completions)
        result["completion_coverage_rate"] = float(covered_completions / total_completions)
        result["num_tokens"] = float(total_tokens)
        result["num_covered_tokens"] = float(covered_tokens)
        result["token_coverage_rate"] = float(covered_tokens / total_tokens) if total_tokens > 0 else 0.0

        # Template-level means aggregated across templates (when available)
        if len(per_template_dim_means) > 0:
            for d in dimensions:
                result[f"avg_template_mean::{d}"] = float(
                    np.mean([x[f"template_mean::{d}"] for x in per_template_dim_means])
                )

        return result

    def _validate_and_normalize_lexicon(
        self,
        norms_lexicon: Dict[str, Dict[str, float]],
        lowercase: bool,
    ) -> Dict[str, Dict[str, float]]:
        if not isinstance(norms_lexicon, dict) or len(norms_lexicon) == 0:
            raise ValueError("norms_lexicon must be a non-empty dict[str, dict[str, float]]")

        normalized: Dict[str, Dict[str, float]] = {}
        expected_dims: List[str] | None = None

        for word, dim_scores in norms_lexicon.items():
            if not isinstance(dim_scores, dict) or len(dim_scores) == 0:
                raise ValueError(f"Lexicon entry for '{word}' must be a non-empty dict of dimension scores")

            dims = sorted(list(dim_scores.keys()))
            if expected_dims is None:
                expected_dims = dims
            elif dims != expected_dims:
                raise ValueError(
                    f"All lexicon entries must share the same dimensions. "
                    f"Expected {expected_dims}, got {dims} for '{word}'."
                )

            casted: Dict[str, float] = {}
            for d, value in dim_scores.items():
                if not isinstance(value, (int, float)):
                    raise ValueError(f"Score for word='{word}', dimension='{d}' must be numeric")
                v = float(value)
                if np.isnan(v) or np.isinf(v):
                    raise ValueError(f"Score for word='{word}', dimension='{d}' is invalid: {v}")
                casted[d] = v

            key = word.lower() if lowercase else word
            normalized[key] = casted

        return normalized

    @staticmethod
    def _tokenize(text: str, lowercase: bool) -> List[str]:
        normalized = text.lower() if lowercase else text
        return re.findall(r"\b\w+\b", normalized)
