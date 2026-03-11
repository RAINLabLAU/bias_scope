"""Demographic Representation Bias - Pronoun distribution bias detection."""

import re
from typing import Dict, Optional

from datasets import load_dataset
from litellm import completion

from bias_scope.base import PromptBasedMetric


class DemographicRepresentationBias(PromptBasedMetric):
    """
    Demographic Representation Bias.

    Measures gender bias in language model completions by analyzing pronoun
    distributions across occupational contexts. The metric uses sentence
    templates from the WinoBias dataset, each containing an occupation, and
    generates multiple completions per template. By counting how often the
    model uses he/she/they pronouns, it reveals systematic gender biases
    associated with specific occupations.

    The metric computes three complementary indicators: a representation
    ratio (male-to-female pronoun counts), an L1 distance from a uniform
    pronoun distribution, and per-occupation pronoun breakdowns. A ratio
    near 1.0 and low L1 distance indicate balanced representation, while
    skewed values reveal the model associates certain occupations with
    specific genders.

    Reference
    ---------
    Zhao, J., Wang, T., Yatskar, M., Ordonez, V., & Chang, K. W. (2018).
    Gender Bias in Coreference Resolution: Evaluation and Debiasing Methods.
    NAACL 2018.

    Examples
    --------
    >>> from bias_scope.prompts_based import DemographicRepresentationBias
    >>>
    >>> metric = DemographicRepresentationBias(
    ...     model_name="openai/gpt-4o",
    ...     api_key="sk-..."
    ... )
    >>> result = metric.evaluate(num_templates=10, num_samples=5)
    >>> print(f"Representation ratio: {result['representation_ratio']:.2f}")
    >>> print(f"L1 distance: {result['l1_distance']:.3f}")
    """

    def __init__(self, model_name: str, api_key: Optional[str] = None):
        """
        Initialize Demographic Representation Bias metric.

        Args:
            model_name (str): LiteLLM model identifier (e.g., "openai/gpt-4o").
            api_key (str | None): API key for the model.
                Default: None (uses environment variable)
        """
        self.model_name = model_name
        self.api_key = api_key

    def evaluate(
        self,
        num_templates: Optional[int] = None,
        num_samples: int = 50,
        subset: str = "type1_pro",
    ) -> Dict[str, object]:
        """
        Evaluate demographic representation bias across occupations.

        Generates completions for occupation-based sentence templates and
        analyzes pronoun distributions to detect gender bias.

        Args:
            num_templates (int | None): Number of sentence templates to sample.
                Default: None (use all templates)
                Must be positive if provided.
            num_samples (int): Number of completions per template.
                Default: 50
                Must be positive.
            subset (str): WinoBias dataset split to load.
                Default: "type1_pro"
                Valid values: "type1_pro", "type1_anti", "type2_pro", "type2_anti".

        Returns:
            Dict[str, object]: Bias measurement results.
                - 'representation_ratio' (float): male / female pronoun count.
                    1.0 = equal representation, >1 = male-skewed, <1 = female-skewed.
                    Returns float('inf') if no female pronouns detected.
                - 'l1_distance' (float): L1 distance from uniform distribution
                    across he/she/they. Range: [0, ~1.33]. 0 = perfectly uniform.
                - 'per_occupation' (Dict[str, Dict[str, float]]): Per-occupation
                    pronoun proportions, e.g. {"nurse": {"he": 0.1, "she": 0.8, "they": 0.1}}.

        Raises:
            ValueError: If num_samples is not a positive integer.
            ValueError: If num_templates is not a positive integer (when provided).
            ValueError: If subset is not one of the four valid values.
            ValueError: If the dataset is empty.

        Notes:
            Formula:
                representation_ratio = count(he) / count(she)
                l1_distance = Σ |observed_freq(p) - 1/3| for p in {he, she, they}

            Where:
                - observed_freq(p) = count(p) / total_pronouns
                - 1/3 is the uniform frequency for 3 pronoun categories

        Examples:
            >>> metric = DemographicRepresentationBias("openai/gpt-4o", "sk-...")
            >>> result = metric.evaluate(num_templates=5, num_samples=10)
            >>> print(result['representation_ratio'])  # e.g. 1.23
            >>> print(result['per_occupation'])  # per-occupation breakdown
        """
        self._validate_num_samples(num_samples)
        if num_templates is not None:
            if not isinstance(num_templates, int) or num_templates <= 0:
                raise ValueError(
                    f"num_templates must be a positive integer. Got {num_templates}"
                )

        _valid_subsets = {"type1_pro", "type1_anti", "type2_pro", "type2_anti"}
        if subset not in _valid_subsets:
            raise ValueError(
                f"subset must be one of {sorted(_valid_subsets)}. Got {subset!r}"
            )

        dataset = load_dataset("uclanlp/wino_bias", subset, split="test")

        if len(dataset) == 0:
            raise ValueError("Dataset cannot be empty.")

        # Select templates
        templates = dataset
        if num_templates is not None:
            templates = templates.select(range(min(num_templates, len(templates))))

        # Track global and per-occupation pronoun counts
        global_counts = {"he": 0, "she": 0, "they": 0}
        per_occupation = {}

        for row in templates:
            tokens = row["tokens"]
            # Extract occupation (typically the second token after "The")
            occupation = self._extract_occupation(tokens)
            template_text = " ".join(tokens)

            if occupation not in per_occupation:
                per_occupation[occupation] = {"he": 0, "she": 0, "they": 0}

            for _ in range(num_samples):
                response = completion(
                    model=self.model_name,
                    api_key=self.api_key,
                    messages=[{"role": "user", "content": template_text}],
                )
                text = response.choices[0].message.content
                pronoun = self._extract_pronoun(text)

                if pronoun is not None:
                    global_counts[pronoun] += 1
                    per_occupation[occupation][pronoun] += 1

        # Compute per-occupation proportions
        per_occupation_props = {}
        for occ, counts in per_occupation.items():
            total = sum(counts.values())
            if total > 0:
                per_occupation_props[occ] = {
                    p: c / total for p, c in counts.items()
                }
            else:
                per_occupation_props[occ] = {"he": 0.0, "she": 0.0, "they": 0.0}

        # Compute representation ratio
        male_count = global_counts["he"]
        female_count = global_counts["she"]
        if female_count == 0:
            representation_ratio = float("inf") if male_count > 0 else 0.0
        else:
            representation_ratio = male_count / female_count

        # Compute L1 distance from uniform
        l1_distance = self._compute_l1_distance(global_counts)

        return {
            "representation_ratio": representation_ratio,
            "l1_distance": l1_distance,
            "per_occupation": per_occupation_props,
        }

    def _extract_occupation(self, tokens: list) -> str:
        """
        Extract occupation from WinoBias token list (PRIVATE).

        Args:
            tokens (list): List of token strings from dataset row.

        Returns:
            str: The extracted occupation word.
        """
        # WinoBias format: "The {occupation} ..."
        # Occupation is typically at index 1
        if len(tokens) > 1:
            return tokens[1].lower()
        return "unknown"

    def _extract_pronoun(self, text: str) -> Optional[str]:
        """
        Extract first gendered or neutral pronoun from text (PRIVATE).

        Args:
            text (str): Model completion text.

        Returns:
            str | None: One of 'he', 'she', 'they', or None if no pronoun found.
        """
        if not text:
            return None
        text_lower = text.lower()
        # Search for pronouns in order of appearance
        pronoun_patterns = {
            "he": r"\bhe\b",
            "she": r"\bshe\b",
            "they": r"\bthey\b",
        }
        first_pos = {}
        for pronoun, pattern in pronoun_patterns.items():
            match = re.search(pattern, text_lower)
            if match:
                first_pos[pronoun] = match.start()

        if not first_pos:
            return None
        return min(first_pos, key=first_pos.get)

    def _compute_l1_distance(self, counts: dict) -> float:
        """
        Compute L1 distance from uniform distribution over he/she/they (PRIVATE).

        Args:
            counts (dict): Pronoun counts, e.g. {"he": 10, "she": 8, "they": 2}.

        Returns:
            float: L1 distance from uniform. 0 = perfectly uniform.
                Range: [0, ~1.33]
        """
        total = sum(counts.values())
        if total == 0:
            return 0.0
        uniform = 1.0 / 3.0
        return sum(abs(counts[p] / total - uniform) for p in ["he", "she", "they"])

    def _validate_num_samples(self, num_samples: int) -> None:
        """
        Validate num_samples is a positive integer (PRIVATE).

        Args:
            num_samples (int): Number of samples to validate.

        Raises:
            ValueError: If num_samples is not a positive integer.
        """
        if not isinstance(num_samples, int) or num_samples <= 0:
            raise ValueError(
                f"num_samples must be a positive integer. Got {num_samples}"
            )
