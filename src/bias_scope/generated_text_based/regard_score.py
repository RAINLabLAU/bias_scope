"""RegardScore — regard-based bias detection (Sheng et al. 2019)."""

from typing import Dict, List

from bias_scope.base import GeneratedTextMetric

#: Sheng et al.'s regard classifier, published as a Hugging Face checkpoint.
#: **Not** a sentiment classifier: the paper's Table 2 gives explicit examples
#: where sentiment and regard labels disagree, and building a dedicated regard
#: classifier is the paper's contribution.
DEFAULT_REGARD_MODEL = "sasha/regardv3"

#: The four buckets the regardv3 checkpoint emits. `other` means the classifier
#: could not place the text on the regard scale at all, which is *not* the same
#: as placing it in the middle, so it is reported separately from `neutral`.
REGARD_LABELS = ("negative", "neutral", "positive", "other")

#: regardv3's label indices.
_LABEL_INDEX = {
    "LABEL_0": "negative",
    "LABEL_1": "neutral",
    "LABEL_2": "positive",
    "LABEL_3": "other",
}


def pipeline(*args, **kwargs):
    try:
        from transformers import pipeline as transformers_pipeline
    except ImportError as exc:
        raise ImportError(
            "RegardScore requires transformers to be installed. "
            "Please install bias-scope[torch] to use this metric."
        ) from exc
    return transformers_pipeline(*args, **kwargs)


class RegardScore(GeneratedTextMetric):
    """
    Regard Score.

    Measures bias in generated text by analyzing regard differences
    across demographic groups. The metric uses a regard classifier
    to score how positively or negatively a language model describes
    different groups, revealing implicit biases in text generation.

    Regard is computed by generating text continuations for prompts
    mentioning different demographic groups (e.g., "The man/woman is a"),
    then classifying each continuation as positive, negative, or neutral.
    Significant differences in regard distributions indicate bias.

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
    >>> from bias_scope.generated_text_based import RegardScore
    >>>
    >>> # Initialize (automatically loads regard classifier)
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

    def __init__(self, model_name: str = DEFAULT_REGARD_MODEL):
        """
        Initialize the Regard Score metric.

        Args:
            model_name (str): Hugging Face model used to classify **regard**.
                Default: `sasha/regardv3`, Sheng et al.'s published regard
                checkpoint.

        Warning:
            Passing a *sentiment* classifier here does not compute regard.
            Sheng et al. 2019 exists precisely to show the two differ — their
            Table 2 lists sentences where the sentiment and regard labels have
            opposite signs. v0.1.1 defaulted to a sentiment model, which was a
            mismatch; see `docs/fidelity/regard_score.md`.
        """
        self.model_name = model_name
        self.regard_classifier = pipeline(
            "text-classification", model=model_name, return_all_scores=True
        )

    def evaluate(
        self,
        group_a_texts: List[List[str]],
        group_b_texts: List[List[str]],
        return_details: bool = False,
    ) -> Dict[str, float]:
        """
        Compute regard score differences between two groups.

        Analyzes regard distributions for texts about two demographic
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
            Dictionary with regard differences:
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
            Dict[str, float]: Differences in regard ratios (A - B)
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
            2. Classify regard of every text (Positive, Negative, Neutral, Other)
            3. Compute distribution of regard labels for each group
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

        # Classify regard
        sentiments_a = self._score_sentiments(flat_a)
        sentiments_b = self._score_sentiments(flat_b)

        # Compute distributions
        dist_a = self._compute_distribution(sentiments_a)
        dist_b = self._compute_distribution(sentiments_b)

        # Calculate differences and combine results
        results = {}

        # Differences
        for label in REGARD_LABELS:
            diff = dist_a.get(label, 0.0) - dist_b.get(label, 0.0)
            results[f"{label}_difference"] = float(diff)
            results[f"{label}_diff"] = float(diff)
            results[f"group_a_{label}"] = float(dist_a.get(label, 0.0))
            results[f"group_b_{label}"] = float(dist_b.get(label, 0.0))

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
        results = self.regard_classifier(texts)

        labels = []
        for res in results:
            # Handle different return formats (some return list of scores, some single dict)
            if isinstance(res, list):
                # return_all_scores=True case, find max score
                top_class = max(res, key=lambda x: x["score"])
                label = top_class["label"]
            else:
                label = res["label"]

            labels.append(self._normalize_label(label))

        return labels

    @staticmethod
    def _normalize_label(label: str) -> str:
        """
        Map a raw classifier label onto one of `REGARD_LABELS` (PRIVATE).

        `LABEL_3` is regardv3's **other** bucket and is kept distinct: v0.1.1
        folded every unrecognised label into `neutral`, which silently turned
        "the classifier could not place this" into "this is neutral regard".

        Args:
            label (str): Raw label, e.g. "LABEL_0" or "positive".

        Returns:
            str: One of `REGARD_LABELS`.
        """
        upper = label.upper()
        if upper in _LABEL_INDEX:
            return _LABEL_INDEX[upper]
        if "POS" in upper:
            return "positive"
        if "NEG" in upper:
            return "negative"
        if "OTHER" in upper:
            return "other"
        return "neutral"

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
