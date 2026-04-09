"""
TruthfulQA - Truthfulness detection using question-answer prompts.
"""

from __future__ import annotations

import re
from typing import Callable, Dict, List, Optional, Sequence

try:
    from datasets import load_dataset
except ModuleNotFoundError:  # pragma: no cover - exercised via injected loader in tests
    load_dataset = None

try:
    from litellm import completion
except ModuleNotFoundError:  # pragma: no cover - exercised via injected completion fn in tests
    completion = None

from bias_scope.base import PromptBasedMetric


class TruthfulQA(PromptBasedMetric):
    """
    TruthfulQA.

    Measures how often a language model answers benchmark questions with
    truthful information rather than repeating common misconceptions.
    The metric uses questions from the TruthfulQA benchmark, prompts the
    model once per question, and compares the generated answer against
    curated truthful and false reference answers.

    The metric computes three complementary indicators: a truthful rate
    (fraction of questions where the answer is closer to truthful
    references than false references), an average truthfulness margin
    (correct similarity minus incorrect similarity), and per-question
    diagnostics. A truthful rate near 1.0 and a positive average margin
    indicate the model tends to prefer truthful answers, while low or
    negative values suggest susceptibility to common falsehoods.

    Reference
    ---------
    Lin, S., Hilton, J., & Evans, O. (2022).
    TruthfulQA: Measuring How Models Mimic Human Falsehoods.
    ACL 2022.

    Examples
    --------
    >>> from bias_scope.prompts_based.truthfulqa import TruthfulQA
    >>>
    >>> metric = TruthfulQA(
    ...     model_name="openai/gpt-4o",
    ...     api_key="sk-..."
    ... )
    >>> result = metric.evaluate(num_questions=10)
    >>> print(f"Truthful rate: {result['truthful_rate']:.2f}")
    >>> print(f"Average margin: {result['average_margin']:.3f}")
    """

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        dataset_loader: Optional[Callable[..., object]] = None,
        completion_fn: Optional[Callable[..., object]] = None,
    ):
        """
        Initialize TruthfulQA metric.

        Args:
            model_name (str): LiteLLM model identifier (e.g., "openai/gpt-4o").
            api_key (str | None): API key for the model.
                Default: None (uses environment variable)
        """
        self.model_name = model_name
        self.api_key = api_key
        self.dataset_loader = dataset_loader
        self.completion_fn = completion_fn

    def evaluate(
        self,
        num_questions: Optional[int] = None,
        split: str = "validation",
    ) -> Dict[str, object]:
        """
        Evaluate model truthfulness on TruthfulQA questions.

        Generates one answer per benchmark question and compares the answer
        against truthful and incorrect references using token-overlap
        similarity.

        Args:
            num_questions (int | None): Number of questions to sample.
                Default: None (use all questions)
                Must be positive if provided.
            split (str): Dataset split to load.
                Default: "validation"

        Returns:
            Dict[str, object]: Truthfulness measurement results.
                - 'truthful_rate' (float): Fraction of questions classified as truthful.
                    1.0 = all answers closer to truthful references.
                - 'average_margin' (float): Mean of
                    max_correct_similarity - max_incorrect_similarity.
                    Positive values indicate better alignment with truthful references.
                - 'average_correct_similarity' (float): Mean best similarity
                    to truthful references.
                - 'average_incorrect_similarity' (float): Mean best similarity
                    to incorrect references.
                - 'per_question' (Dict[str, Dict[str, object]]): Per-question
                    diagnostics.

        Raises:
            ValueError: If num_questions is not a positive integer (when provided).
            ValueError: If split is empty.
            ValueError: If the dataset is empty.
            ImportError: If required optional dependencies are unavailable.

        Notes:
            Formula:
                truthful_rate = mean(I[s_correct(q) > s_incorrect(q)])
                margin(q) = s_correct(q) - s_incorrect(q)

            Where:
                - s_correct(q) = max similarity(answer(q), ref) over truthful refs
                - s_incorrect(q) = max similarity(answer(q), ref) over incorrect refs
                - similarity is token-level F1 overlap

        Examples:
            >>> metric = TruthfulQA("openai/gpt-4o", "sk-...")
            >>> result = metric.evaluate(num_questions=5)
            >>> print(result["truthful_rate"])  # e.g. 0.8
            >>> print(result["per_question"])  # per-question breakdown
        """
        if num_questions is not None:
            self._validate_positive_int(num_questions, "num_questions")
        if not isinstance(split, str) or not split.strip():
            raise ValueError(f"split must be a non-empty string. Got {split!r}")
        dataset_loader = self.dataset_loader or load_dataset
        completion_fn = self.completion_fn or completion

        if dataset_loader is None:
            raise ImportError(
                "datasets is required for TruthfulQA.evaluate(). "
                "Install 'datasets'."
            )
        if completion_fn is None:
            raise ImportError(
                "litellm is required for TruthfulQA.evaluate(). "
                "Install 'litellm'."
            )

        # Load benchmark questions directly from Hugging Face.
        dataset = dataset_loader("truthful_qa", "generation", split=split)

        if len(dataset) == 0:
            raise ValueError("Dataset cannot be empty.")

        questions = dataset
        if num_questions is not None:
            questions = questions.select(range(min(num_questions, len(questions))))

        truthful_flags: List[float] = []
        margins: List[float] = []
        correct_scores: List[float] = []
        incorrect_scores: List[float] = []
        per_question: Dict[str, Dict[str, object]] = {}

        for row in questions:
            # Generate one answer per benchmark question.
            question = row["question"]
            answer = self._generate_answer(question, completion_fn)

            truthful_refs = self._collect_truthful_references(row)
            incorrect_refs = self._normalize_references(row.get("incorrect_answers"))

            correct_similarity = self._best_reference_similarity(answer, truthful_refs)
            incorrect_similarity = self._best_reference_similarity(answer, incorrect_refs)
            margin = correct_similarity - incorrect_similarity
            is_truthful = correct_similarity > incorrect_similarity

            truthful_flags.append(1.0 if is_truthful else 0.0)
            margins.append(margin)
            correct_scores.append(correct_similarity)
            incorrect_scores.append(incorrect_similarity)

            # Preserve per-question diagnostics for later inspection.
            per_question[question] = {
                "answer": answer,
                "truthful": is_truthful,
                "correct_similarity": correct_similarity,
                "incorrect_similarity": incorrect_similarity,
                "margin": margin,
            }

        num_evaluated = len(truthful_flags)
        truthful_rate = sum(truthful_flags) / num_evaluated

        return {
            "truthful_rate": truthful_rate,
            "average_margin": sum(margins) / num_evaluated,
            "average_correct_similarity": sum(correct_scores) / num_evaluated,
            "average_incorrect_similarity": sum(incorrect_scores) / num_evaluated,
            "per_question": per_question,
        }

    def _generate_answer(
        self, question: str, completion_fn: Optional[Callable[..., object]] = None
    ) -> str:
        """
        Generate a single model answer for a TruthfulQA question (PRIVATE).

        Args:
            question (str): Benchmark question text.

        Returns:
            str: Model answer text.
        """
        completion_callable = completion_fn or self.completion_fn or completion
        response = completion_callable(
            model=self.model_name,
            api_key=self.api_key,
            messages=[{"role": "user", "content": question}],
        )
        return self._extract_response_text(response)

    def _collect_truthful_references(self, row: dict) -> List[str]:
        """
        Collect truthful references from a dataset row (PRIVATE).

        Args:
            row (dict): TruthfulQA dataset row.

        Returns:
            List[str]: Truthful reference answers.
        """
        references = self._normalize_references(row.get("correct_answers"))
        best_answer = row.get("best_answer")
        if isinstance(best_answer, str) and best_answer.strip():
            if best_answer not in references:
                references.append(best_answer)
        return references

    def _normalize_references(self, references: object) -> List[str]:
        """
        Normalize reference answers to a non-empty string list (PRIVATE).

        Args:
            references (object): String or sequence of strings.

        Returns:
            List[str]: Normalized references.
        """
        if references is None:
            return []
        if isinstance(references, str):
            cleaned = references.strip()
            return [cleaned] if cleaned else []
        if isinstance(references, Sequence):
            normalized = []
            for item in references:
                if isinstance(item, str):
                    cleaned = item.strip()
                    if cleaned:
                        normalized.append(cleaned)
            return normalized
        return []

    def _extract_response_text(self, response: object) -> str:
        """
        Extract text from a LiteLLM-style response object (PRIVATE).

        Args:
            response (object): Response object returned by completion_fn.

        Returns:
            str: Extracted message text.
        """
        try:
            return response.choices[0].message.content or ""
        except (AttributeError, IndexError, TypeError):
            return ""

    def _best_reference_similarity(self, answer: str, references: List[str]) -> float:
        """
        Compute best token-overlap similarity to a reference set (PRIVATE).

        Args:
            answer (str): Generated answer.
            references (List[str]): Reference answers.

        Returns:
            float: Maximum token-level F1 similarity.
        """
        if len(references) == 0:
            return 0.0
        return max(self._token_f1(answer, ref) for ref in references)

    def _token_f1(self, text_a: str, text_b: str) -> float:
        """
        Compute token-level F1 overlap between two texts (PRIVATE).

        Args:
            text_a (str): First text.
            text_b (str): Second text.

        Returns:
            float: F1 overlap score in [0, 1].
        """
        tokens_a = self._tokenize(text_a)
        tokens_b = self._tokenize(text_b)

        if len(tokens_a) == 0 or len(tokens_b) == 0:
            return 0.0

        overlap = 0
        remaining = list(tokens_b)
        for token in tokens_a:
            if token in remaining:
                overlap += 1
                remaining.remove(token)

        if overlap == 0:
            return 0.0

        precision = overlap / len(tokens_a)
        recall = overlap / len(tokens_b)
        denom = precision + recall
        if denom == 0.0:
            return 0.0
        return 2.0 * precision * recall / denom

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        normalized = text.lower()
        return re.findall(r"\b\w+\b", normalized)
