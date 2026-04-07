"""
Opinion Consistency Across Personas - Consistency evaluation over persona-conditioned opinions.
"""

from __future__ import annotations

import math
import re
from typing import Dict, List, Optional

try:
    from datasets import load_dataset
except ModuleNotFoundError:  # pragma: no cover - handled in tests
    load_dataset = None

try:
    from litellm import completion
except ModuleNotFoundError:  # pragma: no cover - handled in tests
    completion = None

from bias_scope.base import PromptBasedMetric


class OpinionConsistencyAcrossPersonas(PromptBasedMetric):
    """
    Opinion Consistency Across Personas.

    Measures how consistently a model answers the same question when the
    prompt is conditioned on different personas. The metric loads
    persona-conditioned opinion prompts, generates one answer per prompt,
    extracts the model's choice label, groups generations by question,
    and reports the dominant-answer agreement within each group.

    What this implementation measures:
        If a model is highly stable across personas, it should tend to choose
        the same answer option for the same underlying question even after the
        persona framing changes. This metric treats consistency as within-question
        agreement over generated answer choices.

    Core idea:
        For each question q with multiple persona-conditioned prompts:
            1. Generate one answer for each persona variant
            2. Extract the selected choice label (A/B/C/...)
            3. Compute the majority-choice fraction within that question

    Returned aggregate signals:
        - opinion_consistency: average majority-choice fraction
        - average_valid_response_rate
        - average_normalized_entropy
    """

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
    ):
        self.model_name = model_name
        self.api_key = api_key

    def evaluate(
        self,
        num_questions: Optional[int] = None,
        split: str = "test",
        min_personas_per_question: int = 2,
    ) -> Dict[str, object]:
        """
        Evaluate opinion consistency across persona-conditioned prompts.

        This is the ONLY public method exposed by the metric.

        Args:
            num_questions (int | None): Number of grouped questions to evaluate.
            split (str): Dataset split to load.
            min_personas_per_question (int): Minimum number of persona prompts
                required for a question to be included.
        """
        if num_questions is not None:
            self._validate_positive_int(num_questions, "num_questions")
        self._validate_positive_int(
            min_personas_per_question, "min_personas_per_question"
        )
        if not isinstance(split, str) or not split.strip():
            raise ValueError(f"split must be a non-empty string. Got {split!r}")
        if load_dataset is None:
            raise ImportError(
                "datasets is required for OpinionConsistencyAcrossPersonas.evaluate(). "
                "Install 'datasets'."
            )
        if completion is None:
            raise ImportError(
                "litellm is required for OpinionConsistencyAcrossPersonas.evaluate(). "
                "Install 'litellm'."
            )

        # Load persona-conditioned opinion prompts directly from Hugging Face.
        dataset = load_dataset("RiverDong/OpinionQA", split=split)
        if len(dataset) == 0:
            raise ValueError("Dataset cannot be empty.")

        # Group rows by question so consistency is measured within a question,
        # not across unrelated prompts.
        grouped_questions = self._group_rows_by_question(dataset)
        grouped_questions = [
            group
            for group in grouped_questions
            if len(group["rows"]) >= min_personas_per_question
        ]
        if len(grouped_questions) == 0:
            raise ValueError("No question groups satisfy min_personas_per_question.")
        if num_questions is not None:
            grouped_questions = grouped_questions[:num_questions]

        consistency_scores: List[float] = []
        valid_response_rates: List[float] = []
        entropy_scores: List[float] = []
        per_question: Dict[str, Dict[str, object]] = {}

        for group in grouped_questions:
            question_id = group["question_id"]
            prompts = group["rows"]
            extracted_answers: List[str] = []
            generations: List[Dict[str, str]] = []

            for row in prompts:
                # Generate one answer per persona variant for the same question.
                generated_text = self._generate_answer(row["prompt"])
                extracted_choice = self._extract_choice(generated_text)
                generations.append(
                    {
                        "uid": row["uid"],
                        "generated_text": generated_text,
                        "generated_choice": extracted_choice or "",
                        "reference_choice": row["answer"],
                    }
                )
                if extracted_choice is not None:
                    extracted_answers.append(extracted_choice)

            valid_count = len(extracted_answers)
            total_count = len(prompts)
            valid_rate = valid_count / total_count

            if valid_count == 0:
                # No parseable choices means zero consistency and maximum uncertainty.
                consistency = 0.0
                normalized_entropy = 1.0
                majority_choice = ""
            else:
                counts = self._count_choices(extracted_answers)
                majority_choice = max(counts, key=counts.get)
                consistency = counts[majority_choice] / valid_count
                normalized_entropy = self._normalized_entropy(counts, valid_count)

            consistency_scores.append(consistency)
            valid_response_rates.append(valid_rate)
            entropy_scores.append(normalized_entropy)

            # Keep a full per-question diagnostic record for later analysis.
            per_question[question_id] = {
                "question_text": group["question_text"],
                "folder": group["folder"],
                "num_personas": float(total_count),
                "num_valid_responses": float(valid_count),
                "valid_response_rate": valid_rate,
                "consistency": consistency,
                "majority_choice": majority_choice,
                "normalized_entropy": normalized_entropy,
                "responses": generations,
            }

        num_groups = len(consistency_scores)
        return {
            "opinion_consistency": sum(consistency_scores) / num_groups,
            "average_valid_response_rate": sum(valid_response_rates) / num_groups,
            "average_normalized_entropy": sum(entropy_scores) / num_groups,
            "per_question": per_question,
        }

    def _group_rows_by_question(self, dataset: object) -> List[Dict[str, object]]:
        """
        Group persona-conditioned rows by question id (PRIVATE).

        The source dataset stores one row per persona/question pair. This
        helper reassembles those rows into question-level groups.
        """
        grouped: Dict[str, Dict[str, object]] = {}
        for row in dataset:
            question_id = row.get("question_id", "")
            prompt = row.get("prompt", "")
            answer = row.get("answer", "")
            uid = row.get("uid", "")
            folder = row.get("folder", "")
            question_text = self._extract_question_text(prompt)

            bucket = grouped.setdefault(
                question_id,
                {
                    "question_id": question_id,
                    "question_text": question_text,
                    "folder": folder,
                    "rows": [],
                },
            )
            bucket["rows"].append(
                {
                    "prompt": prompt,
                    "answer": answer,
                    "uid": uid,
                }
            )

        return [grouped[key] for key in sorted(grouped.keys())]

    def _generate_answer(self, prompt_text: str) -> str:
        """Generate one answer for a persona-conditioned prompt (PRIVATE)."""
        response = completion(
            model=self.model_name,
            api_key=self.api_key,
            messages=[{"role": "user", "content": prompt_text}],
        )
        return self._extract_response_text(response)

    def _extract_response_text(self, response: object) -> str:
        """Extract text from a LiteLLM-style response object (PRIVATE)."""
        try:
            return response.choices[0].message.content or ""
        except (AttributeError, IndexError, TypeError):
            return ""

    def _extract_choice(self, generated_text: str) -> Optional[str]:
        """
        Extract a multiple-choice label from generated text (PRIVATE).

        This keeps the metric robust to short answers like "A" and to
        formatted answers like "(B)".
        """
        text = generated_text.strip().upper()
        if not text:
            return None

        direct_match = re.search(r"\b([A-Z])\b", text)
        if direct_match and direct_match.group(1) in {"A", "B", "C", "D", "E"}:
            return direct_match.group(1)

        choice_match = re.search(r"\(([A-Z])\)", text)
        if choice_match and choice_match.group(1) in {"A", "B", "C", "D", "E"}:
            return choice_match.group(1)

        return None

    def _extract_question_text(self, prompt_text: object) -> str:
        """Extract the inner question text from XML-like prompt markup (PRIVATE)."""
        if not isinstance(prompt_text, str):
            return ""
        match = re.search(r"<question>\s*(.*?)\s*</question>", prompt_text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return prompt_text.strip()

    def _count_choices(self, choices: List[str]) -> Dict[str, int]:
        """Count extracted answer choices for one question group (PRIVATE)."""
        counts: Dict[str, int] = {}
        for choice in choices:
            counts[choice] = counts.get(choice, 0) + 1
        return counts

    def _normalized_entropy(self, counts: Dict[str, int], total: int) -> float:
        """Compute normalized answer entropy for one question group (PRIVATE)."""
        if total <= 1 or len(counts) <= 1:
            return 0.0

        entropy = 0.0
        for count in counts.values():
            probability = count / total
            entropy -= probability * math.log(probability)

        max_entropy = math.log(len(counts))
        if max_entropy == 0.0:
            return 0.0
        return entropy / max_entropy
