from __future__ import annotations

from types import SimpleNamespace

import pytest

from bias_scope.prompts_based.opinion_consistency_across_personas import (
    OpinionConsistencyAcrossPersonas,
)


class FakeDataset:
    def __init__(self, rows):
        self._rows = list(rows)

    def __len__(self):
        return len(self._rows)

    def __iter__(self):
        return iter(self._rows)


def _response(text: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))]
    )


def test_opinion_consistency_scores_consistent_answers(monkeypatch):
    rows = [
        {
            "prompt": "<persona>p1</persona> <question>Q1?</question> <choices> (A): Yes (B): No </choices>",
            "answer": "A",
            "uid": "u1",
            "folder": "f1",
            "question_id": "q1",
        },
        {
            "prompt": "<persona>p2</persona> <question>Q1?</question> <choices> (A): Yes (B): No </choices>",
            "answer": "B",
            "uid": "u2",
            "folder": "f1",
            "question_id": "q1",
        },
    ]
    answers = {
        rows[0]["prompt"]: _response("A"),
        rows[1]["prompt"]: _response("A"),
    }

    from bias_scope.prompts_based import opinion_consistency_across_personas as metric_module

    monkeypatch.setattr(
        metric_module, "load_dataset", lambda *args, **kwargs: FakeDataset(rows)
    )
    monkeypatch.setattr(
        metric_module,
        "completion",
        lambda **kwargs: answers[kwargs["messages"][0]["content"]],
    )

    metric = OpinionConsistencyAcrossPersonas(model_name="test-model")
    result = metric.evaluate()

    assert result["opinion_consistency"] == pytest.approx(1.0)
    assert result["average_valid_response_rate"] == pytest.approx(1.0)
    assert result["per_question"]["q1"]["majority_choice"] == "A"


def test_opinion_consistency_scores_mixed_answers(monkeypatch):
    rows = [
        {
            "prompt": "<persona>p1</persona> <question>Q1?</question> <choices> (A): Yes (B): No </choices>",
            "answer": "A",
            "uid": "u1",
            "folder": "f1",
            "question_id": "q1",
        },
        {
            "prompt": "<persona>p2</persona> <question>Q1?</question> <choices> (A): Yes (B): No </choices>",
            "answer": "B",
            "uid": "u2",
            "folder": "f1",
            "question_id": "q1",
        },
    ]
    answers = {
        rows[0]["prompt"]: _response("A"),
        rows[1]["prompt"]: _response("B"),
    }

    from bias_scope.prompts_based import opinion_consistency_across_personas as metric_module

    monkeypatch.setattr(
        metric_module, "load_dataset", lambda *args, **kwargs: FakeDataset(rows)
    )
    monkeypatch.setattr(
        metric_module,
        "completion",
        lambda **kwargs: answers[kwargs["messages"][0]["content"]],
    )

    metric = OpinionConsistencyAcrossPersonas(model_name="test-model")
    result = metric.evaluate()

    assert result["opinion_consistency"] == pytest.approx(0.5)
    assert result["average_normalized_entropy"] == pytest.approx(1.0)


def test_opinion_consistency_respects_num_questions_limit(monkeypatch):
    rows = [
        {
            "prompt": "<question>Q1?</question>",
            "answer": "A",
            "uid": "u1",
            "folder": "f1",
            "question_id": "q1",
        },
        {
            "prompt": "<question>Q1?</question>",
            "answer": "A",
            "uid": "u2",
            "folder": "f1",
            "question_id": "q1",
        },
        {
            "prompt": "<question>Q2?</question>",
            "answer": "B",
            "uid": "u3",
            "folder": "f2",
            "question_id": "q2",
        },
        {
            "prompt": "<question>Q2?</question>",
            "answer": "B",
            "uid": "u4",
            "folder": "f2",
            "question_id": "q2",
        },
    ]

    from bias_scope.prompts_based import opinion_consistency_across_personas as metric_module

    monkeypatch.setattr(
        metric_module, "load_dataset", lambda *args, **kwargs: FakeDataset(rows)
    )
    monkeypatch.setattr(metric_module, "completion", lambda **kwargs: _response("A"))

    metric = OpinionConsistencyAcrossPersonas(model_name="test-model")
    result = metric.evaluate(num_questions=1)

    assert len(result["per_question"]) == 1


def test_opinion_consistency_raises_on_invalid_num_questions():
    metric = OpinionConsistencyAcrossPersonas(model_name="test-model")

    with pytest.raises(ValueError):
        metric.evaluate(num_questions=0)


def test_opinion_consistency_raises_when_no_group_meets_min_personas(monkeypatch):
    rows = [
        {
            "prompt": "<question>Q1?</question>",
            "answer": "A",
            "uid": "u1",
            "folder": "f1",
            "question_id": "q1",
        }
    ]

    from bias_scope.prompts_based import opinion_consistency_across_personas as metric_module

    monkeypatch.setattr(
        metric_module, "load_dataset", lambda *args, **kwargs: FakeDataset(rows)
    )
    monkeypatch.setattr(metric_module, "completion", lambda **kwargs: _response("A"))

    metric = OpinionConsistencyAcrossPersonas(model_name="test-model")

    with pytest.raises(ValueError, match="No question groups"):
        metric.evaluate(min_personas_per_question=2)


def test_opinion_consistency_raises_on_empty_dataset(monkeypatch):
    from bias_scope.prompts_based import opinion_consistency_across_personas as metric_module

    monkeypatch.setattr(
        metric_module, "load_dataset", lambda *args, **kwargs: FakeDataset([])
    )
    monkeypatch.setattr(metric_module, "completion", lambda **kwargs: _response("A"))

    metric = OpinionConsistencyAcrossPersonas(model_name="test-model")

    with pytest.raises(ValueError, match="Dataset cannot be empty"):
        metric.evaluate()


def test_opinion_consistency_extract_response_text_falls_back_to_empty_string():
    metric = OpinionConsistencyAcrossPersonas(model_name="test-model")

    assert metric._extract_response_text(object()) == ""


def test_opinion_consistency_category_property():
    metric = OpinionConsistencyAcrossPersonas(model_name="test-model")

    assert metric.category == "prompt_based"
