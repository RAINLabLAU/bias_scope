from __future__ import annotations

from types import SimpleNamespace

import pytest

from bias_scope.prompts_based.bold import BOLD


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


def test_bold_evaluate_scores_biased_generations(monkeypatch):
    rows = [
        {
            "domain": "gender",
            "name": "person_a",
            "category": "American_actors",
            "prompts": ["She is known for being ", "He is known for being "],
            "wikipedia": ["a respected actor.", "a respected actor."],
        }
    ]
    answers = {
        "She is known for being ": _response("lazy and weak."),
        "He is known for being ": _response("aggressive and dangerous."),
    }

    from bias_scope.prompts_based import bold as bold_module

    monkeypatch.setattr(
        bold_module, "load_dataset", lambda *args, **kwargs: FakeDataset(rows)
    )
    monkeypatch.setattr(
        bold_module,
        "completion",
        lambda **kwargs: answers[kwargs["messages"][0]["content"]],
    )

    metric = BOLD(model_name="test-model")
    result = metric.evaluate()

    assert result["bias_rate"] == pytest.approx(1.0)
    assert result["average_generated_bias"] > result["average_reference_bias"]
    assert result["by_domain"]["gender"] > 0.0
    assert result["per_prompt"]["She is known for being "]["biased"] is True


def test_bold_evaluate_scores_non_biased_generations(monkeypatch):
    rows = [
        {
            "domain": "profession",
            "name": "person_b",
            "category": "scientists",
            "prompts": ["They are known for being "],
            "wikipedia": ["a respected researcher."],
        }
    ]

    from bias_scope.prompts_based import bold as bold_module

    monkeypatch.setattr(
        bold_module, "load_dataset", lambda *args, **kwargs: FakeDataset(rows)
    )
    monkeypatch.setattr(
        bold_module, "completion", lambda **kwargs: _response("thoughtful and talented.")
    )

    metric = BOLD(model_name="test-model")
    result = metric.evaluate()

    assert result["bias_rate"] == pytest.approx(0.0)
    assert result["average_generated_bias"] == pytest.approx(0.0)
    assert result["per_prompt"]["They are known for being "]["biased"] is False


def test_bold_respects_num_prompts_limit(monkeypatch):
    rows = [
        {
            "domain": "race",
            "name": "p1",
            "category": "group",
            "prompts": ["P1", "P2"],
            "wikipedia": ["R1", "R2"],
        }
    ]

    from bias_scope.prompts_based import bold as bold_module

    monkeypatch.setattr(
        bold_module, "load_dataset", lambda *args, **kwargs: FakeDataset(rows)
    )
    monkeypatch.setattr(bold_module, "completion", lambda **kwargs: _response("safe"))

    metric = BOLD(model_name="test-model")
    result = metric.evaluate(num_prompts=1)

    assert len(result["per_prompt"]) == 1


def test_bold_raises_on_invalid_num_prompts():
    metric = BOLD(model_name="test-model")

    with pytest.raises(ValueError):
        metric.evaluate(num_prompts=0)


def test_bold_raises_on_invalid_threshold():
    metric = BOLD(model_name="test-model")

    with pytest.raises(ValueError, match="bias_threshold"):
        metric.evaluate(bias_threshold=2.0)


def test_bold_raises_on_empty_dataset(monkeypatch):
    from bias_scope.prompts_based import bold as bold_module

    monkeypatch.setattr(
        bold_module, "load_dataset", lambda *args, **kwargs: FakeDataset([])
    )
    monkeypatch.setattr(bold_module, "completion", lambda **kwargs: _response(""))

    metric = BOLD(model_name="test-model")

    with pytest.raises(ValueError, match="Dataset cannot be empty"):
        metric.evaluate()


def test_bold_extract_response_text_falls_back_to_empty_string():
    metric = BOLD(model_name="test-model")

    assert metric._extract_response_text(object()) == ""


def test_bold_category_property():
    metric = BOLD(model_name="test-model")

    assert metric.category == "prompt_based"
