from __future__ import annotations

from types import SimpleNamespace

import pytest

from bias_scope.prompts_based.realtoxicityprompts import RealToxicityPrompts


class FakeDataset:
    def __init__(self, rows):
        self._rows = list(rows)

    def __len__(self):
        return len(self._rows)

    def __iter__(self):
        return iter(self._rows)

    def select(self, indices):
        return FakeDataset([self._rows[i] for i in indices])


def _response(text: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))]
    )


def test_realtoxicityprompts_evaluate_scores_toxic_generations(monkeypatch):
    rows = [
        {
            "prompt": {"text": "Describe this person:", "toxicity": 0.1},
            "continuation": {"text": "They seem kind.", "toxicity": 0.05},
        },
        {
            "prompt": {"text": "What do you think of them?", "toxicity": 0.2},
            "continuation": {"text": "They seem rude.", "toxicity": 0.2},
        },
    ]
    answers = {
        "Describe this person:": _response("They are stupid and worthless."),
        "What do you think of them?": _response("I hate them, they are garbage."),
    }

    from bias_scope.prompts_based import realtoxicityprompts as rtp_module

    monkeypatch.setattr(
        rtp_module, "load_dataset", lambda *args, **kwargs: FakeDataset(rows)
    )
    monkeypatch.setattr(
        rtp_module,
        "completion",
        lambda **kwargs: answers[kwargs["messages"][0]["content"]],
    )

    metric = RealToxicityPrompts(model_name="test-model")
    result = metric.evaluate()

    assert result["toxicity_rate"] == pytest.approx(1.0)
    assert result["average_generated_toxicity"] > result["average_prompt_toxicity"]
    assert result["per_prompt"]["Describe this person:"]["toxic"] is True


def test_realtoxicityprompts_evaluate_scores_non_toxic_generations(monkeypatch):
    rows = [
        {
            "prompt": {"text": "Tell me about your friend.", "toxicity": 0.4},
            "continuation": {"text": "They are nice.", "toxicity": 0.1},
        }
    ]

    from bias_scope.prompts_based import realtoxicityprompts as rtp_module

    monkeypatch.setattr(
        rtp_module, "load_dataset", lambda *args, **kwargs: FakeDataset(rows)
    )
    monkeypatch.setattr(
        rtp_module, "completion", lambda **kwargs: _response("They are kind and helpful.")
    )

    metric = RealToxicityPrompts(model_name="test-model")
    result = metric.evaluate()

    assert result["toxicity_rate"] == pytest.approx(0.0)
    assert result["average_generated_toxicity"] == pytest.approx(0.0)
    assert result["per_prompt"]["Tell me about your friend."]["toxic"] is False


def test_realtoxicityprompts_respects_num_prompts_limit(monkeypatch):
    rows = [
        {"prompt": {"text": "P1", "toxicity": 0.0}, "continuation": {"toxicity": 0.0}},
        {"prompt": {"text": "P2", "toxicity": 0.0}, "continuation": {"toxicity": 0.0}},
    ]

    from bias_scope.prompts_based import realtoxicityprompts as rtp_module

    monkeypatch.setattr(
        rtp_module, "load_dataset", lambda *args, **kwargs: FakeDataset(rows)
    )
    monkeypatch.setattr(rtp_module, "completion", lambda **kwargs: _response("safe"))

    metric = RealToxicityPrompts(model_name="test-model")
    result = metric.evaluate(num_prompts=1)

    assert len(result["per_prompt"]) == 1


def test_realtoxicityprompts_raises_on_invalid_num_prompts():
    metric = RealToxicityPrompts(model_name="test-model")

    with pytest.raises(ValueError):
        metric.evaluate(num_prompts=0)


def test_realtoxicityprompts_raises_on_invalid_threshold():
    metric = RealToxicityPrompts(model_name="test-model")

    with pytest.raises(ValueError, match="toxicity_threshold"):
        metric.evaluate(toxicity_threshold=1.5)


def test_realtoxicityprompts_raises_on_empty_dataset(monkeypatch):
    from bias_scope.prompts_based import realtoxicityprompts as rtp_module

    monkeypatch.setattr(
        rtp_module, "load_dataset", lambda *args, **kwargs: FakeDataset([])
    )
    monkeypatch.setattr(rtp_module, "completion", lambda **kwargs: _response(""))

    metric = RealToxicityPrompts(model_name="test-model")

    with pytest.raises(ValueError, match="Dataset cannot be empty"):
        metric.evaluate()


def test_realtoxicityprompts_extract_response_text_falls_back_to_empty_string():
    metric = RealToxicityPrompts(model_name="test-model")

    assert metric._extract_response_text(object()) == ""


def test_realtoxicityprompts_category_property():
    metric = RealToxicityPrompts(model_name="test-model")

    assert metric.category == "prompt_based"
