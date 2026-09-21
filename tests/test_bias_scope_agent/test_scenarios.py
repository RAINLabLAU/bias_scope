"""`bias_scope_agent.scenarios`: the scripted turns, and picking the scenario
for a model the user only named.

The autonomous mode asks for a model id and nothing else, so the kind of
target (masked LM, causal LM, sentence encoder, API model) must be inferred
the way `inspect_model` does it, then turned into the three scripted turns.
"""

from __future__ import annotations

from bias_scope_agent import scenarios


def test_an_openrouter_id_is_an_api_target(monkeypatch):
    monkeypatch.setattr(scenarios, "inspect_model",
                        lambda m: {"guessed_kind": None, "guessed_source": "litellm_model_id"})
    scenario, spec = scenarios.scenario_for_model("openrouter/meta-llama/llama-3.1-8b-instruct")
    assert scenario == "api" and spec["backend_kind"] == "litellm"
    turns = scenarios.turns_for_model("openrouter/meta-llama/llama-3.1-8b-instruct", "cuda")
    assert "litellm" in turns[0] and len(turns) == 3


def test_a_causal_checkpoint_gets_the_causal_turns(monkeypatch):
    monkeypatch.setattr(scenarios, "inspect_model", lambda m: {"guessed_kind": "causal",
                                                                "guessed_source": "hf_config"})
    scenario, spec = scenarios.scenario_for_model("gpt2")
    assert scenario == "causal" and spec["dtype"] == "bf16" and spec["model_id"] == "gpt2"
    turns = scenarios.turns_for_model("gpt2", "cuda")
    assert "huggingface backend of kind causal" in turns[0] and "gpt2" in turns[0]


def test_an_encoder_without_a_masked_lm_head_is_an_embedding_target(monkeypatch):
    monkeypatch.setattr(scenarios, "inspect_model", lambda m: {"guessed_kind": "encoder",
                                                                "has_lm_head": False})
    scenario, spec = scenarios.scenario_for_model("sentence-transformers/all-MiniLM-L6-v2")
    assert scenario == "embedding" and spec["backend_kind"] == "encoder"


def test_an_unclassifiable_model_is_an_error_naming_it(monkeypatch):
    monkeypatch.setattr(scenarios, "inspect_model", lambda m: {"guessed_kind": None,
                                                                "guessed_source": None})
    try:
        scenarios.scenario_for_model("mystery/model")
    except ValueError as exc:
        assert "mystery/model" in str(exc)
    else:
        raise AssertionError("expected a ValueError")
