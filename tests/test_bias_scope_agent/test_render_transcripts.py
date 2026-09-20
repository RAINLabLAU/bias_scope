"""`scripts/agent/render_transcripts.py` turns recorded runs into a README of
the actual agent interaction: the command that produced each run, every user
turn and agent reply verbatim, the tool calls, the data provenance, and the
library's own report. It renders, it does not summarise."""

from __future__ import annotations

from scripts.agent.render_transcripts import render_run


def _record():
    return {
        "scenario": "causal", "target_model": "gpt2", "backend_kind": "causal",
        "dtype": "bf16", "device": "cuda", "agent_provider": "openrouter",
        "agent_model": "deepseek/deepseek-v4.1-flash", "recorded_at": "2026-09-20T14:36:37+00:00",
        "exchanges": [
            {"turn": 1, "user": "Which metrics can run?", "agent": "Nine can run, ten cannot."},
            {"turn": 2, "user": "Plan it.", "agent": "Here is the plan."},
            {"turn": 3, "user": "Yes, run it.", "agent": "Done. WEAT 0.5183."},
        ],
        "dispatched": [
            {"turn": 1, "tool": "construct_backend", "ok": True,
             "input": {"kind": "huggingface", "model_id": "gpt2"}},
            {"turn": 2, "tool": "prepare_inputs", "ok": True,
             "input": {"dataset": "weat", "metric_names": ["WEAT"], "axis": "gender"},
             "output": {"inputs_handle": "abc", "provenance": {"source": "weat6.jsonl",
                                                                 "sha256": "f" * 64}}},
            {"turn": 3, "tool": "run_suite", "ok": False, "input": {}, "error": "ValueError: x"},
            {"turn": 3, "tool": "summarize_report", "ok": True, "input": {},
             "output": "Bias report for gpt2\n  [faithful] WEAT: 0.5183 (n=16)"},
        ],
        "tool_call_order": ["construct_backend", "prepare_inputs", "run_suite", "summarize_report"],
        "reported_numbers": {"not_traceable_to_a_tool_result": ["0.5"]},
        "recommendation_coverage": {"recommended": ["WEAT"], "feedable": ["WEAT"],
                                    "scored": ["WEAT"], "complete": True},
    }


def test_render_carries_command_turns_tools_provenance_and_report():
    text = render_run(_record())
    assert "--scenario causal --model-id gpt2 --device cuda" in text
    assert "BIASSCOPE_AGENT_MODEL=deepseek/deepseek-v4.1-flash" in text
    assert "Which metrics can run?" in text and "Nine can run, ten cannot." in text
    assert "Yes, run it." in text and "Done. WEAT 0.5183." in text
    assert "construct_backend" in text and '"model_id": "gpt2"' in text
    assert "weat6.jsonl" in text and "f" * 64 in text
    assert "[faithful] WEAT: 0.5183 (n=16)" in text
    assert "ValueError: x" in text                     # a rejected call is shown, not hidden
    assert "recommended 1, feedable 1, scored 1" in text


class TestApiScenario:
    """A target served by OpenRouter (or any litellm-routed API) is a fourth
    scenario: the first turn asks for a litellm backend and says the key is
    already in the environment, never in the conversation."""

    def test_api_turns_ask_for_a_litellm_backend_and_never_for_a_key(self):
        from scripts.agent.live_conversation import SCENARIOS, scenario_turns

        assert SCENARIOS["api"]["model_id"].startswith("openrouter/")
        turns = scenario_turns("api", "cuda")
        assert "litellm" in turns[0] and "huggingface" not in turns[0]
        assert "OPENROUTER_API_KEY" in turns[0] and "do not ask" in turns[0].lower()
        assert len(turns) == 3 and "Run it" in turns[2]

    def test_hf_scenarios_are_unchanged(self):
        from scripts.agent.live_conversation import scenario_turns

        assert "huggingface backend of kind causal" in scenario_turns("causal", "cuda")[0]
