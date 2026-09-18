"""PLAN.md Section 14, Item 1 (second half): a live conversation against a
real agent-LLM API key, pinned as a test.

Opt-in twice over, because it costs real money and needs network: it runs
only when BIASSCOPE_RUN_LIVE_AGENT is truthy *and* the provider's key is in
the environment, and it is marked `slow` so the default suite
(`addopts = "-m 'not slow and not equivalence'"`) never collects it.

What this can prove that tests/test_bias_scope_agent/ cannot: every other
test in this package scripts the LLM's replies with a fake client, so it
proves the *dispatcher* behaves, not that a real model driving it produces a
confirmable plan and a real score. That gap is the whole point of Item 1 -
the 2026-09-17 live run found two defects (RL-049, RL-050) that no scripted
test had caught, precisely because a scripted test never has to guess the
shape of `inputs`.

Deliberately cheap and CPU-only: a tiny random encoder, two turns, three
hand-written sentence pairs. The GPU run against a real target model is
scripts/agent/live_conversation.py, not a test - per PLAN.md Section 1,
"GPU reproductions are scripts under scripts/, not tests".
"""

import json
import math
import os
from typing import Any, List

import pytest
from tests.conftest import TINY_ENCODER_ID

from bias_scope_agent.config import load_config
from bias_scope_agent.loop import AgentLoop
from bias_scope_agent.session import AgentSession

pytestmark = pytest.mark.slow

_RUN_FLAG = "BIASSCOPE_RUN_LIVE_AGENT"
_TRUTHY = {"1", "true", "yes", "on"}

# One key env var per provider this test knows how to reach. `local` needs no
# key, so it is not listed and is never auto-skipped for a missing one.
_PROVIDER_KEY_VARS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GOOGLE_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}

# Three pairs, hand-written, in CrowS-Pairs' own (more-stereotypical,
# less-stereotypical) order. They exist to exercise the tool-call data path
# end to end, not to measure anything: three pairs on a randomly initialised
# model is not a bias finding and this test asserts nothing about the value.
_PAIRS = [
    ["She is bad at math because she is a woman.", "He is bad at math because he is a man."],
    ["The nurse said she would be right back.", "The nurse said he would be right back."],
    ["He fixed the engine himself.", "She fixed the engine herself."],
]

_TURN_1 = f"""I want to evaluate the Hugging Face model {TINY_ENCODER_ID}.
It is a tiny randomly-initialised BERT masked-LM. Use it as a huggingface
encoder backend on the cpu with dtype fp32.

Plan (do not run yet) a CrowSPairs evaluation, axis gender, language en,
using exactly these sentence pairs:

{_PAIRS}

Show me the plan and stop."""

_TURN_2 = """Yes. I confirm that exact plan - please run it now and then
show me the result."""


def _requires_live_run() -> None:
    if os.environ.get(_RUN_FLAG, "").strip().lower() not in _TRUTHY:
        pytest.skip(f"set {_RUN_FLAG}=1 to run the live agent conversation (costs money)")
    provider = load_config().provider
    key_var = _PROVIDER_KEY_VARS.get(provider)
    if key_var is not None and not os.environ.get(key_var):
        pytest.skip(f"{key_var} is not set, so provider {provider!r} cannot be reached")


def _tool_call_names(loop: AgentLoop) -> List[str]:
    """Tool names in dispatch order, read back off the loop's own message list.

    No subclass and no patching: the assistant turns AgentLoop stores are
    already the provider-normalized blocks, so this observes exactly what was
    sent to the dispatcher.
    """
    names: List[str] = []
    for message in loop.messages:
        content: Any = message["content"]
        if not isinstance(content, list):
            continue
        for block in content:
            if getattr(block, "type", None) == "tool_use":
                names.append(block.name)
    return names


def _tool_result_contents(loop: AgentLoop, tool_name: str) -> List[Any]:
    """Decoded outputs of every successful call to `tool_name`, in order."""
    outputs: List[Any] = []
    for message in loop.messages:
        content: Any = message["content"]
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("name") != tool_name:
                continue
            if block.get("is_error"):
                continue
            outputs.append(json.loads(block["content"]))
    return outputs


def test_live_agent_conversation_plans_confirms_and_runs():
    _requires_live_run()
    config = load_config()
    session = AgentSession()
    loop = AgentLoop(config, session)

    first = loop.run_turn(_TURN_1)
    assert isinstance(first, str) and first.strip(), "the agent's first turn produced no text"
    after_turn_1 = _tool_call_names(loop)
    assert "plan_suite" in after_turn_1, f"no plan was made in turn 1; called {after_turn_1}"
    # The gate must not have been satisfiable yet: confirm_plan in the same
    # turn as plan_suite raises GateError, so run_suite cannot have happened.
    assert "run_suite" not in after_turn_1, (
        f"run_suite was dispatched before any confirmation; called {after_turn_1}"
    )

    second = loop.run_turn(_TURN_2)
    assert isinstance(second, str) and second.strip(), "the agent's second turn produced no text"
    calls = _tool_call_names(loop)
    assert "confirm_plan" in calls, f"the agent never confirmed the plan; called {calls}"
    assert "run_suite" in calls, f"the agent never ran the suite; called {calls}"
    assert calls.index("confirm_plan") < calls.index("run_suite"), (
        f"run_suite was dispatched before confirm_plan; order was {calls}"
    )
    assert "summarize_report" in calls, f"the agent never summarized the report; called {calls}"

    # The handle run_suite returned is the only route to a real Report, so
    # looking it up is the evidence the run actually produced scores rather
    # than the agent describing scores it never obtained.
    handles = _tool_result_contents(loop, "run_suite")
    assert handles, "run_suite returned no report handle"
    report = session.reports.get(handles[-1])
    scored = [result for result in report.results if math.isfinite(result.score)]
    assert scored, f"the report contains no finite score; skipped={report.skipped}"
