"""Scripted-conversation tests for AgentLoop, using a fake Anthropic client
(no real API call). Prove two things the brief requires: run_suite is never
successfully dispatched before a plan_suite result exists in the transcript,
and the gate actually stops execution (a spy on tools.run_suite is never
invoked when the script tries to jump the gate), not just that the log looks
tidy in the happy path.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List
from unittest.mock import patch

from bias_scope_agent.config import AgentConfig
from bias_scope_agent.loop import AgentLoop
from bias_scope_agent.session import AgentSession


@dataclass
class FakeToolUseBlock:
    type: str
    name: str = ""
    id: str = ""
    input: Dict[str, Any] = field(default_factory=dict)
    text: str = ""


@dataclass
class FakeResponse:
    content: List[FakeToolUseBlock]
    stop_reason: str


class FakeMessages:
    """Records calls in the shape older assertions expect (client.messages.calls
    with "system"/"messages" keys), while FakeClient.create is the actual
    entry point AgentLoop calls - matching the provider-agnostic interface
    every real Provider adapter implements (see providers.py)."""

    def __init__(self, responses: List[FakeResponse]):
        self._responses = list(responses)
        self.calls: List[Dict[str, Any]] = []

    def create(self, *, system, messages):
        self.calls.append({"system": system, "messages": messages})
        return self._responses.pop(0)


class FakeClient:
    def __init__(self, responses: List[FakeResponse]):
        self.messages = FakeMessages(responses)

    def create(self, *, system, messages):
        return self.messages.create(system=system, messages=messages)


def text_block(text: str) -> FakeToolUseBlock:
    return FakeToolUseBlock(type="text", text=text)


def tool_use(name: str, tool_input: Dict[str, Any], call_id: str) -> FakeToolUseBlock:
    return FakeToolUseBlock(type="tool_use", name=name, id=call_id, input=tool_input)


def _happy_path_client() -> FakeClient:
    handle_placeholder = "handle-will-be-filled-in"
    return FakeClient(
        [
            # Turn 1: construct backend, recommend + explain, plan.
            FakeResponse(
                content=[
                    tool_use(
                        "construct_backend",
                        {"kind": "litellm", "model_id": "gpt-4o-mini"},
                        "call-1",
                    )
                ],
                stop_reason="tool_use",
            ),
            FakeResponse(
                content=[
                    tool_use(
                        "recommend_metrics_tool", {"backend_handle": handle_placeholder}, "call-2"
                    ),
                    tool_use(
                        "explain_exclusions_tool", {"backend_handle": handle_placeholder}, "call-3"
                    ),
                ],
                stop_reason="tool_use",
            ),
            FakeResponse(
                content=[
                    tool_use(
                        "plan_suite",
                        {"backend_handle": handle_placeholder, "metric_names": ["BOLD"]},
                        "call-4",
                    )
                ],
                stop_reason="tool_use",
            ),
            FakeResponse(
                content=[text_block("Here is the plan. Shall I run it?")],
                stop_reason="end_turn",
            ),
            # Turn 2 (user says "yes"): confirm, run, summarize.
            FakeResponse(
                content=[tool_use("confirm_plan", {"plan_id": "plan-placeholder"}, "call-5")],
                stop_reason="tool_use",
            ),
            FakeResponse(
                content=[
                    tool_use(
                        "run_suite",
                        {
                            "backend_handle": handle_placeholder,
                            "metric_names": ["BOLD"],
                            "inputs": {},
                        },
                        "call-6",
                    )
                ],
                stop_reason="tool_use",
            ),
            FakeResponse(
                content=[
                    tool_use(
                        "summarize_report", {"report_handle": "report-placeholder"}, "call-7"
                    )
                ],
                stop_reason="tool_use",
            ),
            FakeResponse(
                content=[text_block("Here is your report.")],
                stop_reason="end_turn",
            ),
        ]
    )


class _PatchingDispatchLoop(AgentLoop):
    """Rewrites placeholder handles/plan ids in the scripted tool_use blocks
    with the real values the harness generated, so the fake script doesn't
    need to predict uuids. Only used by these tests."""

    _PLACEHOLDERS = {
        "handle-will-be-filled-in": "_last_backend_handle",
        "plan-placeholder": "_last_plan_id",
        "report-placeholder": "_last_report_handle",
    }

    def _rewrite_placeholders(self, blocks):
        for block in blocks:
            if block.type != "tool_use":
                continue
            for key, value in list(block.input.items()):
                if not isinstance(value, str):
                    continue
                attr_name = self._PLACEHOLDERS.get(value)
                if attr_name and getattr(self, attr_name):
                    block.input[key] = getattr(self, attr_name)

    def _capture_handle(self, block, result):
        if result.get("is_error"):
            return
        content = json.loads(result["content"])
        if block.name == "construct_backend":
            self._last_backend_handle = content
        elif block.name == "plan_suite":
            self._last_plan_id = content["plan_id"]
        elif block.name == "run_suite":
            self._last_report_handle = content

    def _dispatch_tools(self, blocks):
        self._rewrite_placeholders(blocks)
        results = super()._dispatch_tools(blocks)
        tool_use_blocks = [b for b in blocks if b.type == "tool_use"]
        for block, result in zip(tool_use_blocks, results):
            self._capture_handle(block, result)
        return results

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_backend_handle = None
        self._last_plan_id = None
        self._last_report_handle = None


class TestScriptedConversationHappyPath:
    def test_run_suite_never_dispatched_before_plan_suite_appears(self):
        client = _happy_path_client()
        loop = _PatchingDispatchLoop(AgentConfig(), AgentSession(), client=client)

        loop.run_turn("I want to test gpt-4o-mini for gender bias.")
        loop.run_turn("yes, run it")

        # The last call's messages carries the fullest single transcript.
        last_transcript = client.messages.calls[-1]["messages"]
        tool_names_in_order = [
            block.name
            for message in last_transcript
            if isinstance(message.get("content"), list)
            for block in message["content"]
            if isinstance(block, FakeToolUseBlock) and block.type == "tool_use"
        ]
        assert "plan_suite" in tool_names_in_order
        assert "run_suite" in tool_names_in_order
        assert tool_names_in_order.index("plan_suite") < tool_names_in_order.index("run_suite")

    def test_report_is_produced_and_summarized(self):
        client = _happy_path_client()
        loop = _PatchingDispatchLoop(AgentConfig(), AgentSession(), client=client)
        loop.run_turn("I want to test gpt-4o-mini for gender bias.")
        final_text = loop.run_turn("yes, run it")
        assert final_text == "Here is your report."


class TestGateActuallyBlocksExecution:
    def test_early_run_suite_never_invokes_the_real_function(self):
        client = FakeClient(
            [
                FakeResponse(
                    content=[
                        tool_use(
                            "construct_backend",
                            {"kind": "litellm", "model_id": "gpt-4o-mini"},
                            "call-1",
                        )
                    ],
                    stop_reason="tool_use",
                ),
                FakeResponse(
                    # Doctored: attempts run_suite with no prior plan_suite/confirm_plan at all.
                    content=[
                        tool_use(
                            "run_suite",
                            {
                                "backend_handle": "handle-will-be-filled-in",
                                "metric_names": ["BOLD"],
                                "inputs": {},
                            },
                            "call-2",
                        )
                    ],
                    stop_reason="tool_use",
                ),
                FakeResponse(
                    content=[text_block("(should not get here successfully)")],
                    stop_reason="end_turn",
                ),
            ]
        )
        loop = _PatchingDispatchLoop(AgentConfig(), AgentSession(), client=client)

        with patch("bias_scope_agent.loop.tools.run_suite") as spy:
            loop.run_turn("run BOLD on gpt-4o-mini right now")

        spy.assert_not_called()

    def test_early_run_suite_tool_result_is_marked_as_error(self):
        client = FakeClient(
            [
                FakeResponse(
                    content=[
                        tool_use(
                            "construct_backend",
                            {"kind": "litellm", "model_id": "gpt-4o-mini"},
                            "call-1",
                        )
                    ],
                    stop_reason="tool_use",
                ),
                FakeResponse(
                    content=[
                        tool_use(
                            "run_suite",
                            {
                                "backend_handle": "handle-will-be-filled-in",
                                "metric_names": ["BOLD"],
                                "inputs": {},
                            },
                            "call-2",
                        )
                    ],
                    stop_reason="tool_use",
                ),
                FakeResponse(content=[text_block("done")], stop_reason="end_turn"),
            ]
        )
        loop = _PatchingDispatchLoop(AgentConfig(), AgentSession(), client=client)
        loop.run_turn("run BOLD on gpt-4o-mini right now")

        second_call_messages = client.messages.calls[-1]["messages"]
        tool_results = [
            block
            for message in second_call_messages
            if isinstance(message.get("content"), list)
            for block in message["content"]
            if isinstance(block, dict) and block.get("type") == "tool_result"
        ]
        run_suite_results = [r for r in tool_results if r.get("is_error")]
        assert run_suite_results


class TestRecordedFactsReachLaterTurns:
    """Item 5 of the follow-up plan: verify the memory mechanism actually
    plumbs through AgentLoop, not just that render_system_prompt() works in
    isolation (already covered in test_schemas_and_prompt.py).

    What this scripted test can and cannot prove: because the client here is
    a canned response queue rather than a real LLM, it cannot demonstrate
    that a *real* Claude would actually refrain from re-asking a question it
    has already been answered - that requires a live conversation (Item 1).
    What it *can* prove, mechanically: once record_fact has been called, the
    fact is present in the `system` prompt string handed to
    `client.messages.create` on every subsequent turn - i.e. the plumbing
    this behavior depends on is wired correctly end to end through the real
    AgentLoop, not just through the standalone render_system_prompt().
    """

    def test_fact_recorded_in_turn_two_is_present_in_turn_threes_system_prompt(self):
        client = FakeClient(
            [
                # Turn 1: inspect_model comes back low-confidence (identifier
                # is bogus; live=False keeps this deterministic, no network),
                # so the agent asks the user directly instead of guessing.
                FakeResponse(
                    content=[
                        tool_use(
                            "inspect_model",
                            {"identifier": "some-ambiguous-model", "live": False},
                            "call-1",
                        )
                    ],
                    stop_reason="tool_use",
                ),
                FakeResponse(
                    content=[text_block("Is this a causal or an encoder model?")],
                    stop_reason="end_turn",
                ),
                # Turn 2: user answers; agent records the fact.
                FakeResponse(
                    content=[
                        tool_use(
                            "record_fact",
                            {"key": "model_kind", "value": "causal"},
                            "call-2",
                        )
                    ],
                    stop_reason="tool_use",
                ),
                FakeResponse(
                    content=[text_block("Got it, thanks.")],
                    stop_reason="end_turn",
                ),
                # Turn 3: a well-behaved script does not re-ask or re-inspect -
                # this file cannot prove a *real* LLM would do the same, only
                # that the fact was available to it in the system prompt.
                FakeResponse(
                    content=[text_block("Sure, what would you like to do next?")],
                    stop_reason="end_turn",
                ),
            ]
        )
        loop = AgentLoop(AgentConfig(), AgentSession(), client=client)

        loop.run_turn("I have a model called some-ambiguous-model.")
        loop.run_turn("it's causal")
        loop.run_turn("okay, what metrics can I run on it?")

        third_turn_system_prompt = client.messages.calls[-1]["system"]
        assert "model_kind" in third_turn_system_prompt
        assert "causal" in third_turn_system_prompt

    def test_fact_is_absent_from_the_very_first_turns_system_prompt(self):
        # Sanity check for the assertion above: the fact must not simply be
        # present in every prompt regardless of whether record_fact ran.
        client = FakeClient(
            [
                FakeResponse(
                    content=[text_block("Hi, tell me about your model.")],
                    stop_reason="end_turn",
                )
            ]
        )
        loop = AgentLoop(AgentConfig(), AgentSession(), client=client)
        loop.run_turn("hello")
        first_turn_system_prompt = client.messages.calls[0]["system"]
        assert "model_kind" not in first_turn_system_prompt


class TestTextWrittenAlongsideAToolCallReachesTheUser:
    """RL-055: `run_turn` returned only the *final* response's text.

    Any prose the model wrote in the same message as a tool call was appended
    to `self.messages` and then dropped, and `cli.py` prints only the return
    value - so it never reached the user. Observed live: a real agent rendered
    the plan alongside its `plan_suite` call, then said "I've shown you the
    plan above. Please reply confirming" in a 315-character turn that contained
    no plan. The scripted user confirmed a plan that was never displayed.

    That matters because the confirm-before-run gate rests on three premises -
    a plan was shown, a turn boundary passed, confirm_plan was called. The last
    two are enforced in session.py. This defect made the first one routinely
    false, so the gate could be satisfied with nothing ever shown.
    """

    def _client_that_narrates_then_calls_a_tool(self) -> FakeClient:
        return FakeClient(
            [
                FakeResponse(
                    content=[
                        text_block("HERE IS THE PLAN THE USER MUST REVIEW"),
                        tool_use("record_fact", {"key": "k", "value": "v"}, "call-1"),
                    ],
                    stop_reason="tool_use",
                ),
                FakeResponse(
                    content=[text_block("I've shown you the plan above.")],
                    stop_reason="end_turn",
                ),
            ]
        )

    def test_intermediate_text_is_returned_not_discarded(self):
        loop = AgentLoop(
            AgentConfig(), AgentSession(), client=self._client_that_narrates_then_calls_a_tool()
        )
        returned = loop.run_turn("go")
        assert "HERE IS THE PLAN THE USER MUST REVIEW" in returned
        assert "I've shown you the plan above." in returned

    def test_the_final_text_still_comes_last(self):
        loop = AgentLoop(
            AgentConfig(), AgentSession(), client=self._client_that_narrates_then_calls_a_tool()
        )
        returned = loop.run_turn("go")
        assert returned.index("HERE IS THE PLAN") < returned.index("I've shown you")

    def test_a_turn_with_no_tool_calls_is_unchanged(self):
        client = FakeClient(
            [FakeResponse(content=[text_block("just an answer")], stop_reason="end_turn")]
        )
        assert AgentLoop(AgentConfig(), AgentSession(), client=client).run_turn("go") == (
            "just an answer"
        )
