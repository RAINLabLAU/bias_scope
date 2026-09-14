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
    def __init__(self, responses: List[FakeResponse]):
        self._responses = list(responses)
        self.calls: List[Dict[str, Any]] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._responses.pop(0)


class FakeClient:
    def __init__(self, responses: List[FakeResponse]):
        self.messages = FakeMessages(responses)


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
