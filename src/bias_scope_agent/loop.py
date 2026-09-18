"""The tool-calling loop: one agent LLM (Claude, GPT, or Gemini - see
providers.py), dispatching tool calls against a single AgentSession.

The confirm-before-run gate (session.py) is enforced here, in
`_dispatch_one`, before the real `run_suite` is ever invoked - not just
stated in the system prompt. Tool implementations are looked up on the
`tools`/`introspection` modules by name at dispatch time (not bound once at
construction), so a test can patch e.g. `bias_scope_agent.loop.tools.run_suite`
and prove the real function was never called when the gate rejects a call.

This loop itself never branches on provider: `self.client` is a provider
adapter (see providers.py) presenting one normalized interface -
`create(system=, messages=) -> NormalizedResponse` - regardless of which LLM
is actually behind it, and `self.messages` stays in that same
provider-agnostic shape between turns.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from bias_scope_agent import introspection, tools
from bias_scope_agent.config import AgentConfig
from bias_scope_agent.providers import build_provider
from bias_scope_agent.session import AgentSession, GateError
from bias_scope_agent.system_prompt import render_system_prompt

_CAUGHT_TOOL_ERRORS = (GateError, KeyError, ValueError, TypeError)


def _text_of(content: Any) -> str:
    return "".join(
        getattr(block, "text", "") for block in content if getattr(block, "type", None) == "text"
    )


class AgentLoop:
    def __init__(self, config: AgentConfig, session: AgentSession, client: Optional[Any] = None):
        self.config = config
        self.session = session
        self.client = client if client is not None else build_provider(config)
        self.messages: List[Dict[str, Any]] = []

    def run_turn(self, user_text: str, *, max_tool_rounds: int = 8) -> str:
        """Run one user turn to completion and return everything the model said.

        Every round's text is collected, not just the last response's. A model
        routinely writes prose in the same message as a tool call - the plan it
        is asking the user to confirm, most importantly - and returning only
        the final response dropped all of it, since cli.py prints nothing else.
        A live run confirmed a plan that had never been displayed as a result
        (REVIEW_LATER.md RL-055).
        """
        self.session.advance_turn()
        self.messages.append({"role": "user", "content": user_text})
        said: List[str] = []
        for _ in range(max_tool_rounds):
            response = self.client.create(
                system=render_system_prompt(self.session), messages=self.messages
            )
            self.messages.append({"role": "assistant", "content": response.content})
            text = _text_of(response.content).strip()
            if text:
                said.append(text)
            if response.stop_reason != "tool_use":
                return "\n\n".join(said)
            results = self._dispatch_tools(response.content)
            self.messages.append({"role": "user", "content": results})
        said.append("(stopped after too many tool calls in one turn)")
        return "\n\n".join(said)

    def _dispatch_one(self, block: Any) -> Any:
        if block.name == "inspect_model":
            return introspection.inspect_model(**block.input)
        if block.name == "run_suite":
            self.session.check_run_gate(
                block.input["backend_handle"],
                block.input["metric_names"],
                block.input.get("axis", "gender"),
                block.input.get("language", "en"),
            )
        fn = getattr(tools, block.name)
        return fn(self.session, **block.input)

    def _dispatch_tools(self, blocks: Any) -> List[Dict[str, Any]]:
        results = []
        for block in blocks:
            if block.type != "tool_use":
                continue
            try:
                output = self._dispatch_one(block)
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "name": block.name,
                        "content": json.dumps(output),
                    }
                )
            except _CAUGHT_TOOL_ERRORS as exc:
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "name": block.name,
                        "content": str(exc),
                        "is_error": True,
                    }
                )
        return results
