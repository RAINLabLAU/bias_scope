"""The Textual front end for the agent: `You >` on one side, `BiasScope>` on
the other, the agent's Markdown rendered rather than printed raw, and the
tools it calls shown as it calls them. Driven headless with a fake loop, so
no LLM, model or terminal is needed.
"""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("textual")

from bias_scope_agent.tui import BiasScopeApp  # noqa: E402


class FakeLoop:
    """Stands in for AgentLoop: echoes Markdown and reports two tool calls."""

    def __init__(self):
        self.turns = []
        self.on_tool = None

    def run_turn(self, user_text):
        self.turns.append(user_text)
        if self.on_tool:
            self.on_tool("construct_backend", {"model_id": "gpt2"})
            self.on_tool("recommend_metrics_tool", {})
        table = "| metric | score |\n|---|---|\n| WEAT | 0.4 |"
        return f"Here is a **table**:\n\n{table}\n\nYou said: {user_text}"


def _run(coro):
    return asyncio.run(coro)


def test_a_turn_shows_the_user_line_the_tool_calls_and_the_rendered_reply():
    loop = FakeLoop()
    app = BiasScopeApp(loop)

    async def scenario():
        async with app.run_test() as pilot:
            await pilot.click("#prompt")
            await pilot.press(*"measure gpt2", "enter")
            await pilot.pause(0.5)
            await app.workers.wait_for_complete()
            await pilot.pause(0.2)
            return app.transcript()

    entries = _run(scenario())
    roles = [role for role, _ in entries]
    assert roles[0] == "You >"
    assert "construct_backend" in " ".join(text for role, text in entries if role == "tool")
    assert roles[-1] == "BiasScope>"
    assert "You said: measure gpt2" in entries[-1][1]
    assert loop.turns == ["measure gpt2"]


def test_the_reply_is_rendered_markdown_not_raw_text():
    from textual.widgets import Markdown

    app = BiasScopeApp(FakeLoop())

    async def scenario():
        async with app.run_test() as pilot:
            await pilot.click("#prompt")
            await pilot.press(*"hi", "enter")
            await pilot.pause(0.5)
            await app.workers.wait_for_complete()
            await pilot.pause(0.2)
            return [w for w in app.query(Markdown)]

    widgets = _run(scenario())
    assert widgets, "the agent's reply must be a Markdown widget, not a plain label"


def test_quit_words_are_not_sent_to_the_agent():
    loop = FakeLoop()
    app = BiasScopeApp(loop)

    async def scenario():
        async with app.run_test() as pilot:
            await pilot.click("#prompt")
            await pilot.press(*"exit", "enter")
            await pilot.pause(0.2)

    _run(scenario())
    assert loop.turns == []
