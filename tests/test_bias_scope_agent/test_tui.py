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


class TestScriptedPlayback:
    """The scripted runner (scripts/agent/live_conversation.py) plays the user
    itself. Given a `script`, the app submits each turn when the previous
    reply has arrived, shows the conversation as it happens, and exits with
    the transcript when the script is exhausted - so the recorded run and the
    rendered run are the same run."""

    def test_the_script_is_played_in_order_and_the_app_exits_with_the_transcript(self):
        loop = FakeLoop()
        app = BiasScopeApp(loop, script=["first turn", "second turn"])

        async def scenario():
            async with app.run_test() as pilot:
                for _ in range(40):
                    await pilot.pause(0.1)
                    if app.return_value is not None:
                        break
            return app.return_value

        entries = _run(scenario())
        assert loop.turns == ["first turn", "second turn"]
        users = [text for role, text in entries if role == "You >"]
        replies = [text for role, text in entries if role == "BiasScope>"]
        assert users == ["first turn", "second turn"]
        assert len(replies) == 2 and "second turn" in replies[-1]

    def test_the_runner_records_the_same_exchanges_through_the_app(self, monkeypatch):
        from scripts.agent import live_conversation as lc

        loop = FakeLoop()
        loop.session = type("S", (), {"turn": 0})()
        loop.dispatched = []
        monkeypatch.setattr(lc, "RecordingLoop", lambda config, session: loop)
        record = lc.run_conversation(object(), ["one", "two"], tui=True)
        assert [e["user"] for e in record["exchanges"]] == ["one", "two"]
        assert all("You said" in e["agent"] for e in record["exchanges"])


class TestTypingWorksWithoutClicking:
    """Reported 2026-09-21: 'I cannot add an input to the TUI'. Textual gives
    initial focus to the first focusable widget, and the scrolling transcript
    is one, so keystrokes went to it. The prompt must own focus on start and
    after every reply, and the transcript must not compete for it."""

    def test_the_prompt_has_focus_on_start_and_keys_land_in_it(self):
        app = BiasScopeApp(FakeLoop())

        async def scenario():
            async with app.run_test() as pilot:
                await pilot.pause(0.2)
                focused_at_start = app.focused.id if app.focused else None
                await pilot.press(*"hello")
                return focused_at_start, app.query_one("#prompt").value

        focused, typed = _run(scenario())
        assert focused == "prompt"
        assert typed == "hello"

    def test_focus_returns_to_the_prompt_after_a_reply(self):
        app = BiasScopeApp(FakeLoop())

        async def scenario():
            async with app.run_test() as pilot:
                await pilot.pause(0.2)
                await pilot.press(*"hi", "enter")
                await pilot.pause(0.5)
                await app.workers.wait_for_complete()
                await pilot.pause(0.3)
                await pilot.press(*"again")
                return app.focused.id if app.focused else None, app.query_one("#prompt").value

        focused, typed = _run(scenario())
        assert focused == "prompt" and typed == "again"
