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


class TestATypedConversationCanBeRecordedLikeAScriptedOne:
    """`live_conversation.py --interactive`: you type the turns in the UI, and
    on exit the same transcript file is written as for a scripted run."""

    def test_entries_become_exchanges_in_turn_order(self):
        from scripts.agent.live_conversation import exchanges_from_entries

        entries = [("You >", "set up gpt2"), ("tool", "  · construct_backend"),
                   ("BiasScope>", "Ready."), ("You >", "plan it"), ("BiasScope>", "Plan.")]
        assert exchanges_from_entries(entries) == [
            {"turn": 1, "user": "set up gpt2", "agent": "Ready."},
            {"turn": 2, "user": "plan it", "agent": "Plan."},
        ]

    def test_an_unanswered_last_turn_is_kept_with_an_empty_reply(self):
        from scripts.agent.live_conversation import exchanges_from_entries

        entries = [("You >", "hi"), ("BiasScope>", "hello"), ("You >", "bye")]
        assert exchanges_from_entries(entries)[-1] == {"turn": 2, "user": "bye", "agent": ""}


class TestKeyboardShortcutsToLeave:
    """Escape, Ctrl-Q and Ctrl-C all leave, even while the cursor is in the
    input line, and hand back the transcript."""

    @pytest.mark.parametrize("key", ["escape", "ctrl+q", "ctrl+c"])
    def test_the_key_exits_with_the_transcript(self, key):
        app = BiasScopeApp(FakeLoop())

        async def scenario():
            async with app.run_test() as pilot:
                await pilot.pause(0.2)
                await pilot.press(*"half-typed")
                await pilot.press(key)
                await pilot.pause(0.3)
                return app.return_value

        assert _run(scenario()) == []          # left before any turn was sent

    def test_the_footer_names_a_way_out(self):
        app = BiasScopeApp(FakeLoop())

        async def scenario():
            async with app.run_test() as pilot:
                await pilot.pause(0.3)
                return [b.binding.key for b in app.active_bindings.values()]

        keys = _run(scenario())
        assert "escape" in keys and "ctrl+q" in keys


class _FakeInfo:
    def __init__(self, family, fidelity):
        self.family, self.fidelity = family, fidelity


class _FakeResult:
    def __init__(self, metric, score, n, family="embedding", fidelity="faithful", deviation=""):
        self.metric, self.score, self.n, self.info = metric, score, n, _FakeInfo(family, fidelity)
        self.protocol = {"resources": [{"name": "x", "deviation": deviation}] if deviation else []}


class _FakeReport:
    model_id = "gpt2"
    results = [_FakeResult("WEAT", 0.4006, 16), _FakeResult("EMT", 0.053, 25, "generated_text",
                                                             "faithful", "local classifier")]
    skipped = {"SEAT": "declined: zero variance"}


class ReportingLoop(FakeLoop):
    """A loop whose turn ends with summarize_report, like a real evaluation."""

    def __init__(self):
        super().__init__()
        self.session = type("S", (), {})()
        self.session.reports = type("R", (), {"get": staticmethod(lambda handle: _FakeReport())})()

    def run_turn(self, user_text):
        self.turns.append(user_text)
        if self.on_tool:
            self.on_tool("run_suite", {})
            self.on_tool("summarize_report", {"report_handle": "h1"})
        return "Done, see the report."


class TestResultsTableAndRerunPrompt:
    """When a turn ends with summarize_report, the TUI shows the report as a
    table (metric, family, score, n, fidelity, deviation; skipped metrics with
    their reason) and then asks whether to run another bias test. Yes starts a
    fresh loop and a clean transcript; no leaves."""

    def _drive(self, app, *keys):
        async def scenario():
            async with app.run_test() as pilot:
                await pilot.pause(0.2)
                await pilot.press(*"run it", "enter")
                await pilot.pause(0.5)
                await app.workers.wait_for_complete()
                await pilot.pause(0.4)
                state = {"rows": app.results_rows(),
                         "question": app.query_one("#prompt").placeholder}
                for key in keys:
                    await pilot.press(*key, "enter") if len(key) > 1 else await pilot.press(key)
                    await pilot.pause(0.4)
                state["after"] = app.transcript()
                state["placeholder_after"] = app.query_one("#prompt").placeholder
                state["return"] = app.return_value
                return state
        return _run(scenario())

    def test_the_report_is_rendered_as_a_table_with_skips(self):
        state = self._drive(BiasScopeApp(ReportingLoop()))
        rows = state["rows"]
        assert ("WEAT", "embedding", "0.4006", "16", "faithful", "") in rows
        assert ("EMT", "generated_text", "0.053", "25", "faithful", "local classifier") in rows
        assert ("SEAT", "", "skipped", "", "", "declined: zero variance") in rows
        assert "another" in state["question"].lower()

    def test_yes_starts_a_fresh_loop_and_a_clean_transcript(self):
        made = []

        def make_loop():
            made.append(ReportingLoop())
            return made[-1]

        app = BiasScopeApp(make_loop(), make_loop=make_loop)
        state = self._drive(app, "yes")
        assert len(made) == 2 and app.loop is made[1]
        assert state["after"] == []                       # transcript cleared for the new run
        assert "another" not in state["placeholder_after"].lower()

    def test_no_leaves_with_the_transcript(self):
        app = BiasScopeApp(ReportingLoop(), make_loop=ReportingLoop)
        state = self._drive(app, "no")
        assert state["return"] is not None
        assert [r for r, _ in state["return"]][-1] == "BiasScope>"


class TestAutonomousMode:
    """Asks for a model and nothing else. Plays the three scripted turns for
    that model (the plan is confirmed on the user's behalf), shows the
    results table, then asks for the next model; 'no' leaves. Each run's
    transcript is kept under `runs` for the caller to record."""

    def _app(self, made):
        def make_loop():
            made.append(ReportingLoop())
            return made[-1]

        return BiasScopeApp(make_loop(), make_loop=make_loop, autonomous=True,
                            turns_for=lambda model: [f"set up {model}", "plan it", "run it"])

    def test_it_asks_for_a_model_then_runs_the_script_and_asks_again(self):
        made = []
        app = self._app(made)

        async def scenario():
            async with app.run_test() as pilot:
                await pilot.pause(0.3)
                first_question = app.query_one("#prompt").placeholder
                await pilot.press(*"gpt2", "enter")
                for _ in range(60):
                    await pilot.pause(0.1)
                    if app.runs:
                        break
                await pilot.pause(0.3)
                return first_question, app.query_one("#prompt").placeholder, app.results_rows()

        first, again, rows = _run(scenario())
        assert "model" in first.lower()
        assert made[0].turns == ["set up gpt2", "plan it", "run it"]
        assert app.runs[0][0] == "gpt2"
        users = [text for role, text in app.runs[0][1] if role == "You >"]
        assert users == ["set up gpt2", "plan it", "run it"]
        assert rows and rows[0][0] == "WEAT"
        assert "another" in again.lower() or "model" in again.lower()

    def test_the_next_model_gets_a_fresh_loop_and_no_leaves(self):
        made = []
        app = self._app(made)

        async def scenario():
            async with app.run_test() as pilot:
                await pilot.pause(0.3)
                await pilot.press(*"gpt2", "enter")
                for _ in range(60):
                    await pilot.pause(0.1)
                    if len(app.runs) == 1:
                        break
                await pilot.press(*"bert-base-uncased", "enter")
                for _ in range(60):
                    await pilot.pause(0.1)
                    if len(app.runs) == 2:
                        break
                await pilot.press(*"no", "enter")
                await pilot.pause(0.3)
                return app.return_value

        _run(scenario())
        assert len(made) == 2 and made[1].turns[0] == "set up bert-base-uncased"
        assert [m for m, _ in app.runs] == ["gpt2", "bert-base-uncased"]


class TestTheRunnerDefaultsToInteractiveAndHasAnAutonomousMode:
    """2026-09-21: 'let the interactive the default one' and 'one option
    fully autonomous where it just asks the user for the model ... and when it
    ends the tests it re-asks for a new model'. `--scenario` is now opt-in;
    `--autonomous` records one transcript per model the user names."""

    def test_no_flags_means_interactive_and_scenario_means_scripted(self):
        from scripts.agent import live_conversation as lc

        assert lc.mode_of(lc.parse_args([])) == "interactive"
        assert lc.mode_of(lc.parse_args(["--scenario", "encoder"])) == "scripted"
        assert lc.mode_of(lc.parse_args(["--autonomous"])) == "autonomous"

    def test_each_autonomous_run_becomes_its_own_record(self):
        from scripts.agent import live_conversation as lc

        first, second = FakeLoop(), FakeLoop()
        first.dispatched = [{"tool": "construct_backend",
                             "input": {"model_id": "gpt2"}, "ok": True}]
        second.dispatched = [{"tool": "construct_backend",
                              "input": {"model_id": "bert-base-uncased"}, "ok": True}]
        runs = [("gpt2", [("You >", "set up gpt2"), ("BiasScope>", "done")]),
                ("bert-base-uncased", [("You >", "set up bert"), ("BiasScope>", "done too")])]
        records = lc.records_from_runs(runs, [first, second])
        assert [r["target_model"] for r in records] == ["gpt2", "bert-base-uncased"]
        assert records[0]["exchanges"] == [{"turn": 1, "user": "set up gpt2", "agent": "done"}]
        assert records[1]["dispatched"] is second.dispatched

    def test_the_cli_autonomous_flag_reaches_the_app(self, monkeypatch):
        from bias_scope_agent import cli, tui

        seen = {}

        def fake_run_autonomous(loop, make_loop, turns_for):
            seen["turns_for"] = turns_for
            seen["fresh"] = make_loop()
            return []

        monkeypatch.setattr(tui, "run_autonomous", fake_run_autonomous)
        monkeypatch.setattr(cli, "_tui_available", lambda: True)
        monkeypatch.setattr(cli, "load_config", lambda: object())
        monkeypatch.setattr(cli, "AgentLoop", lambda config, session: FakeLoop())
        assert cli.main(["--autonomous", "--device", "cpu"]) == 0
        assert callable(seen["turns_for"]) and isinstance(seen["fresh"], FakeLoop)
