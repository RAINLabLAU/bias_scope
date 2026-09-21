"""A terminal UI for the agent, built on Textual.

The plain REPL prints the agent's Markdown as raw text. Here each turn is a
pair of blocks: `You >` with what was typed, and `BiasScope>` with the reply
rendered by Textual's Markdown widget, so tables, bold and lists come out as
formatted text. While the agent works, the tools it calls appear as dim lines
under the user's turn, in the order they are dispatched - the same dispatch
log the scripted runner records, shown live.

Launched by `bias-scope-agent` when it runs in a terminal; `--plain` keeps the
old REPL. Nothing here changes what the agent does: the same AgentLoop, the
same gate, the same tools.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.widgets import DataTable, Footer, Header, Input, Markdown, Static

_QUIT_WORDS = {"exit", "quit"}
_YES = {"y", "yes", "sure", "ok", "again"}
_NO = {"n", "no", "exit", "quit", "done"}
_PROMPT_HINT = "You >  ask about a model, ask for a plan, confirm it  (Esc quits)"
_RERUN_QUESTION = "Run another bias test on a model? (yes / no)"
_MODEL_QUESTION = ("Which model should I evaluate? (a Hugging Face id, or "
                   "openrouter/<vendor>/<model>; 'no' to finish)")
_RESULT_COLUMNS = ("metric", "family", "score", "n", "fidelity", "deviation")


class BiasScopeApp(App):
    """One scrolling transcript and one input line."""

    TITLE = "BiasScope agent"
    CSS = """
    Screen { background: $surface; }
    Header { background: #1d4ed8; color: #f8fafc; text-style: bold; }
    #log { padding: 1 2; }
    .you { color: #38bdf8; text-style: bold; margin-top: 1; }
    .you-text { color: #e0f2fe; margin-left: 2; }
    .agent { color: #4ade80; text-style: bold; margin-top: 1; }
    .tool { color: #fbbf24; margin-left: 2; }
    Markdown { margin-left: 2; }
    DataTable { margin-left: 2; margin-bottom: 1; height: auto; }
    #prompt { dock: bottom; border: tall #38bdf8; }
    #prompt:focus { border: tall #4ade80; }
    Footer { background: #1e293b; }
    """
    # priority=True: the keys work while the cursor is in the input line.
    BINDINGS = [
        Binding("escape", "quit", "Quit", priority=True),
        Binding("ctrl+q", "quit", "Quit", priority=True),
        Binding("ctrl+c", "quit", "Quit", priority=True, show=False),
    ]
    # Textual focuses the first focusable widget on start; without this the
    # scrolling transcript took the keystrokes and nothing could be typed.
    AUTO_FOCUS = "#prompt"

    def __init__(
        self,
        loop: Any,
        script: Optional[Sequence[str]] = None,
        make_loop: Optional[Callable[[], Any]] = None,
        autonomous: bool = False,
        turns_for: Optional[Callable[[str], List[str]]] = None,
    ) -> None:
        """`script`: turns to play automatically, one after each reply, after
        which the app exits with its transcript (the scripted runner's mode).
        Without it the user types the turns. `make_loop` builds a fresh loop
        when the user asks to evaluate another model after a report.
        `autonomous`: ask only for a model id, play `turns_for(model)` (the
        plan is confirmed on the user's behalf), show the table, ask for the
        next model; every finished run is kept in `runs` as (model, entries)."""
        super().__init__()
        self.loop = loop
        self.make_loop = make_loop
        self._autonomous = autonomous
        self._turns_for = turns_for
        self._awaiting_model = False
        self._current_model: Optional[str] = None
        self.runs: List[Tuple[str, List[Tuple[str, str]]]] = []
        self._entries: List[Tuple[str, str]] = []
        self._script: List[str] = list(script or [])
        self._scripted = bool(script)
        self._report_handle: Optional[str] = None
        self._results_rows: List[Tuple[str, ...]] = []
        self._awaiting_rerun = False
        self._attach(loop)

    def _attach(self, loop: Any) -> None:
        if hasattr(loop, "on_tool"):
            loop.on_tool = self._tool_called_in_worker

    def results_rows(self) -> List[Tuple[str, ...]]:
        """The last report as table rows (metric, family, score, n, fidelity, deviation)."""
        return list(self._results_rows)

    def on_mount(self) -> None:
        prompt = self.query_one("#prompt", Input)
        if self._scripted:
            prompt.disabled = True
            self.call_later(self._play_next)
        elif self._autonomous:
            self._ask_model()
        else:
            prompt.focus()

    def _ask_model(self) -> None:
        self._awaiting_model = True
        prompt = self.query_one("#prompt", Input)
        prompt.placeholder = _MODEL_QUESTION
        prompt.disabled = False
        prompt.focus()

    async def _start_model(self, model_id: str) -> None:
        """Autonomous mode: a fresh loop for this model, then its script."""
        self._awaiting_model = False
        if self.runs and self.make_loop is not None:
            self.loop = self.make_loop()
            self._attach(self.loop)
            await self.query_one("#log", VerticalScroll).remove_children()
            self._entries = []
            self._results_rows = []
        self._current_model = model_id
        try:
            self._script = list(self._turns_for(model_id)) if self._turns_for else []
        except Exception as exc:  # noqa: BLE001 - shown, then ask again
            log = self.query_one("#log", VerticalScroll)
            await log.mount(Static(Text(f"  {type(exc).__name__}: {exc}"), classes="tool"))
            self._ask_model()
            return
        self.query_one("#prompt", Input).disabled = True
        await self._play_next()

    async def _play_next(self) -> None:
        if self._script:
            await self._submit(self._script.pop(0))

    def compose(self) -> ComposeResult:
        yield Header()
        yield VerticalScroll(id="log", can_focus=False)
        yield Input(placeholder=_PROMPT_HINT, id="prompt")
        yield Footer()

    def action_quit(self) -> None:
        """Ctrl-C: leave, handing the transcript to whoever ran the app."""
        self.exit(self.transcript())

    def transcript(self) -> List[Tuple[str, str]]:
        """(role, text) per block: 'You >', 'tool' or 'BiasScope>'."""
        return list(self._entries)

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        event.input.value = ""
        if not text:
            return
        if self._awaiting_model:
            if text.lower() in _NO:
                self.exit(self.transcript())
            else:
                await self._start_model(text)
            return
        if self._awaiting_rerun:
            await self._answer_rerun(text.lower())
            return
        if text.lower() in _QUIT_WORDS:
            self.exit(self.transcript())
            return
        await self._submit(text)

    async def _answer_rerun(self, answer: str) -> None:
        if answer in _YES:
            await self._restart()
        elif answer in _NO:
            self.exit(self.transcript())
        # anything else: keep asking

    async def _restart(self) -> None:
        """A fresh loop (new session, no backends, no plan) and a clean screen."""
        if self.make_loop is not None:
            self.loop = self.make_loop()
            self._attach(self.loop)
        await self.query_one("#log", VerticalScroll).remove_children()
        self._entries = []
        self._results_rows = []
        self._awaiting_rerun = False
        prompt = self.query_one("#prompt", Input)
        prompt.placeholder = _PROMPT_HINT
        prompt.disabled = False
        prompt.focus()

    async def _submit(self, text: str) -> None:
        log = self.query_one("#log", VerticalScroll)
        await log.mount(Static("You >", classes="you"), Static(Text(text), classes="you-text"))
        self._entries.append(("You >", text))
        self.query_one("#prompt", Input).disabled = True
        self.sub_title = "BiasScope is working..."
        log.scroll_end(animate=False)
        self.ask(text)

    @work(thread=True, exclusive=True)
    def ask(self, text: str) -> None:
        """The agent turn runs off the UI thread: it calls an API and may
        load a model; the UI must keep painting meanwhile."""
        try:
            reply = self.loop.run_turn(text)
        except Exception as exc:  # noqa: BLE001 - shown to the user, not swallowed
            reply = f"**The agent failed:** `{type(exc).__name__}: {exc}`"
        self.call_from_thread(self._show_reply, reply)

    def _tool_called_in_worker(self, name: str, arguments: Dict[str, Any]) -> None:
        if name == "summarize_report":
            self._report_handle = arguments.get("report_handle")
        self.call_from_thread(self._show_tool, name, arguments)

    def _show_results_table(self, handle: str) -> None:
        """The report behind `summarize_report`, as a table under the reply."""
        try:
            report = self.loop.session.reports.get(handle)
        except Exception:  # noqa: BLE001 - a stale handle just means no table
            return
        rows: List[Tuple[str, ...]] = []
        for result in report.results:
            deviations = [r.get("deviation", "") for r in result.protocol.get("resources", [])]
            rows.append((result.metric, result.info.family, f"{result.score:.4g}",
                         str(result.n), result.info.fidelity,
                         "; ".join(d for d in deviations if d)))
        for name, reason in report.skipped.items():
            rows.append((name, "", "skipped", "", "", reason))
        self._results_rows = rows
        table = DataTable(zebra_stripes=True)
        table.add_columns(*_RESULT_COLUMNS)
        table.add_rows(rows)
        log = self.query_one("#log", VerticalScroll)
        log.mount(Static(f"Results for {report.model_id}", classes="agent"), table)

    def _ask_rerun(self) -> None:
        self._awaiting_rerun = True
        prompt = self.query_one("#prompt", Input)
        prompt.placeholder = _RERUN_QUESTION
        prompt.disabled = False
        prompt.focus()

    def _show_tool(self, name: str, arguments: Dict[str, Any]) -> None:
        detail = ", ".join(f"{k}={v!r}" for k, v in arguments.items() if isinstance(v, str))
        line = f"  · {name}" + (f"  {detail[:90]}" if detail else "")
        self.query_one("#log", VerticalScroll).mount(Static(Text(line), classes="tool"))
        self._entries.append(("tool", line))

    def _show_reply(self, reply: str) -> None:
        log = self.query_one("#log", VerticalScroll)
        log.mount(Static("BiasScope>", classes="agent"), Markdown(reply))
        self._entries.append(("BiasScope>", reply))
        self.sub_title = ""
        reported, self._report_handle = self._report_handle, None
        if reported:
            self._show_results_table(reported)
        log.scroll_end(animate=False)
        if self._script:
            self.call_later(self._play_next)
        elif self._autonomous:
            self.runs.append((self._current_model or "", self.transcript()))
            self._ask_model()
        elif self._scripted:
            self.set_timer(0.3, lambda: self.exit(self.transcript()))
        elif reported:
            self._ask_rerun()
        else:
            prompt = self.query_one("#prompt", Input)
            prompt.disabled = False
            prompt.focus()


def run_tui(loop: Any, make_loop: Optional[Callable[[], Any]] = None) -> int:
    BiasScopeApp(loop, make_loop=make_loop).run()
    return 0


def run_interactive(loop: Any, make_loop: Optional[Callable[[], Any]] = None):
    """Let the user type the turns; return the transcript when they leave."""
    return list(BiasScopeApp(loop, make_loop=make_loop).run() or [])


def run_autonomous(
    loop: Any, make_loop: Callable[[], Any], turns_for: Callable[[str], List[str]]
) -> List[Tuple[str, List[Tuple[str, str]]]]:
    """Ask only for model ids; each one is evaluated end to end with the plan
    confirmed on the user's behalf. Returns every finished run's (model, entries)."""
    app = BiasScopeApp(loop, make_loop=make_loop, autonomous=True, turns_for=turns_for)
    app.run()
    return list(app.runs)


def play_script(loop: Any, turns: Sequence[str], headless: bool = False) -> List[Tuple[str, str]]:
    """Run `turns` through the app and return its transcript when done."""
    entries = BiasScopeApp(loop, script=turns).run(headless=headless)
    return list(entries or [])
