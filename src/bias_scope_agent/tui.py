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

from typing import Any, Dict, List, Optional, Sequence, Tuple

from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Footer, Header, Input, Markdown, Static

_QUIT_WORDS = {"exit", "quit"}


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
    #prompt { dock: bottom; border: tall #38bdf8; }
    #prompt:focus { border: tall #4ade80; }
    Footer { background: #1e293b; }
    """
    BINDINGS = [("ctrl+c", "quit", "Quit")]

    def __init__(self, loop: Any, script: Optional[Sequence[str]] = None) -> None:
        """`script`: turns to play automatically, one after each reply, after
        which the app exits with its transcript (the scripted runner's mode).
        Without it the user types the turns."""
        super().__init__()
        self.loop = loop
        self._entries: List[Tuple[str, str]] = []
        self._script: List[str] = list(script or [])
        self._scripted = bool(script)
        if hasattr(loop, "on_tool"):
            loop.on_tool = self._tool_called_in_worker

    def on_mount(self) -> None:
        if self._scripted:
            self.query_one("#prompt", Input).disabled = True
            self.call_later(self._play_next)

    async def _play_next(self) -> None:
        if self._script:
            await self._submit(self._script.pop(0))

    def compose(self) -> ComposeResult:
        yield Header()
        yield VerticalScroll(id="log")
        yield Input(placeholder="You >  ask about a model, ask for a plan, confirm it", id="prompt")
        yield Footer()

    def transcript(self) -> List[Tuple[str, str]]:
        """(role, text) per block: 'You >', 'tool' or 'BiasScope>'."""
        return list(self._entries)

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        event.input.value = ""
        if not text:
            return
        if text.lower() in _QUIT_WORDS:
            self.exit()
            return
        await self._submit(text)

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
        self.call_from_thread(self._show_tool, name, arguments)

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
        log.scroll_end(animate=False)
        if self._script:
            self.call_later(self._play_next)
        elif self._scripted:
            self.set_timer(0.3, lambda: self.exit(self.transcript()))
        else:
            prompt = self.query_one("#prompt", Input)
            prompt.disabled = False
            prompt.focus()


def run_tui(loop: Any) -> int:
    BiasScopeApp(loop).run()
    return 0


def play_script(loop: Any, turns: Sequence[str], headless: bool = False) -> List[Tuple[str, str]]:
    """Run `turns` through the app and return its transcript when done."""
    entries = BiasScopeApp(loop, script=turns).run(headless=headless)
    return list(entries or [])
