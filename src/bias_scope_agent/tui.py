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

from typing import Any, Dict, List, Tuple

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
    #log { padding: 1 2; }
    .you { color: $accent; text-style: bold; margin-top: 1; }
    .you-text { margin-left: 2; }
    .agent { color: $success; text-style: bold; margin-top: 1; }
    .tool { color: $text-muted; margin-left: 2; }
    Markdown { margin-left: 2; }
    #prompt { dock: bottom; }
    """
    BINDINGS = [("ctrl+c", "quit", "Quit")]

    def __init__(self, loop: Any) -> None:
        super().__init__()
        self.loop = loop
        self._entries: List[Tuple[str, str]] = []
        if hasattr(loop, "on_tool"):
            loop.on_tool = self._tool_called_in_worker

    def compose(self) -> ComposeResult:
        yield Header()
        yield VerticalScroll(id="log")
        yield Input(placeholder="You > ask about a model, plan, then confirm", id="prompt")
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
        log = self.query_one("#log", VerticalScroll)
        await log.mount(Static("You >", classes="you"), Static(Text(text), classes="you-text"))
        self._entries.append(("You >", text))
        event.input.disabled = True
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
        prompt = self.query_one("#prompt", Input)
        prompt.disabled = False
        prompt.focus()
        self.sub_title = ""
        log.scroll_end(animate=False)


def run_tui(loop: Any) -> int:
    BiasScopeApp(loop).run()
    return 0
