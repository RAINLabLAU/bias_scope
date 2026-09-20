"""Minimal REPL entry point. Not the focus of this feature - a thin shell
around AgentLoop so the agent is reachable without embedding it in another
program.
"""

from __future__ import annotations

import sys
from typing import List, Optional

from bias_scope_agent.config import load_config
from bias_scope_agent.loop import AgentLoop
from bias_scope_agent.session import AgentSession

_QUIT_COMMANDS = {"exit", "quit"}


def _tui_available() -> bool:
    try:
        import textual  # noqa: F401
    except ImportError:
        return False
    return sys.stdin.isatty() and sys.stdout.isatty()


def main(argv: Optional[List[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    plain = "--plain" in args  # the only flag; configuration is via environment variables
    config = load_config()
    loop = AgentLoop(config, AgentSession())
    if not plain and _tui_available():
        from bias_scope_agent.tui import run_tui

        return run_tui(loop)
    print("bias-scope-agent - type 'exit' or Ctrl-D to quit")
    while True:
        try:
            user_text = input("you> ")
        except EOFError:
            print()
            break
        if user_text.strip().lower() in _QUIT_COMMANDS:
            break
        print(f"agent> {loop.run_turn(user_text)}")
    return 0
