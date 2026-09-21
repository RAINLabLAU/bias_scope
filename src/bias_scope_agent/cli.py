"""Minimal REPL entry point. Not the focus of this feature - a thin shell
around AgentLoop so the agent is reachable without embedding it in another
program.
"""

from __future__ import annotations

import argparse
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


def _default_device() -> str:
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="bias-scope-agent",
        description="Talk to the BiasScope agent. The agent LLM is configured by "
        "BIASSCOPE_AGENT_PROVIDER / BIASSCOPE_AGENT_MODEL and the provider's API key.",
    )
    parser.add_argument("--plain", action="store_true",
                        help="plain text REPL instead of the Textual UI")
    parser.add_argument("--autonomous", action="store_true",
                        help="ask only for a model id; set up, plan, confirm and run every "
                        "recommended metric the harness can feed, show the table, ask again")
    parser.add_argument("--device", default=None,
                        help="device for --autonomous runs (default: cuda if available)")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(list(sys.argv[1:] if argv is None else argv))
    config = load_config()
    loop = AgentLoop(config, AgentSession())
    if not args.plain and _tui_available():
        from bias_scope_agent import tui

        def fresh() -> AgentLoop:  # each model gets its own session
            return AgentLoop(config, AgentSession())

        if args.autonomous:
            from bias_scope_agent.scenarios import turns_for_model

            device = args.device or _default_device()
            tui.run_autonomous(loop, fresh, lambda model: turns_for_model(model, device))
            return 0
        # After a report the UI offers another run; each one gets a fresh session.
        return tui.run_tui(loop, make_loop=fresh)
    if args.autonomous:
        print("--autonomous needs the Textual UI (a terminal, and not --plain)")
        return 2
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
