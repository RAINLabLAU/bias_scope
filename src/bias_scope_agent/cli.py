"""Minimal REPL entry point. Not the focus of this feature - a thin shell
around AgentLoop so the agent is reachable without embedding it in another
program.
"""

from __future__ import annotations

from typing import List, Optional

from bias_scope_agent.config import load_config
from bias_scope_agent.loop import AgentLoop
from bias_scope_agent.session import AgentSession

_QUIT_COMMANDS = {"exit", "quit"}


def main(argv: Optional[List[str]] = None) -> int:
    del argv  # no CLI flags in v1; configuration is via environment variables
    config = load_config()
    loop = AgentLoop(config, AgentSession())
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
