"""A thin, single-LLM tool-calling agent that drives the bias_scope library.

See bias_scope_agent.loop.AgentLoop for the tool-calling loop, and
bias_scope_agent.cli for the REPL entry point (`python -m bias_scope_agent`).
"""

from bias_scope_agent.config import AgentConfig, load_config
from bias_scope_agent.loop import AgentLoop
from bias_scope_agent.session import AgentSession

__all__ = ["AgentConfig", "load_config", "AgentLoop", "AgentSession"]
