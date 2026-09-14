"""Agent LLM configuration, loaded once before any user interaction.

Target-model details are supplied by the user at runtime (via tool calls),
not configured here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping, Optional

_TRUTHY = {"1", "true", "yes", "on"}
_FALSY = {"0", "false", "no", "off"}

_DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-5-20250929"

# One reasonable default model per supported agent-LLM provider, used when
# BIASSCOPE_AGENT_PROVIDER is set but BIASSCOPE_AGENT_MODEL is not - a model
# string for the wrong provider is not a sensible fallback.
_DEFAULT_MODELS = {
    "anthropic": _DEFAULT_ANTHROPIC_MODEL,
    "openai": "gpt-4o-mini",
    "gemini": "gemini-2.0-flash",
    # Whatever model name the local server resolves - almost certainly not
    # correct for every setup (depends entirely on what the user has pulled),
    # but a name common enough to be a reasonable guess for Ollama users.
    "local": "llama3.1",
}


@dataclass(frozen=True)
class AgentConfig:
    model: str = _DEFAULT_ANTHROPIC_MODEL
    max_tokens: int = 2048
    inspect_model_live: bool = True
    provider: str = "anthropic"


def _parse_bool(raw: str, *, var_name: str) -> bool:
    lowered = raw.strip().lower()
    if lowered in _TRUTHY:
        return True
    if lowered in _FALSY:
        return False
    raise ValueError(f"{var_name}={raw!r} is not a recognized boolean value")


def load_config(env: Optional[Mapping[str, str]] = None) -> AgentConfig:
    """Build an AgentConfig from environment variables.

    No API key field: the anthropic SDK already reads ANTHROPIC_API_KEY from
    the environment on its own, so this config deliberately does not
    duplicate that (see REVIEW_LATER.md RL-038).
    """
    source = os.environ if env is None else env
    defaults = AgentConfig()

    raw_provider = source.get("BIASSCOPE_AGENT_PROVIDER")
    provider = defaults.provider if raw_provider is None else raw_provider.strip().lower()
    if provider not in _DEFAULT_MODELS:
        raise ValueError(
            f"BIASSCOPE_AGENT_PROVIDER={raw_provider!r} must be one of "
            f"{sorted(_DEFAULT_MODELS)}"
        )

    raw_model = source.get("BIASSCOPE_AGENT_MODEL")
    model = raw_model if raw_model is not None else _DEFAULT_MODELS[provider]

    raw_max_tokens = source.get("BIASSCOPE_AGENT_MAX_TOKENS")
    if raw_max_tokens is None:
        max_tokens = defaults.max_tokens
    else:
        try:
            max_tokens = int(raw_max_tokens)
        except ValueError as exc:
            raise ValueError(
                f"BIASSCOPE_AGENT_MAX_TOKENS={raw_max_tokens!r} is not an integer"
            ) from exc

    raw_inspect_live = source.get("BIASSCOPE_AGENT_INSPECT_LIVE")
    if raw_inspect_live is None:
        inspect_model_live = defaults.inspect_model_live
    else:
        inspect_model_live = _parse_bool(raw_inspect_live, var_name="BIASSCOPE_AGENT_INSPECT_LIVE")

    return AgentConfig(
        model=model, max_tokens=max_tokens, inspect_model_live=inspect_model_live, provider=provider
    )
