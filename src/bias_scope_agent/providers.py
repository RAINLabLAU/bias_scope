"""Provider adapters: translate bias_scope_agent's single internal tool
representation (schemas.TOOLS, Anthropic tool-use JSON shape) to and from
each provider's own function-calling wire format, so AgentLoop's dispatch
logic (_dispatch_tools, _text_of in loop.py) never has to branch on provider
- every adapter's `create()` takes the same (system, messages) shape and
returns the same NormalizedResponse shape.

AgentLoop's own transcript (`self.messages`) is kept in this provider-
agnostic normalized form throughout a conversation:
  - a plain user text turn:      {"role": "user", "content": "..."}
  - a batch of tool results:     {"role": "user", "content": [tool_result, ...]}
    where tool_result is {"type": "tool_result", "tool_use_id", "content",
    "name", "is_error"?} - "name" is bookkeeping this package adds (Anthropic
    and OpenAI correlate tool results by id and ignore it; Gemini's protocol
    correlates by name, so its adapter is the one that actually needs it).
  - an assistant turn:           {"role": "assistant", "content": [NormalizedBlock, ...]}

Each adapter re-translates the *whole* transcript on every call rather than
maintaining its own incremental native-format history - the minimum needed
for correctness, not the most efficient possible design, but it keeps every
adapter stateless and keeps AgentLoop the one owner of conversation state.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from uuid import uuid4

from bias_scope_agent.config import AgentConfig
from bias_scope_agent.schemas import TOOLS


@dataclass
class NormalizedBlock:
    type: str  # "text" or "tool_use"
    text: str = ""
    name: str = ""
    id: str = ""
    input: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NormalizedResponse:
    content: List[NormalizedBlock]
    stop_reason: str  # "tool_use" or "end_turn"


# ---------------------------------------------------------------- Anthropic


def _build_anthropic_client() -> Any:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. bias_scope_agent needs it to talk to "
            "Claude - export it before running, e.g.:\n"
            "  export ANTHROPIC_API_KEY=sk-ant-...\n"
            "(get a key at https://console.anthropic.com/settings/keys)"
        )
    try:
        import anthropic
    except ImportError as exc:
        raise RuntimeError(
            "the anthropic package is not installed. Install it with: "
            'pip install "bias-scope[agent]"'
        ) from exc
    return anthropic.Anthropic()


def _anthropic_tool_result_block(result: Dict[str, Any]) -> Dict[str, Any]:
    block = {
        "type": "tool_result",
        "tool_use_id": result["tool_use_id"],
        "content": result["content"],
    }
    if result.get("is_error"):
        block["is_error"] = True
    return block


def _anthropic_block_from_normalized(block: NormalizedBlock) -> Dict[str, Any]:
    if block.type == "text":
        return {"type": "text", "text": block.text}
    return {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}


def _anthropic_message(message: Dict[str, Any]) -> Dict[str, Any]:
    content = message["content"]
    if isinstance(content, str):
        return {"role": message["role"], "content": content}
    if message["role"] == "assistant":
        blocks = [_anthropic_block_from_normalized(b) for b in content]
    else:
        blocks = [_anthropic_tool_result_block(r) for r in content]
    return {"role": message["role"], "content": blocks}


def _normalize_anthropic_response(response: Any) -> NormalizedResponse:
    blocks = []
    for block in response.content:
        if block.type == "text":
            blocks.append(NormalizedBlock(type="text", text=block.text))
        else:
            blocks.append(
                NormalizedBlock(type="tool_use", name=block.name, id=block.id, input=block.input)
            )
    return NormalizedResponse(content=blocks, stop_reason=response.stop_reason)


class AnthropicProvider:
    def __init__(self, config: AgentConfig, client: Optional[Any] = None):
        self.config = config
        self.client = client if client is not None else _build_anthropic_client()

    def create(self, *, system: str, messages: List[Dict[str, Any]]) -> NormalizedResponse:
        response = self.client.messages.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            system=system,
            messages=[_anthropic_message(m) for m in messages],
            tools=TOOLS,
        )
        return _normalize_anthropic_response(response)


# ------------------------------------------------------------------ OpenAI


def _build_openai_client() -> Any:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. bias_scope_agent needs it to talk to "
            "OpenAI - export it before running, e.g.:\n"
            "  export OPENAI_API_KEY=sk-...\n"
            "(get a key at https://platform.openai.com/api-keys)"
        )
    try:
        import openai
    except ImportError as exc:
        raise RuntimeError(
            "the openai package is not installed. Install it with: "
            'pip install "bias-scope[agent-openai]"'
        ) from exc
    return openai.OpenAI()


def _openai_tools() -> List[Dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            },
        }
        for t in TOOLS
    ]


def _openai_assistant_message(blocks: List[NormalizedBlock]) -> Dict[str, Any]:
    text = "\n".join(b.text for b in blocks if b.type == "text")
    tool_calls = [
        {
            "id": b.id,
            "type": "function",
            "function": {"name": b.name, "arguments": json.dumps(b.input)},
        }
        for b in blocks
        if b.type == "tool_use"
    ]
    message: Dict[str, Any] = {"role": "assistant", "content": text or None}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message


def _openai_tool_message(result: Dict[str, Any]) -> Dict[str, Any]:
    content = result["content"]
    if result.get("is_error"):
        content = f"ERROR: {content}"
    return {"role": "tool", "tool_call_id": result["tool_use_id"], "content": content}


def _openai_messages_from(message: Dict[str, Any]) -> List[Dict[str, Any]]:
    content = message["content"]
    if isinstance(content, str):
        return [{"role": message["role"], "content": content}]
    if message["role"] == "assistant":
        return [_openai_assistant_message(content)]
    return [_openai_tool_message(r) for r in content]


def _normalize_openai_response(response: Any) -> NormalizedResponse:
    message = response.choices[0].message
    blocks = []
    if message.content:
        blocks.append(NormalizedBlock(type="text", text=message.content))
    for call in message.tool_calls or []:
        blocks.append(
            NormalizedBlock(
                type="tool_use",
                id=call.id,
                name=call.function.name,
                input=json.loads(call.function.arguments),
            )
        )
    finish_reason = response.choices[0].finish_reason
    stop_reason = "tool_use" if finish_reason == "tool_calls" else "end_turn"
    return NormalizedResponse(content=blocks, stop_reason=stop_reason)


class OpenAIProvider:
    def __init__(self, config: AgentConfig, client: Optional[Any] = None):
        self.config = config
        self.client = client if client is not None else _build_openai_client()

    def create(self, *, system: str, messages: List[Dict[str, Any]]) -> NormalizedResponse:
        openai_messages = [{"role": "system", "content": system}]
        for message in messages:
            openai_messages.extend(_openai_messages_from(message))
        response = self.client.chat.completions.create(
            model=self.config.model, messages=openai_messages, tools=_openai_tools()
        )
        return _normalize_openai_response(response)


# -------------------------------------------------------------------- Local

# Any server exposing an OpenAI-compatible chat completions API: Ollama,
# llama.cpp's server, LM Studio, vLLM, and others all converged on this
# wire format, so this reuses the openai package as a generic client rather
# than adding a new dependency. Defaults target Ollama (the most common
# self-hosted runner) but work with any of them via the env vars below.
_DEFAULT_LOCAL_BASE_URL = "http://localhost:11434/v1"
_DEFAULT_LOCAL_API_KEY = "local"  # most local servers ignore this entirely


def _build_local_client() -> Any:
    base_url = os.environ.get("BIASSCOPE_AGENT_LOCAL_BASE_URL", _DEFAULT_LOCAL_BASE_URL)
    api_key = os.environ.get("BIASSCOPE_AGENT_LOCAL_API_KEY", _DEFAULT_LOCAL_API_KEY)
    try:
        import openai
    except ImportError as exc:
        raise RuntimeError(
            "the openai package is not installed (bias_scope_agent's local "
            "provider reuses it as a generic OpenAI-compatible client). "
            'Install it with: pip install "bias-scope[agent-openai]"'
        ) from exc
    return openai.OpenAI(base_url=base_url, api_key=api_key)


class LocalProvider(OpenAIProvider):
    """A locally-served model behind an OpenAI-compatible endpoint (Ollama,
    llama.cpp server, LM Studio, vLLM, ...). No API key is required by
    default - unlike the cloud providers, that would defeat the point.
    Inherits create() unchanged from OpenAIProvider: the wire format is
    identical, only the endpoint and auth differ, so only client
    construction is overridden here.
    """

    def __init__(self, config: AgentConfig, client: Optional[Any] = None):
        self.config = config
        self.client = client if client is not None else _build_local_client()


# --------------------------------------------------------------- OpenRouter

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def _build_openrouter_client() -> Any:
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. bias_scope_agent needs it to talk to "
            "OpenRouter - export it before running, e.g.:\n"
            "  export OPENROUTER_API_KEY=sk-or-...\n"
            "(get a key at https://openrouter.ai/settings/keys)"
        )
    try:
        import openai
    except ImportError as exc:
        raise RuntimeError(
            "the openai package is not installed (bias_scope_agent's OpenRouter "
            "provider reuses it as a generic OpenAI-compatible client). Install "
            'it with: pip install "bias-scope[agent-openai]"'
        ) from exc
    return openai.OpenAI(base_url=_OPENROUTER_BASE_URL, api_key=os.environ["OPENROUTER_API_KEY"])


class OpenRouterProvider(OpenAIProvider):
    """OpenRouter exposes a plain OpenAI-compatible chat completions API, so
    this is OpenAIProvider pointed at a different endpoint - same reasoning
    and same reuse pattern as LocalProvider, just with a required API key
    since OpenRouter is a hosted, metered service rather than something on
    localhost.
    """

    def __init__(self, config: AgentConfig, client: Optional[Any] = None):
        self.config = config
        self.client = client if client is not None else _build_openrouter_client()


# ----------------------------------------------------------------- LiteLLM


def _import_litellm() -> Any:
    try:
        import litellm
    except ImportError as exc:
        raise RuntimeError(
            "the litellm package is not installed. Install it with: "
            'pip install "bias-scope[llm]"'
        ) from exc
    return litellm


def _wrap_litellm_client(litellm_module: Any) -> Any:
    """Makes litellm.completion() answer to the same
    `.chat.completions.create(...)` shape OpenAIProvider.create() already
    calls. litellm.completion()'s response is already OpenAI-shaped (that is
    the whole point of litellm), so no separate translation is needed - just
    this shim, so LiteLLMProvider can reuse create() unchanged.
    """
    completions = SimpleNamespace(create=litellm_module.completion)
    return SimpleNamespace(chat=SimpleNamespace(completions=completions))


class LiteLLMProvider(OpenAIProvider):
    """A general escape hatch: any provider litellm itself supports (100+,
    including OpenRouter's full catalog via "openrouter/<slug>" model
    strings, e.g. "openrouter/anthropic/claude-3.5-sonnet"), routed through
    litellm.completion() directly rather than a hand-built adapter per
    provider. Reuses OpenAIProvider.create() unchanged via _wrap_litellm_client.

    Deliberately has no required API key at construction, unlike the other
    cloud providers: litellm resolves the right environment variable itself
    from the model string's provider prefix - there is no single variable to
    eagerly check for generically. A missing/wrong key surfaces as litellm's
    own authentication error on the first real call instead.
    """

    def __init__(self, config: AgentConfig, client: Optional[Any] = None):
        self.config = config
        self.client = _wrap_litellm_client(client if client is not None else _import_litellm())


# ------------------------------------------------------------------ Gemini

# Gemini's function-declaration schema rejects JSON-schema keywords it does
# not recognize (e.g. "default", used in a couple of schemas.py properties)
# - kept to a conservative, known-safe subset rather than passing schemas
# through unmodified.
_GEMINI_SCHEMA_KEYS = {"type", "properties", "items", "required", "enum", "description"}


def _build_gemini_client() -> Any:
    if not os.environ.get("GOOGLE_API_KEY"):
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. bias_scope_agent needs it to talk to "
            "Gemini - export it before running, e.g.:\n"
            "  export GOOGLE_API_KEY=...\n"
            "(get a key at https://aistudio.google.com/apikey)"
        )
    try:
        import google.genai as genai
    except ImportError as exc:
        raise RuntimeError(
            "the google-genai package is not installed. Install it with: "
            'pip install "bias-scope[agent-gemini]"'
        ) from exc
    return genai.Client()


def _strip_unsupported_schema_fields(schema: Dict[str, Any]) -> Dict[str, Any]:
    cleaned = {k: v for k, v in schema.items() if k in _GEMINI_SCHEMA_KEYS}
    if "properties" in cleaned:
        cleaned["properties"] = {
            name: _strip_unsupported_schema_fields(sub)
            for name, sub in cleaned["properties"].items()
        }
    if "items" in cleaned:
        cleaned["items"] = _strip_unsupported_schema_fields(cleaned["items"])
    return cleaned


def _gemini_model_content(blocks: List[NormalizedBlock]) -> Dict[str, Any]:
    parts = []
    for block in blocks:
        if block.type == "text":
            parts.append({"text": block.text})
        else:
            parts.append({"function_call": {"name": block.name, "args": block.input}})
    return {"role": "model", "parts": parts}


def _gemini_tool_result_content(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    parts = [
        {
            "function_response": {
                "name": r["name"],
                "response": {"result": r["content"], "error": r.get("is_error", False)},
            }
        }
        for r in results
    ]
    return {"role": "user", "parts": parts}


def _gemini_contents_from(message: Dict[str, Any]) -> List[Dict[str, Any]]:
    content = message["content"]
    if isinstance(content, str):
        role = "user" if message["role"] == "user" else "model"
        return [{"role": role, "parts": [{"text": content}]}]
    if message["role"] == "assistant":
        return [_gemini_model_content(content)]
    return [_gemini_tool_result_content(content)]


def _normalize_gemini_response(response: Any) -> NormalizedResponse:
    parts = response.candidates[0].content.parts
    blocks = []
    has_function_call = False
    for part in parts:
        function_call = getattr(part, "function_call", None)
        if function_call is not None:
            has_function_call = True
            # Gemini does not assign an id to a function call the way
            # Anthropic/OpenAI do; synthesize one for our own bookkeeping.
            # It is never sent back to Gemini - tool results correlate by
            # name there, not id.
            blocks.append(
                NormalizedBlock(
                    type="tool_use",
                    id=uuid4().hex,
                    name=function_call.name,
                    input=dict(function_call.args),
                )
            )
        elif getattr(part, "text", None):
            blocks.append(NormalizedBlock(type="text", text=part.text))
    stop_reason = "tool_use" if has_function_call else "end_turn"
    return NormalizedResponse(content=blocks, stop_reason=stop_reason)


class GeminiProvider:
    def __init__(self, config: AgentConfig, client: Optional[Any] = None):
        self.config = config
        self.client = client if client is not None else _build_gemini_client()

    def _tool_declarations(self) -> List[Dict[str, Any]]:
        return [
            {
                "function_declarations": [
                    {
                        "name": t["name"],
                        "description": t["description"],
                        "parameters": _strip_unsupported_schema_fields(t["input_schema"]),
                    }
                    for t in TOOLS
                ]
            }
        ]

    def create(self, *, system: str, messages: List[Dict[str, Any]]) -> NormalizedResponse:
        contents = []
        for message in messages:
            contents.extend(_gemini_contents_from(message))
        response = self.client.models.generate_content(
            model=self.config.model,
            contents=contents,
            config={"system_instruction": system, "tools": self._tool_declarations()},
        )
        return _normalize_gemini_response(response)


# --------------------------------------------------------------------- API

_PROVIDERS = {
    "anthropic": AnthropicProvider,
    "openai": OpenAIProvider,
    "gemini": GeminiProvider,
    "local": LocalProvider,
    "openrouter": OpenRouterProvider,
    "litellm": LiteLLMProvider,
}


def build_provider(config: AgentConfig, client: Optional[Any] = None) -> Any:
    provider_cls = _PROVIDERS.get(config.provider)
    if provider_cls is None:
        raise ValueError(f"provider must be one of {sorted(_PROVIDERS)}, got {config.provider!r}")
    return provider_cls(config, client=client)
