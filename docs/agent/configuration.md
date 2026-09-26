# Configuration

The agent is configured **entirely by environment variable**, and the target model is
named in conversation instead. Nothing loads a `.env` file for you, so source it yourself:

```bash
set -a; . ./.env; set +a
```

## Choosing the agent LLM

| Variable | Values | Default |
|---|---|---|
| `BIASSCOPE_AGENT_PROVIDER` | `anthropic`, `openai`, `gemini`, `local`, `openrouter`, `litellm` | `anthropic` |
| `BIASSCOPE_AGENT_MODEL` | Any slug the provider accepts | Per provider, below |
| `BIASSCOPE_AGENT_MAX_TOKENS` | An integer | `2048` |
| `BIASSCOPE_AGENT_INSPECT_LIVE` | `1`/`true`/`yes`/`on`, or `0`/`false`/`no`/`off` | `true` |

Set `BIASSCOPE_AGENT_INSPECT_LIVE=0` to stop `inspect_model` making live Hugging Face Hub
lookups. It then uses local config files and LiteLLM's bundled model registry.

When `BIASSCOPE_AGENT_MODEL` is unset, the model defaults by provider:

| Provider | Default model |
|---|---|
| `anthropic` | `claude-sonnet-4-5-20250929` |
| `openai` | `gpt-4o-mini` |
| `gemini` | `gemini-2.0-flash` |
| `local` | `llama3.1` |
| `openrouter` | `anthropic/claude-3.5-sonnet` |
| `litellm` | `openrouter/anthropic/claude-3.5-sonnet` |

The agent LLM must support **tool calling**. A model that only returns text, or a
structured-output endpoint that returns no text at all, cannot drive it.

## Credentials

| Provider | Variable | Notes |
|---|---|---|
| `anthropic` | `ANTHROPIC_API_KEY` | |
| `openai` | `OPENAI_API_KEY` | |
| `gemini` | `GOOGLE_API_KEY` | |
| `openrouter` | `OPENROUTER_API_KEY` | The endpoint is fixed, not overridable. |
| `litellm` | Resolved by LiteLLM from the model prefix | Not checked up front. |
| `local` | `BIASSCOPE_AGENT_LOCAL_API_KEY` (default `local`) and `BIASSCOPE_AGENT_LOCAL_BASE_URL` (default `http://localhost:11434/v1`) | Most local servers ignore the key. |

A missing key fails at startup naming the variable, instead of as a traceback from deep
inside an SDK on the first turn.

## Examples

```bash
# Anthropic (the default)
export ANTHROPIC_API_KEY=sk-...

# OpenRouter: one key, its whole catalogue
export BIASSCOPE_AGENT_PROVIDER=openrouter
export OPENROUTER_API_KEY=sk-or-...
export BIASSCOPE_AGENT_MODEL=deepseek/deepseek-v4.1-flash

# A local server (Ollama, llama.cpp, LM Studio, vLLM): no key needed
export BIASSCOPE_AGENT_PROVIDER=local
export BIASSCOPE_AGENT_MODEL=llama3.1
export BIASSCOPE_AGENT_LOCAL_BASE_URL=http://localhost:11434/v1

# LiteLLM: anything it routes to, using its own model-string conventions
export BIASSCOPE_AGENT_PROVIDER=litellm
export BIASSCOPE_AGENT_MODEL=openrouter/anthropic/claude-3.5-sonnet
```

## How the providers relate

Each provider adapter translates the agent's single internal tool schema into that
provider's function-calling format and normalizes the reply, so the loop never branches on
the provider. The OpenAI, OpenRouter and local adapters share one translation, and LiteLLM
reuses it because its responses follow the same format.

The `agent-openai` extra is reused for a **local** agent LLM: any OpenAI-compatible server
works. Installing `bias-scope[llm]` adds LiteLLM as a general escape hatch.

## Startup options

| Option | Effect |
|---|---|
| `--plain` | The line-by-line REPL instead of the terminal UI. |
| `--autonomous` | Asks only for a model id, then evaluates it end to end. See [running it](running.md). |
| `--device DEVICE` | The device for `--autonomous` runs. Default: CUDA if available. |
| `--no-fetch` | Do not download missing dataset files at startup. `BIASSCOPE_AGENT_AUTO_FETCH=0` does the same. |

By default the agent checks for the authors' dataset files when it starts and downloads
any that are missing, so it never plans a run and then stops on an absent file. See
[datasets](datasets.md).
