# Installation

## Requirements

- Python 3.10 or newer
- pip

## Install from PyPI

```bash
pip install bias-scope
```

The core install is deliberately light. Heavier dependencies are optional extras, and a
missing extra raises an error that names the one to install.

## Optional extras

```bash
pip install "bias-scope[torch]"
pip install "bias-scope[embeddings]"
pip install "bias-scope[datasets]"
pip install "bias-scope[llm]"
pip install "bias-scope[viz]"
pip install "bias-scope[agent]"
pip install "bias-scope[all]"
```

| Extra | Adds | Needed for |
|---|---|---|
| `torch` | torch, transformers, accelerate | Probability-based metrics, local Hugging Face models (`HuggingFaceBackend`), `BertPLLScorer`, and transformer-backed generated-text metrics such as `RegardScore` |
| `embeddings` | sentence-transformers | The built-in embedding helper used by embedding-based convenience paths |
| `datasets` | datasets | Prompt-based benchmark loaders |
| `llm` | litellm | Prompt-based model calls, and `LiteLLMBackend` for chat APIs |
| `perspective` | google-api-python-client | The Perspective API client for toxicity scoring |
| `viz` | matplotlib | [Plots](../framework/viz.md) and HTML reports with figures |
| `agent` | anthropic, huggingface_hub | The [agent](../agent/index.md), with Claude as the agent LLM |
| `agent-openai` | openai | GPT, a local OpenAI-compatible server, or OpenRouter as the agent LLM |
| `agent-gemini` | google-genai | Gemini as the agent LLM |
| `all` | Every extra above except the developer and docs tools | The full feature set |

## Install from source

```bash
git clone https://github.com/RAINLabLAU/bias_scope.git
cd bias_scope
pip install -e ".[all,dev]"
```

The `dev` extra adds the test and lint tools, so `python -m pytest` and `ruff check src tests`
work.

## Build these docs locally

```bash
pip install -e ".[docs]"
mkdocs serve
```

`mkdocs build --strict` is the check that the docs build without warnings.

## Dependencies

### Core

| Package | Version | Purpose |
|---|---|---|
| numpy | >=2.1.3 | Numerical operations |
| requests | >=2.28.0 | HTTP requests (Perspective API) |
| textual | >=0.80 | The agent's terminal UI |

### Where the datasets come from

Metrics that need the authors' datasets read them from `third_party/`, which is not part of
the installed package. The agent downloads what it needs at startup. To fetch everything
from a source checkout:

```bash
python scripts/sources/fetch_sources.py --all
```

See [Datasets](../agent/datasets.md).
