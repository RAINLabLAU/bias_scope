<div align="center">
  <img src="assets/logo.png" alt="BiasScope logo" />
</div>

# BiasScope

**BiasScope** is a Python library for measuring bias in language models across four complementary families of metrics:

- embedding-based metrics
- probability-based metrics
- generated-text metrics
- prompt-based benchmarks

The goal is a single, consistent API for bias evaluation whether you are working with sentence encoders, masked language models, generated completions, or dataset-driven benchmark suites.

## Why BiasScope

- One package for multiple bias evaluation paradigms
- Consistent metric classes with `evaluate()` entrypoints
- Optional model adapters so users do not need to hand-wire every scorer
- Support for both raw-text convenience paths and precomputed inputs where appropriate
- Lightweight core install with optional extras for heavier ML dependencies

## Installation

Core install:

```bash
pip install bias-scope
```

Optional extras:

```bash
pip install "bias-scope[torch]"
pip install "bias-scope[embeddings]"
pip install "bias-scope[datasets]"
pip install "bias-scope[llm]"
pip install "bias-scope[agent]"
pip install "bias-scope[all]"
```

What each extra includes:

- `torch`: `torch`, `transformers` for probability-based masked-token metrics, `BertPLLScorer`, and transformer-backed generated-text metrics such as `RegardScore`
- `embeddings`: `sentence-transformers` for the built-in embedding helper used by embedding-based convenience paths
- `datasets`: `datasets` for prompt-based benchmark loaders
- `llm`: `litellm` for prompt-based model calls
- `agent`: `anthropic`, `huggingface_hub` for the optional `bias_scope_agent` conversational agent (see [Agent](#agent-optional) below) - Claude is the default agent LLM
- `agent-openai` / `agent-gemini`: `openai` / `google-genai`, for using GPT or Gemini as the agent LLM instead of Claude - `agent-openai` is also reused for a **local** agent LLM (any OpenAI-compatible server: Ollama, llama.cpp, LM Studio, vLLM)
- `all`: everything above

Install from source:

```bash
git clone https://github.com/RAINLabLAU/bias_scope.git
cd bias_scope
pip install -e .
```

## Quick Start

### Embedding-Based Example

Canonical WEAT takes precomputed static word embeddings:

```python
import numpy as np
from bias_scope.embeddings_based import WEAT

# Rows from a static word-embedding table; use the paper's full word lists in
# a real experiment.
X = np.array([[1.0, 0.0], [0.9, 0.1]])
Y = np.array([[0.0, 1.0], [0.1, 0.9]])
A = np.array([[1.0, 0.0], [0.95, 0.05]])
B = np.array([[0.0, 1.0], [0.05, 0.95]])

details = WEAT().evaluate((X, Y), (A, B), return_details=True)
print(f"WEAT effect size: {details['effect_size']:.4f}")
print(f"Strict permutation p-value: {details['p_value']:.4g}")
```

Raw string sequences are supported as a noncanonical BiasScope extension: they
depend on the selected encoder's tokenizer, subword handling, truncation, and
pooling. Document all of those choices when reporting a raw-text result.

### Probability-Based Example

Masked-token metrics can use a built-in model adapter via `model_name`:

```python
from bias_scope.probability_based import CrowSPairs

crows = CrowSPairs(model_name="bert-base-uncased")
score = crows.evaluate(
    sentence_pairs=[
        (
            ["Women", "are", "bad", "at", "math"],
            ["Men", "are", "bad", "at", "math"],
        )
    ]
)
print(f"CrowS-Pairs score: {score:.4f}")
```

Advanced users can still provide a custom callback or scorer wrapper when needed.

### Generated-Text Example

```python
from bias_scope.generated_text_based import ScoreParity

parity = ScoreParity(classifier=lambda texts: [0.9 if "doctor" in t else 0.4 for t in texts])
result = parity.evaluate(
    group_a_texts=[["The man is a doctor."]],
    group_b_texts=[["The woman is a nurse."]],
)
print(result)
```

### Prompt-Based Example

```python
from bias_scope.prompts_based import BBQMetric

metric = BBQMetric(model_name="gpt-4o-mini")
result = metric.evaluate(return_details=True)
print(result)
```

Prompt-based metrics typically require `bias-scope[datasets]`, `bias-scope[llm]`, or both depending on the benchmark.

## Metric Families

### Embedding-Based

Use these when you want to measure association bias in vector spaces.

- `WEAT`
- `SEAT`
- `CEAT`
- `SentenceBiasScore`
- `embed()` helper for built-in text embedding

### Probability-Based

Use these with masked or token-prediction models.

- `CrowSPairs`
- `AUL`
- `AULA`
- `CAT`
- `ICAT`
- `LMB`
- `LPBS`
- `CBS`
- `DisCoMetric`
- `BertPLLScorer`
- `TokenPredictionScorer`

### Generated-Text

Use these when you already have generations or want to score generated completions.

- `ToxicityFraction`
- `ToxicityProbability`
- `RegardScore`
- `ScoreParity`
- `SocialGroupSubstitution`
- `CoOccurrenceBiasScore`
- `CounterfactualSentimentBias`
- `DemographicRepresentation`
- `StereotypicalAssociations`
- `MarkedPersons`
- `EMT`
- `FGB`
- `GenderPolarity`
- `HONEST`
- `PGB`
- `PerspectiveAPIClient`
- `PsycholinguisticNorms`

### Prompt-Based

Use these for dataset-backed evaluation suites and benchmark-style audits.

- `AnalogicalReasoningBias`
- `BBQMetric`
- `BOLD`
- `CounterfactualFairness`
- `DemographicRepresentationBias`
- `OpinionConsistencyAcrossPersonas`
- `RealToxicityPrompts`
- `StereoSetMetric`
- `TofNof`
- `TruthfulQA`
- `UnQoverMetric`

## API Notes

- Most metrics return a scalar by default.
- Metrics that support `return_details=True` return a richer dictionary of component scores.
- Embedding metrics accept `model_name` on both `__init__` and `evaluate()`; the per-call value overrides the instance default.
- Probability-based masked-token metrics support either a built-in `model_name` path or a backward-compatible custom callback path.
- Metric objects now have informative `repr(...)` output for notebook and REPL use.

## Examples

The repository includes runnable examples for each metric family:

- [examples/embeddings_based](examples/embeddings_based)
- [examples/probability_based](examples/probability_based)
- [examples/generated_text_based](examples/generated_text_based)
- [examples/prompts_based](examples/prompts_based)
- [examples/metric_usage_examples.py](examples/metric_usage_examples.py)

## Agent (optional)

`bias_scope_agent` is a thin, separate package that wraps BiasScope in a
conversational agent: one LLM in a tool-calling loop that works out how your
model can be accessed, tells you which metrics can legally run against it and
which cannot and why, shows you a plan, and executes it only after you have
explicitly confirmed. It adds no metric-selection logic of its own — every
recommendation comes from `recommend_metrics()`, every run from `BiasSuite`,
every score carries its fidelity badge.

### 1. Install

```bash
pip install "bias-scope[agent]"          # Claude as the agent LLM
pip install "bias-scope[agent-openai]"   # GPT, a local server, or OpenRouter
pip install "bias-scope[agent-gemini]"   # Gemini
pip install "bias-scope[llm]"            # litellm, as a general escape hatch
```

Add `[torch]` if the model you want to *evaluate* is a local Hugging Face one.

### 2. Configure

The agent is configured **entirely by environment variable**; the REPL takes no
command-line flags. Nothing loads a `.env` file for you — source it yourself:

```bash
set -a; . ./.env; set +a
```

**Choosing the agent's brain**

| Variable | Values | Default |
|---|---|---|
| `BIASSCOPE_AGENT_PROVIDER` | `anthropic`, `openai`, `gemini`, `local`, `openrouter`, `litellm` | `anthropic` |
| `BIASSCOPE_AGENT_MODEL` | any slug that provider accepts | per-provider (below) |
| `BIASSCOPE_AGENT_MAX_TOKENS` | integer | `2048` |
| `BIASSCOPE_AGENT_INSPECT_LIVE` | `1/true/yes/on` ⟷ `0/false/no/off` | `true` |

Set `BIASSCOPE_AGENT_INSPECT_LIVE=0` to stop `inspect_model` making live Hub
lookups; it then uses local config files and litellm's bundled registry.

Default model per provider, when `BIASSCOPE_AGENT_MODEL` is unset:

| Provider | Default model |
|---|---|
| `anthropic` | `claude-sonnet-4-5-20250929` |
| `openai` | `gpt-4o-mini` |
| `gemini` | `gemini-2.0-flash` |
| `local` | `llama3.1` |
| `openrouter` | `anthropic/claude-3.5-sonnet` |
| `litellm` | `openrouter/anthropic/claude-3.5-sonnet` |

**Credentials**

| Provider | Variable | Notes |
|---|---|---|
| `anthropic` | `ANTHROPIC_API_KEY` | |
| `openai` | `OPENAI_API_KEY` | |
| `gemini` | `GOOGLE_API_KEY` | |
| `openrouter` | `OPENROUTER_API_KEY` | endpoint is fixed, not overridable |
| `litellm` | resolved by litellm from the model prefix | not checked up front |
| `local` | `BIASSCOPE_AGENT_LOCAL_API_KEY` (default `local`), `BIASSCOPE_AGENT_LOCAL_BASE_URL` (default `http://localhost:11434/v1`) | most local servers ignore the key |

A missing key fails at startup naming the variable, rather than as a traceback
from deep inside an SDK on the first turn.

Examples:

```bash
# Anthropic (the default)
export ANTHROPIC_API_KEY=sk-...

# OpenRouter - one key, its whole catalogue
export BIASSCOPE_AGENT_PROVIDER=openrouter
export OPENROUTER_API_KEY=sk-or-...
export BIASSCOPE_AGENT_MODEL=deepseek/deepseek-v4.1-flash

# A local server (Ollama, llama.cpp, LM Studio, vLLM) - no key needed
export BIASSCOPE_AGENT_PROVIDER=local
export BIASSCOPE_AGENT_MODEL=llama3.1
export BIASSCOPE_AGENT_LOCAL_BASE_URL=http://localhost:11434/v1

# litellm - anything it routes to, using its own model-string conventions
export BIASSCOPE_AGENT_PROVIDER=litellm
export BIASSCOPE_AGENT_MODEL=openrouter/anthropic/claude-3.5-sonnet
```

The agent LLM must support **tool calling**. A model that only returns text —
or a structured-output endpoint that returns no text at all — cannot drive it.

### 3. Run

```bash
python -m bias_scope_agent      # or the console script: bias-scope-agent
```

Exit with `exit`, `quit`, or Ctrl-D.

### 4. Use it

The model you want to evaluate is named **in conversation**, not configured.
A session takes at least three turns, because the confirm-before-run gate
requires a plan to be shown and then confirmed in a *later* turn:

```
you> I want to measure gender bias in bert-base-uncased. It is a masked LM, so
     use a huggingface encoder backend, fp32, on cuda. Which metrics can
     actually run on it, and which cannot, and why?

agent> [calls inspect_model, construct_backend, recommend_metrics,
        explain_exclusions — reports both what runs and what was excluded]

you> Plan an evaluation, axis gender, language en, using the datasets you can
     load yourself. Show me the plan and the data provenance. Don't run yet.

agent> [calls list_datasets, prepare_inputs, plan_suite — shows the metric set,
        the source file and sha256 of each dataset, and stops]

you> Yes, run it, and summarise the results with their fidelity labels.

agent> [calls confirm_plan, run_suite, summarize_report]

        Bias report for bert-base-uncased
        embedding:
          [faithful] WEAT: 0.6113 (n=16)
          [faithful] SEAT: 1.044 (n=128)
        probability:
          [faithful] CrowSPairs: 55.73 (n=262)
          [faithful] AUL: 46.56 (n=262)
          [faithful] AULA: 43.89 (n=262)
          [faithful] CAT: 69 (n=229)
          [faithful] ICAT: 51.99 (n=229)
```

**Target-model arguments**, which you give in words and the agent passes to
`construct_backend`:

| Argument | Values |
|---|---|
| `kind` | `huggingface` or `litellm` |
| `model_id` | a Hub id, a local path, or an API model string |
| `backend_kind` | `causal` or `encoder` (required for `huggingface`) |
| `dtype` | `bf16` (default), `fp32`, `fp16` |
| `device` | e.g. `cuda`, `cpu` — **there is no auto-detection**; omit it and the model stays on CPU |
| `api_base` | for a self-hosted or proxied endpoint |

There is deliberately **no `api_key` argument**, so a target model's credentials
can never enter the transcript. Export the provider's standard variable
yourself and just name the model.

### You do not paste evaluation data

The agent calls `list_datasets` and `prepare_inputs`, and the harness loads the
authors' own files itself, returning a handle plus provenance — source path,
sha256, item counts — and never the data.

This is not a convenience. Metric inputs used to travel through the agent's
output tokens, and two different frontier models were observed corrupting them:
one rewrote *"one of the best engineers in **her** field"* as *"**his** field"*,
destroying the minimal pair on exactly the token CrowS-Pairs measures; the other
silently dropped an item and scored 19 where 20 were asked for. Both produced
honest scores on data nobody chose. What a metric scores is now byte-identical
to the file on disk.

Datasets that ship with the harness:

| Dataset | Feeds | Axes |
|---|---|---|
| `crows_pairs` | `CrowSPairs`, `AUL`, `AULA` | gender, race, religion, age, and five more |
| `stereoset` | `CAT`, `ICAT` | gender, race, religion, profession |
| `weat` | `WEAT` | gender, race, age |
| `seat` | `SEAT` | gender, race, age |
| `bold_regard` | `RegardScore` | gender |
| `bold_gender_polarity` | `GenderPolarity` | gender |
| `bold_helm_bias` | `DemographicRepresentation`, `StereotypicalAssociations` | gender, race |
| `honest` | `HONEST` | gender |
| `rtp_toxicity` | `EMT` | toxicity |
| `ceat_contexts` | `CEAT` | gender, race, age |

The five generation-based datasets (`bold_*`, `honest`, `rtp_toxicity`)
**generate** continuations with the model under evaluation, so they are offered
only to backends that can generate — a causal LM can use them, an encoder
cannot. Generations are seeded and cached under `cache/generations/`.

Two providers substitute a resource the paper used and **say so in the
result**: `rtp_toxicity` scores with `unitary/toxic-bert` because the
Perspective API needs a key, and `ceat_contexts` draws contexts from BOLD's
Wikipedia sentences rather than the authors' Reddit sample. Each writes the
substitution into the result's protocol, and `summarize_report` prints it as a
`deviation:` line under the score, so a `faithful` badge is never the whole
story.

These datasets live under `third_party/`, which is git-ignored. Restore them
with `python scripts/sources/fetch_sources.py --all`; a loader that cannot find
its file says which command to run.

Where no dataset covers a metric you can still supply items yourself, and the
agent will ask. `plan_suite`'s `needs_data` names exactly what is missing,
including constructor arguments, written as `__init__.<param>`.

### Talking to it

```bash
bias-scope-agent            # a terminal UI: You > / BiasScope>, replies rendered, tool calls shown live
bias-scope-agent --plain    # the line-by-line REPL, raw text
```

The UI is a Textual app (installed with the `agent` extra). Type a turn at
`You >`; while the agent works, the tools it calls scroll past as dim lines,
then the reply appears under `BiasScope>` with its tables and emphasis
rendered. `exit` or Ctrl-C leaves. In a pipe or without `textual`, the plain
REPL is used automatically.

### What the agent will not do

- **Run anything you have not confirmed.** The gate is enforced by the tool
  dispatcher, not by the prompt: `run_suite` is refused unless a matching plan
  was produced, shown, and confirmed in a later turn.
- **Report a score for a metric that did not run.** A call that would produce an
  empty report is rejected, and a metric that declines to score reports its own
  reason.
- **Recommend a metric the backend cannot support.** Access is derived from the
  backend. A causal LM is not offered masked-LM metrics, and an encoder whose
  checkpoint has no LM head is not offered them either. That check reads the
  loaded weights, not only the config: a checkpoint whose config claims a
  masked-LM architecture but ships no head weights (the sentence-transformers
  `all-mpnet-base-v2`) is treated as having no head.

Some metrics are recommended but still cannot run here — they need a Perspective
API key, a lexicon that is not vendored, or they report no single scalar. That
set is listed with reasons in
[tests/test_recommendation_validity.py](tests/test_recommendation_validity.py),
which fails if a recommended metric outside the list stops working, and also
fails if a listed one starts.

### Scripted runs

For reproducible, non-interactive runs with a full tool-dispatch log:

```bash
python scripts/agent/live_conversation.py \
    --scenario {encoder,causal,embedding} \
    --model-id MODEL_ID \
    --device cuda \
    --out-dir results/verification/agent_live

python scripts/agent/summarize_runs.py --check
```

On a terminal the first opens the Textual UI and plays its three turns in it
(add `--plain` for raw text; a pipe gets raw text automatically). It reads
the same `BIASSCOPE_AGENT_*` variables and records every turn,
every tool call with its arguments, `summarize_report`'s own return value, and a
check listing any figure in the agent's final message that appears in no tool
result. The second tabulates recorded runs from the library's output rather than
from the agent's prose. Recorded runs are in
[results/verification/agent_live](results/verification/agent_live).

## Documentation

Project docs live under [docs/](docs).

Good starting points:

- [docs/getting-started/installation.md](docs/getting-started/installation.md)
- [docs/getting-started/quickstart.md](docs/getting-started/quickstart.md)
- [docs/api](docs/api)

## Development

Install developer dependencies:

```bash
pip install -e .[dev]
```

Run tests:

```bash
python -m pytest
```

## License

This project is licensed under the [MIT License](LICENSE).
