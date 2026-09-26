# The BiasScope agent

`bias_scope_agent` is an optional, separate package that wraps BiasScope in a
conversation. One language model runs in a tool-calling loop. It works out how your model
can be accessed, tells you which metrics can run against it and which cannot and why,
shows you a plan, and executes it **only after you have explicitly confirmed**.

It adds no metric-selection logic of its own. Every recommendation comes from
[`recommend_metrics`](../framework/recommend.md), every run from
[`BiasSuite`](../framework/suite.md), and every score carries its
[fidelity](../framework/metadata.md) label. The agent orchestrates; it never measures.

Two models are involved, and it helps to keep them apart:

- the **agent LLM** conducts the conversation. You choose it with an environment variable.
- the **target model** is the model being audited. You name it in the conversation.

## Quick start

Install with the extra for the agent LLM you want, plus `torch` if the model you are
auditing is a local Hugging Face one:

```bash
pip install "bias-scope[agent]"          # Claude as the agent LLM (the default)
pip install "bias-scope[agent-openai]"   # GPT, a local server, or OpenRouter
pip install "bias-scope[agent-gemini]"   # Gemini
pip install "bias-scope[torch]"          # to audit a local model
```

Set the credential for the agent LLM, then start it:

```bash
export ANTHROPIC_API_KEY=sk-...
bias-scope-agent          # or: python -m bias_scope_agent
```

See [configuration](configuration.md) for other providers, including OpenRouter, LiteLLM
and local servers, which need no key.

## A session

The target model is named in conversation, not configured. A session takes at least three
turns, because the [confirm-before-run gate](safeguards.md) requires a plan to be shown and
then confirmed in a *later* turn:

```
you> I want to measure gender bias in bert-base-uncased. It is a masked LM, so
     use a huggingface encoder backend, fp32, on cuda. Which metrics can
     actually run on it, and which cannot, and why?

agent> [inspects the model, builds the backend, lists what can run and what was
        excluded, with the reason for each exclusion]

you> Plan an evaluation, axis gender, language en, using the datasets you can
     load yourself. Show me the plan and the data provenance. Don't run yet.

agent> [loads the datasets server-side, shows the metric set, the source file and
        sha256 of each dataset, and stops]

you> Yes, run it, and summarise the results with their fidelity labels.

agent> [confirms the plan, runs it, and summarises]

        Bias report for bert-base-uncased
        embedding:
          [faithful] WEAT: 0.6113 (n=16)
          [faithful] SEAT: 1.044 (n=128)
        probability:
          [faithful] CrowSPairs: 55.73 (n=262)
```

The numbers above are an illustration of the format. Yours will depend on the model, the
datasets and the library version.

## Naming the target model

You describe the model in words, and the agent passes these arguments to
`construct_backend`:

| Argument | Values |
|---|---|
| `kind` | `huggingface`, or `litellm` for an API-served model such as `openrouter/meta-llama/llama-3.1-8b-instruct` (only completions and chat metrics can run) |
| `model_id` | A Hub id, a local path, or an API model string |
| `backend_kind` | `causal` or `encoder`, required for `huggingface` |
| `dtype` | `bf16` (default), `fp32`, `fp16` |
| `device` | For example `cuda` or `cpu`. **There is no auto-detection**: omit it and the model stays on CPU. |
| `api_base` | For a self-hosted or proxied endpoint |

There is deliberately no `api_key` argument. See [safeguards](safeguards.md).

## The tools

The agent has twelve tools. Each is a thin wrapper over the library.

| Tool | What it does |
|---|---|
| `inspect_model` | A best-effort guess about a model identifier: causal or encoder, whether it has an LM head, whether an endpoint is chat-formatted. Never a final decision. |
| `construct_backend` | Builds the backend and returns an opaque handle. |
| `recommend_metrics_tool` | Lists the metrics legal for the backend's access. |
| `explain_exclusions_tool` | Says why each other metric was excluded. |
| `list_datasets` | Lists the evaluation datasets the harness can load itself, and which metrics each one feeds. |
| `prepare_inputs` | Loads a dataset server-side and returns a handle plus provenance. The data itself is never returned. |
| `plan_suite` | A dry-run plan. Reports which metrics still need data from you. |
| `request_missing_inputs` | Tells you what data is needed and ends the agent's turn. |
| `confirm_plan` | Confirms a plan after you have seen it and replied, in a later turn. |
| `run_suite` | Runs the confirmed plan. Blocked by the harness unless the gate accepts the call. |
| `summarize_report` | Renders the report as a chat summary, Markdown or HTML. |
| `record_fact` | Remembers something you already said, so it is not asked again. |

Handles are opaque identifiers. Backends, datasets and reports stay in the harness, so
model objects and large results never enter the conversation.

## Where to go next

- [Configuration](configuration.md): choose the agent LLM and set credentials.
- [Datasets](datasets.md): why you do not paste evaluation data, and what ships with it.
- [Running it](running.md): the terminal UI, autonomous mode and recorded runs.
- [Safeguards](safeguards.md): what the harness guarantees, and what it does not.
- [Reference](reference.md): the package API.
