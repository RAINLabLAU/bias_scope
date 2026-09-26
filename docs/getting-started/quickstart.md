# Quick Start

This page goes from "which metrics fit my model?" to a report, then shows one direct
example for each metric family.

## 1. Which metrics can I run?

`recommend_metrics` answers from what the model can actually provide:

```python
from bias_scope import recommend_metrics, explain_exclusions

api_access = ("completions", "chat")          # a hosted chat API: no logits, no embeddings
for rec in recommend_metrics(api_access, axis="gender")[:3]:
    print(rec.fidelity, rec.metric)

print(explain_exclusions(api_access)["AUL"])  # needs logits, which this backend lacks
```

In real use, take the access from a backend instead of typing it, as in the next step. See
[Choosing metrics](../framework/recommend.md).

## 2. Run many metrics and get a report

```python
from bias_scope import BiasSuite, to_markdown
from bias_scope.backends import HuggingFaceBackend

backend = HuggingFaceBackend("bert-base-uncased", kind="encoder", dtype="fp32")
suite = BiasSuite(backend, axis="gender")

print(suite.plan())                  # what would run, without running anything
report = suite.run(inputs=inputs)    # per-metric inputs; a metric without any is skipped
print(to_markdown(report))
```

Every score carries a confidence interval and its fidelity label. A metric with no inputs
is skipped with a reason, never scored as zero. See [Running a suite](../framework/suite.md)
for the inputs format and a version that runs offline.

## 3. Or ask the agent

If you would rather describe the model in words and have the plan proposed for you, use the
[agent](../agent/index.md). It runs nothing until you confirm.

## Direct examples, one per family

Each metric can also be used on its own through `evaluate()`.

### Embedding-based: WEAT

Canonical WEAT takes precomputed static word embeddings:

```python
import numpy as np
from bias_scope.embeddings_based import WEAT

# Rows from a static word-embedding table; use the paper's full word lists in a
# real experiment.
X = np.array([[1.0, 0.0], [0.9, 0.1]])
Y = np.array([[0.0, 1.0], [0.1, 0.9]])
A = np.array([[1.0, 0.0], [0.95, 0.05]])
B = np.array([[0.0, 1.0], [0.05, 0.95]])

details = WEAT().evaluate((X, Y), (A, B), return_details=True)
print(f"WEAT effect size: {details['effect_size']:.4f}")
print(f"Strict permutation p-value: {details['p_value']:.4g}")
# > 0  -> the first target set associates more with the first attribute set
# ~0   -> no differential association (no bias)
```

Raw string sequences are supported as a noncanonical BiasScope extension. They depend on the
selected encoder's tokenizer, subword handling, truncation and pooling, so document all of
those choices when reporting a raw-text result.

### Probability-based: CrowS-Pairs

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

### Generated text: MeanScoreGap

Compares any classifier's scores between two groups of generated texts:

```python
from bias_scope.generated_text_based import MeanScoreGap

gap = MeanScoreGap(classifier=lambda texts: [0.9 if "doctor" in t else 0.4 for t in texts])
result = gap.evaluate(
    group_a_texts=[["The man is a doctor."]],
    group_b_texts=[["The woman is a nurse."]],
)
print(result["difference"])
```

### Prompt-based: BBQ

```python
from bias_scope.prompts_based import BBQMetric

metric = BBQMetric(
    model_name="openrouter/meta-llama/llama-3.1-8b-instruct",
    api_key="your-api-key-here",
)

result = metric.evaluate(num_samples=20, subset="Age")
print(f"Bias score: {result['bias_score']:.2%}")
print(f"Accuracy:   {result['accuracy']:.2%}")
```

Prompt-based metrics need the `datasets` extra, the `llm` extra, or both. Many of the
newer ones also accept responses you already have, so they run without an API. See each
metric's page for its inputs.

## Supported LLM providers (prompt-based metrics)

Prompt-based metrics use [LiteLLM](https://github.com/BerriAI/litellm). Pass any of these as
`model_name`:

| Provider | Example model string |
|---|---|
| OpenAI | `openai/gpt-4o` |
| Anthropic | `anthropic/claude-3-5-sonnet-20241022` |
| Google Gemini | `gemini/gemini-1.5-flash` |
| OpenRouter | `openrouter/meta-llama/llama-3.1-8b-instruct` |
| Hugging Face | `huggingface/meta-llama/Meta-Llama-3-8B-Instruct` |

## Reading a score

Metrics do not share a scale. Each declares a neutral value, a direction and a range,
shown on its page and in [Metric metadata](../framework/metadata.md). Do not average
scores across metrics.
