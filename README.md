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
pip install "bias-scope[all]"
```

What each extra includes:

- `torch`: `torch`, `transformers` for probability-based masked-token metrics, `BertPLLScorer`, and transformer-backed generated-text metrics such as `RegardScore`
- `embeddings`: `sentence-transformers` for the built-in embedding helper used by embedding-based convenience paths
- `datasets`: `datasets` for prompt-based benchmark loaders
- `llm`: `litellm` for prompt-based model calls
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
