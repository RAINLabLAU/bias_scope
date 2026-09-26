# Architecture

BiasScope is a library of metrics with a thin framework on top, and an optional agent on
top of that. Each layer depends only on the one below.

```
   bias_scope_agent        conversational agent (optional, separate package)
            |
   BiasSuite  recommend_metrics  Report  viz      framework layer
            |
   Backend                                        what can be asked of the model
            |
   BiasMetric  ->  EmbeddingMetric | ProbabilityMetric | GeneratedTextMetric | PromptBasedMetric
            |
   MetricInfo                                     what a metric needs and how far to trust it
```

## The layers

**Metrics** live in `src/bias_scope/<family>_based/`, one module per metric. Every metric
inherits `BiasMetric` (in `base.py`) through one of four family base classes. It implements
`evaluate()`. The base class adds `run()`, which wraps `evaluate()` in a
[`BiasResult`](framework/results.md) with an interval, a protocol block and the metric's
metadata.

**Metadata** is a `MetricInfo` per metric, held in one static table (`_metric_info.py`) and
attached to the class. It records the access the metric needs, the neutral value and
direction, and its [fidelity](framework/metadata.md) to the cited paper. Everything above
reads it, and a test checks that these docs match it.

**Backends** wrap a model and declare its access: `embeddings`, `logits`, `completions`,
`chat`. [Access is derived from the backend, not guessed](framework/backends.md).

**The framework** turns those pieces into answers:
[`recommend_metrics`](framework/recommend.md) says which metrics a backend can run,
[`BiasSuite`](framework/suite.md) runs them, and `Report` renders the results.

**The agent** (`src/bias_scope_agent/`) is a separate package that drives the framework
through a language model and a confirm-before-run gate. It adds no metric-selection logic
and does not modify `bias_scope`. See the [agent guide](agent/index.md).

## Rules the code follows

- **A metric that cannot run is skipped with a reason, never scored as zero.**
- **No composite score.** Metrics with different units and neutral values do not average.
- **Every result carries its protocol**: model, dtype, seed, dataset revision and
  resources, so a number can be reproduced.
- **Fidelity is stated, not implied.** A metric that deviates from its paper says so, and
  the label is checked against the sources actually read.
- **Plain code.** No metaclasses, no operator overloading, short functions, and every metric
  docstring states its formula.
- **The core install stays light.** Heavy dependencies (torch, transformers, litellm) are
  optional extras, and a missing extra raises an error that names it.

## Where things are

```
src/bias_scope/
  base.py                 BiasMetric and the four family base classes
  metadata.py             MetricInfo, list_metrics, fidelity_counts
  _metric_info.py         the registry: one MetricInfo per metric
  backends.py             HuggingFaceBackend, LiteLLMBackend, StubBackend
  recommend.py  suite.py  choosing and running metrics
  result.py  report.py    BiasResult, Report, to_markdown, to_html, compare
  stats.py  viz.py        intervals and plots
  embeddings_based/  probability_based/  generated_text_based/  prompts_based/
src/bias_scope_agent/     the conversational agent
docs/fidelity/            one audit note per metric or group of metrics
sources/SOURCES.yaml      the papers and code read for each metric
validation/registry.yaml  published values each metric is checked against
```
