# Running a suite

`BiasSuite` is the unified interface: give it a model, and it selects the metrics that
model can support, runs them, and returns one `Report`.

```python
--8<-- "examples/framework/suite.py"
```

## Two commitments

**There is no composite score.** `Report` has no overall field and never will. Metrics
with different units, ranges and neutral values do not average into anything meaningful,
and a test asserts that the field never appears.

**A metric that cannot run is recorded as skipped, never as a zero.** A missing number is
information; a fabricated one is a defect. Every skipped metric carries its reason in
`report.skipped`.

## Planning first

`suite.plan()` returns the `(name, MetricInfo)` pairs the suite would run without running
anything. It is cheap, and it shows exactly what the backend can and cannot support.
Metrics are chosen the same way `recommend_metrics` chooses them.

To run exactly the metrics you name, pass `metrics=[...]`. They are still checked against
the backend's access, because asking for a metric the model cannot support is an error,
not a preference.

## Supplying inputs

Most metrics need stimuli the suite cannot invent: word lists, sentence pairs, templates
or a dataset. Pass them per metric, as the keyword arguments of that metric's
`evaluate()`:

```python
report = suite.run(
    seed=42,
    inputs={
        "WEAT": {"target_embeddings": (X, Y), "attribute_embeddings": (A, B)},
        "CrowSPairs": {"__init__": {"model_name": "bert-base-uncased"},
                       "sentence_pairs": pairs},
    },
)
```

Two reserved keys:

- `"__init__"` holds constructor arguments, for metrics that need one (a `model_name`, or
  a scorer). It is removed before the remaining arguments reach `evaluate()`.
- `"__protocol__"` lets whoever prepared the data record what it substituted (a
  different classifier or corpus than the paper used). It lands in the result's protocol
  block, which a fidelity badge alone cannot express.

A metric with no entry in `inputs` is skipped with the reason
`requires caller-supplied stimuli`. The [agent](../agent/index.md) fills these in from
the authors' own dataset files.

## Errors

By default `run(on_error="skip")` records any exception from one metric as that metric's
skip reason and carries on, so one failure does not discard the other results. Pass
`on_error="raise"` to propagate instead. A failure is never replaced with a value.

## Comparing runs

`compare(report_a, report_b)` gives paired per-metric deltas. Each delta carries both
confidence intervals and `intervals_overlap`: overlapping intervals mean the difference
should not be read as real. `compare` raises if the two reports cover different metric
sets, because a table over different metrics only looks like a model comparison.

`correlate(reports)` asks whether two metrics rank models the same way. It needs at
least three reports, because with two models every correlation is exactly plus or minus
one.

## API reference

::: bias_scope.suite.BiasSuite

::: bias_scope.report.compare

::: bias_scope.report.correlate

::: bias_scope.report.Delta
