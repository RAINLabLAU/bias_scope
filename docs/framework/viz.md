# Visualization

Plots need the `viz` extra (`pip install "bias-scope[viz]"`, which installs matplotlib).
Each plot is a plain function that takes a report and returns a matplotlib `Figure`.

```python
from bias_scope.viz import plot_profile, plot_dumbbell

fig = plot_profile(report)
fig.savefig("profile.png", dpi=200)

fig = plot_dumbbell(report_model_a, report_model_b)
```

| Function | Shows |
|---|---|
| `plot_profile(report)` | One row per metric, grouped by family. The x axis is the normalized deviation: 0 is neutral and positive is more stereotyped, whatever each metric's own direction. |
| `plot_dumbbell(report_a, report_b)` | Two models on one axis, a marker each, joined by a line. |
| `plot_agreement(matrix, n_models)` | Metric-by-metric correlation heatmap, from `correlate()`. |
| `plot_envelope(runs, metric)` | One metric's value across protocol variants, drawn as a band. |
| `plot_forest(result)` | Per-group breakdown for one metric, with the overall score marked. |
| `figure_to_svg(fig)` | A figure as an SVG string, for embedding. |

## What is deliberately not drawn

These are constraints on what a figure may show, not style preferences:

- **No composite score, gauge, traffic light or single-polygon radar.** Each of those
  implies the metrics share a scale. They do not.
- **Every plotted value carries its confidence interval.** A point estimate drawn alone
  claims a precision the metric does not have. A metric with no interval gets a marker
  only, so the difference is visible.
- **Every figure footer shows the protocol hash, seed, dtype and resource versions**, so a
  figure lifted into a paper still says how it was produced.
- **Fidelity appears as a badge**, so an `adaptation` bar cannot be read as a finding
  about the cited paper.

`plot_dumbbell` raises if the two reports cover different metric sets, and `plot_forest`
raises if the result has no per-group breakdown. An unmatched picture is a misleading
one, not a partial one.

`to_html(report, path, figures=True)` renders the profile plot, and a forest plot for
every metric with a breakdown, to inline SVG, so the report stays a single file.

## API reference

::: bias_scope.viz.plot_profile

::: bias_scope.viz.plot_dumbbell

::: bias_scope.viz.plot_agreement

::: bias_scope.viz.plot_envelope

::: bias_scope.viz.plot_forest

::: bias_scope.viz.figure_to_svg
