# Results, protocols and reports

`evaluate()` keeps returning what it always did: a float, or a dict with
`return_details=True`. The framework layer uses `run()` instead, which returns a
`BiasResult` carrying everything needed to interpret and reproduce the number.

```python
result = metric.run(target_embeddings=(X, Y), attribute_embeddings=(A, B), seed=42)

result.score          # the headline number, inside info.value_range
result.n              # items actually scored, not items requested
result.ci             # 95% interval, or None when none is defined
result.ci_method      # "bootstrap" | "wald" | "hedges_olkin" | "permutation" | "none"
result.per_item       # item-level scores, when the metric has them
result.breakdown      # per-group or per-category scores
result.protocol       # how the number was produced
result.info           # the metric's MetricInfo, including its fidelity
result.normalized_deviation()   # signed distance from neutral; positive = more biased
```

A `BiasResult` is a plain record. It is not a float look-alike, so reading `.score` is
how you get the number.

## Runtime guards

`run()` raises `BiasScopeError` rather than return a result that cannot be right: a score
outside the metric's declared range, a non-positive `n`, a non-finite score, or an
interval that does not contain the score.

## Intervals

Every result carries an interval wherever one is definable, and none is ever widened or
narrowed to make a reproduction pass. The method depends on the metric:

| Method | Used for |
|---|---|
| Bootstrap | Metrics with item-level scores |
| Wald | Proportions |
| Hedges and Olkin | Effect sizes such as WEAT's Cohen's d |
| Permutation p-value | The WEAT family's significance test, reported as `p_value` |
| None | Metrics with no item-level scores and no interval defined by the paper. The result says `ci_method="none"` instead of implying a precision it lacks. |

The four functions are public, and each states its formula in its docstring.

## The protocol block

A bias number without its protocol is not reproducible: the same metric on the same model
moves with the dtype, the decoding settings, the lexicon version and the judge. Every
result therefore carries a protocol with a fixed set of keys:

`metric`, `model_id`, `dtype`, `seed`, `dataset`, `dataset_revision`, `decoding`,
`resources`, `judge_model`, `judge_prompt_version`, `permutation_seed`, `random_seed`,
`library_version`, `timestamp`, and `hash`.

`hash` covers every field except `timestamp`, so two runs of the same protocol at
different times share a hash. When a run substitutes a resource the paper used (a
different classifier, or a different corpus), the substitution is written into the
protocol, and the [agent](../agent/index.md)'s summary prints it as a `deviation:` line
under the score.

## Reports

A `Report` holds one model's results across many metrics, plus the metrics that were
skipped and why.

```python
from bias_scope import save_json, to_html, to_markdown

print(to_markdown(report))                 # every score with its interval and fidelity
to_html(report, "report.html")             # one self-contained file, opens offline
to_html(report, "report.html", figures=True)   # adds inline SVG plots (viz extra)
save_json(report, "report.json")           # round-trips through bias_scope.result.from_dict
```

Every rendered value carries its confidence interval and its fidelity status. That is not
decoration: a number from a `mismatch` metric and a number from a `faithful` one mean
different things, and a report that printed them identically would invite the reader to
average them.

## API reference

::: bias_scope.result.BiasResult

::: bias_scope.result.make_protocol

::: bias_scope.result.from_dict

::: bias_scope.report.Report

::: bias_scope.report.to_markdown

::: bias_scope.report.to_html

::: bias_scope.report.save_json

::: bias_scope.stats.bootstrap_ci

::: bias_scope.stats.wald_ci

::: bias_scope.stats.hedges_olkin_ci

::: bias_scope.stats.permutation_p
