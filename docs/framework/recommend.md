# Choosing metrics

`recommend_metrics` answers a question that used to need expertise: *which metrics can I
actually run on this model, and how much should I trust each one?*

```python
--8<-- "examples/framework/recommend.py"
```

## How it decides

The access filter is a hard constraint, taken from the backend. A chat API cannot provide
logits, so a metric that needs them is not offered, however popular it is. Fidelity is
advisory but always attached: every recommendation carries a reason string, and an
adaptation, original, unaudited or mismatch metric says so in it.

Results are sorted most trustworthy first: `faithful`, then `adaptation`, `original`,
`unaudited` and `mismatch`, alphabetically within each.

| Argument | Default | Effect |
|---|---|---|
| `access` | required | What the backend provides. Take it from `Backend.access`. |
| `axis` | `None` | Bias axis, such as `"gender"`. Recorded in the reason. |
| `language` | `"en"` | A metric is offered only if it declares support for the language. |
| `include_mismatch` | `False` | Offer metrics known to implement a different statistic than they cite. |
| `include_unaudited` | `True` | Offer metrics whose sources have not been read. Their reason carries the warning. |
| `include_non_bias` | `False` | Offer `TruthfulQA` and `TofNof`, which measure truthfulness and sycophancy, not social bias. |
| `family` | `None` | Restrict to one family. |

## Why a metric was left out

`explain_exclusions` is the other half of the answer. A recommender that silently drops
most of the library is hard to trust, so this says, per metric, which constraint removed
it: a missing access mode, missing resources for the language, a `mismatch` status, or
being outside the scope of social bias (`TruthfulQA` and `TofNof`).

The [agent](../agent/index.md) calls both functions and shows you the result, so you never
have to guess why a metric is missing from a plan.

## API reference

::: bias_scope.recommend.recommend_metrics

::: bias_scope.recommend.explain_exclusions

::: bias_scope.recommend.Recommendation
