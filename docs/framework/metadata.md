# Metric metadata and fidelity

Every metric declares a `MetricInfo` record. It is what lets the rest of the library
answer questions about a metric without running it: which models can it run on, what
number means "no bias", and how far can the result be trusted.

```python
from bias_scope import list_metrics, fidelity_counts, normalized_deviation

info = list_metrics()["WEAT"]
print(info.family, info.access, info.neutral_value, info.direction, info.fidelity)
# embedding ('embeddings',) 0.0 signed faithful

normalized_deviation(0.8, info)   # signed distance from neutral; positive = more biased
list_metrics(family="probability", fidelity="original")   # filter the registry
fidelity_counts()                 # {'faithful': ..., 'adaptation': ..., ...}
```

`METRIC_INFO` is the single registry. The [metric pages](../api/overview.md), the
[fidelity index](../fidelity/INDEX.md), `recommend_metrics`, `BiasSuite`, the reports and
the [agent](../agent/index.md) all read it, and a test checks that these docs match it.

## Model access

A metric declares which of four access modes it needs. A backend declares which it
provides. A metric is offered for a backend only if the backend provides all of them.

| Access | What it means | Typical source |
|---|---|---|
| `embeddings` | Vector representations of text | An encoder, a sentence transformer, or a causal LM's hidden states |
| `logits` | Masked-token probabilities | A masked language model with a working masked-LM head |
| `completions` | Free-form text generated from a prompt | A causal LM or a chat API |
| `chat` | Chat-formatted messages | A chat API |

`logits` means **masked-token** logits specifically, because that is the only kind any
metric here consumes. A causal LM has next-token logits but no masked-token prediction,
so it does not advertise `logits`.

## Fidelity

Fidelity says how closely an implementation follows the paper it cites. It is set only
after the paper and the authors' code have been read, and it is checked against the
recorded sources by a test, so a metric cannot claim more than its evidence supports.

| Status | Meaning |
|---|---|
| `faithful` | Same formula and protocol. Any difference is a documented access-mode necessity that does not change the statistic. |
| `adaptation` | Same underlying comparison, but a different access mode or scoring path that **can** change the numbers. Read the deviation note before comparing to a published value. |
| `original` | BiasScope's own operationalization, inspired by a cited idea. It does not carry the cited paper's metric name. |
| `mismatch` | Implements a different statistic than its name claims. Excluded from recommendations by default. |
| `unaudited` | Sources not yet read, so no fidelity claim is made. |

An `adaptation` is not a bug. It is the honest label for a metric that had to change
something (for example, a chat API instead of a completions API) and says what.

## Direction and the neutral value

Metrics do not share a scale. Each declares:

- `neutral_value`: the score that means no bias. It is 0.5 for CrowS-Pairs, 0 for WEAT
  and 100 for iCAT, and for some TrustLLM metrics it is 1, where lower is worse.
- `direction`: `higher_more_biased`, `lower_more_biased`, or `signed`.
- `value_range`: the inclusive bounds of the score.

`normalized_deviation(score, info)` converts any score to a signed distance from
neutral where positive always means more biased. It is a display transform for plots, not
an aggregate: the library never sums or averages scores across metrics.

## API reference

::: bias_scope.metadata.MetricInfo

::: bias_scope.metadata.list_metrics

::: bias_scope.metadata.fidelity_counts

::: bias_scope.metadata.normalized_deviation
