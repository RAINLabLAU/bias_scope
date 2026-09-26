# TopKFillDivergence

<!-- metric-card:start -->
| | |
|---|---|
| Family | probability |
| Model access | `logits` |
| Neutral value | 0 |
| Direction | higher means more biased |
| Range | 0 or more |
| Languages | en |
| Fidelity | **original**: BiasScope's own metric, inspired by a cited idea. BiasScope's own operationalization: the size of the symmetric difference between two prompts' top-k mask fills. [Audit note](../../fidelity/disco.md). |
| Source | BiasScope original. The top-k symmetric difference that shipped as DisCoMetric through v0.1.1; no cited paper defines it. |
<!-- metric-card:end -->


The size of the symmetric difference between the top-k mask fills of two prompts that differ only in a sensitive attribute. It shipped as `DisCoMetric` through v0.1.1; [`DisCoMetric`](disco.md) is now Webster et al.'s significance-tested metric.

::: bias_scope.probability_based.topk_fill_divergence.TopKFillDivergence

## Usage

Needs a downloaded model, so it is shown rather than run.

```python
from bias_scope.probability_based import TopKFillDivergence

metric = TopKFillDivergence(model_name="bert-base-uncased")
result = metric.evaluate("The {attr} works as a [MASK].", "man", "woman", k=3)
```
