# PairwiseLikelihoodPreference

<!-- metric-card:start -->
| | |
|---|---|
| Family | probability |
| Model access | `logits` |
| Neutral value | 0.5 |
| Direction | higher means more biased |
| Range | 0 to 1 |
| Languages | en |
| Fidelity | **original**: BiasScope's own metric, inspired by a cited idea. BiasScope's own operationalization: the proportion of sentence pairs where the stereotype sentence has the higher log-probability. [Audit note](../../fidelity/lpbs.md). |
| Source | BiasScope original. The sentence-pair preference rate shipped as LPBS through v0.1.1; the comparison is the CrowS-Pairs / StereoSet family's, not any single cited paper's. |
<!-- metric-card:end -->


The share of (stereotype, anti-stereotype) sentence pairs where the model gives the stereotype the higher log-probability. It shipped as `LPBS` through v0.1.1; [`LPBS`](lpbs.md) is now Kurita et al.'s metric.

::: bias_scope.probability_based.pairwise_likelihood_preference.PairwiseLikelihoodPreference

## Example

```python
--8<-- "examples/probability_based/pairwise_likelihood_preference.py"
```
