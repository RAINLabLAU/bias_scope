# IdentitySwapConsistency

<!-- metric-card:start -->
| | |
|---|---|
| Family | prompt |
| Model access | `chat` |
| Neutral value | 0 |
| Direction | higher means more biased |
| Range | 0 to 1 |
| Languages | en |
| Fidelity | **original**: BiasScope's own metric, inspired by a cited idea. BiasScope's own operationalization: cosine similarity between response embeddings under identity swaps. [Audit note](../../fidelity/originals.md). |
| Source | BiasScope original. Inspired by Kusner et al. 2017, Counterfactual Fairness, NeurIPS 2017 — https://arxiv.org/abs/1703.06856 |
<!-- metric-card:end -->


Checks whether a model's outputs stay semantically consistent when only the identity
term in a prompt changes (for example "Muslim" to "Christian"). Both prompts are sent
to the model, the two responses are embedded, and their cosine similarity is computed.

Renamed from `CounterfactualFairness` in 0.2.0. The statistic is unchanged, and the old
name still imports (with a `DeprecationWarning`) until 0.3.0. It was renamed because
`CounterfactualFairness` is the exact title of Kusner et al.'s paper, which defines a
causal criterion, and the old name read as a claim to implement it. This is a
consistency check inspired by that idea.

::: bias_scope.prompts_based.identity_swap_consistency.IdentitySwapConsistency

## Example

```python
--8<-- "examples/prompts_based/identity_swap_consistency.py"
```

## Reference

Inspired by Kusner, M. J., Loftus, J., Russell, C., & Silva, R. (2017). Counterfactual
Fairness. *NeurIPS 2017*.
