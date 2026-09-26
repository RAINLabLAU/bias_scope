# Implicit association and decision bias

<!-- metric-card:start -->
**`ImplicitAssociationTest`**

| | |
|---|---|
| Family | prompt |
| Model access | `chat`, `completions` |
| Neutral value | 0 |
| Direction | signed; 0 is neutral |
| Range | -1 to 1 |
| Languages | en |
| Fidelity | **faithful**: same formula and protocol as the cited paper. Uses the reference code's 0.01 divide-by-zero guard in both denominators (analysis/clean.ipynb, `d_score`), which the paper omits; pass epsilon=0.0 for the published formula exactly. [Audit note](../../fidelity/implicit_association.md). |
| Source | Explicitly unbiased large language models still form biased associations, PNAS 2025 — https://arxiv.org/abs/2402.04105 |
| Reference code | https://github.com/baixuechunzi/llm-implicit-bias @ 0d2772e8eb21 |

**`LLMDecisionBias`**

| | |
|---|---|
| Family | prompt |
| Model access | `chat`, `completions` |
| Neutral value | 0.5 |
| Direction | signed; 0 is neutral |
| Range | 0 to 1 |
| Languages | en |
| Fidelity | **faithful**: same formula and protocol as the cited paper. [Audit note](../../fidelity/implicit_association.md). |
| Source | Explicitly unbiased large language models still form biased associations, PNAS 2025, Sec. 2.2 — https://arxiv.org/abs/2402.04105 |
| Reference code | https://github.com/baixuechunzi/llm-implicit-bias @ 0d2772e8eb21 |
<!-- metric-card:end -->


Two psychology-inspired probes from Bai et al. (PNAS 2025): a word-association task and a paired decision task.

::: bias_scope.prompts_based.implicit_association.ImplicitAssociationTest
::: bias_scope.prompts_based.implicit_association.LLMDecisionBias

## Example

```python
--8<-- "examples/prompts_based/implicit_association.py"
```
