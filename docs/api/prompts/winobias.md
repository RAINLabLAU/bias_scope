# WinoBias

<!-- metric-card:start -->
| | |
|---|---|
| Family | prompt |
| Model access | `chat` |
| Neutral value | 0 |
| Direction | signed; 0 is neutral |
| Range | -1 to 1 |
| Languages | en |
| Fidelity | **faithful**: same formula and protocol as the cited paper. [Audit note](../../fidelity/winobias.md). |
| Source | Gender Bias in Coreference Resolution: Evaluation and Debiasing Methods, NAACL 2018 — https://arxiv.org/abs/1804.06876 |
| Reference code | https://github.com/uclanlp/corefBias @ 0bce984dd081 |
<!-- metric-card:end -->


The gap in coreference accuracy between pro- and anti-stereotypical versions of the same sentences (Zhao et al., 2018).

::: bias_scope.prompts_based.winobias.WinoBias

## Example

```python
--8<-- "examples/prompts_based/winobias.py"
```
