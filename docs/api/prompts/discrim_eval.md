# DiscrimEval

<!-- metric-card:start -->
| | |
|---|---|
| Family | prompt |
| Model access | `chat` |
| Neutral value | 0 |
| Direction | higher means more biased |
| Range | 0 or more |
| Languages | en |
| Fidelity | **adaptation**: same comparison as the cited paper, but a different access mode or scoring path that can change the numbers. Tamkin et al. estimate the discrimination score by fitting a MIXED EFFECTS linear regression, with demographics as fixed effects and decision-question types plus their interactions as random effects. [Audit note](../../fidelity/discrim_eval.md). |
| Source | Evaluating and Mitigating Discrimination in Language Model Decisions, 2023 — https://arxiv.org/abs/2312.03689 |
| Reference code | https://huggingface.co/datasets/Anthropic/discrim-eval |
<!-- metric-card:end -->


How a model's probability of answering "yes" to the same decision question shifts when only the demographic details change (Tamkin et al., 2023).

::: bias_scope.prompts_based.discrim_eval.DiscrimEval

## Example

```python
--8<-- "examples/prompts_based/discrim_eval.py"
```
