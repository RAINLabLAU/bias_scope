# StereotypeRuleHitRate

<!-- metric-card:start -->
| | |
|---|---|
| Family | generated_text |
| Model access | `completions` |
| Neutral value | 0 |
| Direction | higher means more biased |
| Range | 0 or more |
| Languages | en |
| Fidelity | **original**: BiasScope's own metric, inspired by a cited idea. BiasScope's own operationalization: how often user-supplied stereotype rules match a generation, by token window or regex. [Audit note](../../fidelity/stereotypical_associations.md). |
| Source | BiasScope original. The rule/window hit-rate matcher that shipped as StereotypicalAssociations through v0.1.1; no cited paper defines it. |
<!-- metric-card:end -->


Counts how often a generation places rule-defined group terms and attribute terms within a token window of each other, per 1,000 generations. It shipped as `StereotypicalAssociations` through v0.1.1; [`StereotypicalAssociations`](stereotypical_associations.md) is now HELM's metric.

::: bias_scope.generated_text_based.stereotype_rule_hit_rate.StereotypeRuleHitRate

## Example

```python
--8<-- "examples/generated_text_based/stereotype_rule_hit_rate.py"
```
