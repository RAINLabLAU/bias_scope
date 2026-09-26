# MeanScoreGap

<!-- metric-card:start -->
| | |
|---|---|
| Family | generated_text |
| Model access | `completions` |
| Neutral value | 0 |
| Direction | signed; 0 is neutral |
| Range | unbounded |
| Languages | en |
| Fidelity | **original**: BiasScope's own metric, inspired by a cited idea. BiasScope's own operationalization: the difference in mean classifier score between two groups, with Cohen's d. [Audit note](../../fidelity/score_parity.md). |
| Source | BiasScope original. Inspired by Borkan et al. 2019, Nuanced Metrics for Measuring Unintended Bias, WWW 2019 companion — https://arxiv.org/abs/1903.04561 |
<!-- metric-card:end -->


Compares any classifier's scores between two demographic groups in generated text.
Renamed from `ScoreParity` in 0.2.0. The statistic is unchanged, and `ScoreParity`
still imports (with a `DeprecationWarning`) until 0.3.0.

::: bias_scope.generated_text_based.mean_score_gap.MeanScoreGap

## Example

```python
--8<-- "examples/generated_text_based/mean_score_gap.py"
```

## Reference

Inspired by Borkan, D., Dixon, L., Sorensen, J., Thain, N., & Vasserman, L. (2019).
Nuanced Metrics for Measuring Unintended Bias with Real Data for Text Classification.
*WWW 2019 Companion*. BiasScope's version is its own operationalization and does not
carry that paper's metric name.
