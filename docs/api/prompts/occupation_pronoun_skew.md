# OccupationPronounSkew

<!-- metric-card:start -->
| | |
|---|---|
| Family | prompt |
| Model access | `chat` |
| Neutral value | 1 |
| Direction | signed; 0 is neutral |
| Range | 0 or more |
| Languages | en |
| Fidelity | **original**: BiasScope's own metric, inspired by a cited idea. BiasScope's own operationalization: the male/female pronoun-count ratio in generations for occupation templates. [Audit note](../../fidelity/demographic_representation_bias.md). |
| Source | BiasScope original. Inspired by Zhao et al. 2018 (WinoBias), Gender Bias in Coreference Resolution, NAACL 2018 — https://arxiv.org/abs/1804.06876 |
<!-- metric-card:end -->


Detects gender skew by counting the pronouns a model uses when it completes
occupation-based sentence templates from the WinoBias dataset.

Renamed from `DemographicRepresentationBias` in 0.2.0. The statistic is unchanged, and
the old name still imports (with a `DeprecationWarning`) until 0.3.0.

::: bias_scope.prompts_based.occupation_pronoun_skew.OccupationPronounSkew

## Example

```python
--8<-- "examples/prompts_based/occupation_pronoun_skew.py"
```

## Reference

Inspired by Zhao, J., Wang, T., Yatskar, M., Ordonez, V., & Chang, K.-W. (2018). Gender
Bias in Coreference Resolution: Evaluation and Debiasing Methods. *NAACL 2018*. The
templates come from that paper's WinoBias dataset; the metric itself is BiasScope's own.
See also [`WinoBias`](winobias.md), which implements the paper's accuracy-gap metric.
