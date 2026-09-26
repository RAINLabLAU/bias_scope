# DecodingTrust

<!-- metric-card:start -->
**`DecodingTrustStereotype`**

| | |
|---|---|
| Family | prompt |
| Model access | `chat`, `completions` |
| Neutral value | 0 |
| Direction | higher means more biased |
| Range | 0 to 1 |
| Languages | en |
| Fidelity | **faithful**: same formula and protocol as the cited paper. Reports the agreement rate itself (0 neutral, higher more biased) rather than the benchmark's `1 - round(mean, 2)` leaderboard score, which is in details["decodingtrust_score"] with the same rounding. [Audit note](../../fidelity/decodingtrust.md). |
| Source | DecodingTrust: A Comprehensive Assessment of Trustworthiness in GPT Models, NeurIPS 2023 Datasets and Benchmarks — https://arxiv.org/abs/2306.11698 |
| Reference code | https://github.com/AI-secure/DecodingTrust @ 161ae8321ced |

**`DecodingTrustFairness`**

| | |
|---|---|
| Family | prompt |
| Model access | `chat`, `completions` |
| Neutral value | 0 |
| Direction | higher means more biased |
| Range | 0 to 1 |
| Languages | en |
| Fidelity | **faithful**: same formula and protocol as the cited paper. Reports the demographic parity difference itself rather than the benchmark's (1 - DPD) * 100 leaderboard score, which is in details. [Audit note](../../fidelity/decodingtrust.md). |
| Source | DecodingTrust, NeurIPS 2023 Datasets and Benchmarks, Sec. 8 (fairness) — https://arxiv.org/abs/2306.11698 |
| Reference code | https://github.com/AI-secure/DecodingTrust @ 161ae8321ced |
<!-- metric-card:end -->


Two metrics from the DecodingTrust benchmark (Wang et al., 2023): how often a model agrees with stereotype statements, and the demographic-parity gap of a model used as a classifier on tabular records.

::: bias_scope.prompts_based.decodingtrust.DecodingTrustStereotype
::: bias_scope.prompts_based.decodingtrust.DecodingTrustFairness

## Example

```python
--8<-- "examples/prompts_based/decodingtrust.py"
```
