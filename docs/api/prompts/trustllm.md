# TrustLLM fairness

<!-- metric-card:start -->
**`TrustLLMStereotypeRecognition`**

| | |
|---|---|
| Family | prompt |
| Model access | `chat`, `completions` |
| Neutral value | 1 |
| Direction | lower means more biased |
| Range | 0 to 1 |
| Languages | en |
| Fidelity | **faithful**: same formula and protocol as the cited paper. Reports accuracy directly, as the benchmark does. 1.0 is the ideal, not 0.0, so `neutral_value` is 1.0 and profile plots measure deviation downward from it. [Audit note](../../fidelity/trustllm.md). |
| Source | TrustLLM: Trustworthiness in Large Language Models, ICML 2024 — https://arxiv.org/abs/2401.05561 (fairness, stereotype recognition) |
| Reference code | https://github.com/HowieHwong/TrustLLM @ 4b864211f326 |

**`TrustLLMStereotypeAgreement`**

| | |
|---|---|
| Family | prompt |
| Model access | `chat`, `completions` |
| Neutral value | 0 |
| Direction | higher means more biased |
| Range | 0 to 1 |
| Languages | en |
| Fidelity | **faithful**: same formula and protocol as the cited paper. The denominator is every response, so an unclear answer counts against agreement rather than being excluded — unlike TrustLLMStereotypeRecognition, which drops its invalid answers. [Audit note](../../fidelity/trustllm.md). |
| Source | TrustLLM: Trustworthiness in Large Language Models, ICML 2024 — https://arxiv.org/abs/2401.05561 (fairness, stereotype agreement) |
| Reference code | https://github.com/HowieHwong/TrustLLM @ 4b864211f326 |

**`TrustLLMDisparagement`**

| | |
|---|---|
| Family | prompt |
| Model access | `chat`, `completions` |
| Neutral value | 1 |
| Direction | lower means more biased |
| Range | 0 to 1 |
| Languages | en |
| Fidelity | **faithful**: same formula and protocol as the cited paper. Reports the smallest p-value across the tested attributes rather than a per-attribute dict, so a mean cannot let an independent attribute mask a dependent one. [Audit note](../../fidelity/trustllm.md). |
| Source | TrustLLM: Trustworthiness in Large Language Models, ICML 2024 — https://arxiv.org/abs/2401.05561 (fairness, disparagement) |
| Reference code | https://github.com/HowieHwong/TrustLLM @ 4b864211f326 |

**`TrustLLMPreference`**

| | |
|---|---|
| Family | prompt |
| Model access | `chat`, `completions` |
| Neutral value | 1 |
| Direction | lower means more biased |
| Range | 0 to 1 |
| Languages | en |
| Fidelity | **faithful**: same formula and protocol as the cited paper. The refusal classifier is a caller-supplied callable rather than a hardwired LibrAI/longformer-action-ro, so the classifier version is recorded in the protocol instead of being implicit. [Audit note](../../fidelity/trustllm.md). |
| Source | TrustLLM: Trustworthiness in Large Language Models, ICML 2024 — https://arxiv.org/abs/2401.05561 (fairness, preference) |
| Reference code | https://github.com/HowieHwong/TrustLLM @ 4b864211f326 |
<!-- metric-card:end -->


Four fairness metrics from the TrustLLM benchmark (Huang et al., 2024). The neutral value differs between them, so check the card for each before reading a score.

::: bias_scope.prompts_based.trustllm.TrustLLMStereotypeRecognition
::: bias_scope.prompts_based.trustllm.TrustLLMStereotypeAgreement
::: bias_scope.prompts_based.trustllm.TrustLLMDisparagement
::: bias_scope.prompts_based.trustllm.TrustLLMPreference

## Example

```python
--8<-- "examples/prompts_based/trustllm.py"
```
