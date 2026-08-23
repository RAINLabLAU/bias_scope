# Bias report — `stub/chat`

Every score carries its interval and its fidelity status. **Scores from different metrics are not comparable and must not be averaged** — the library computes no composite score by design.

## Fidelity of the metrics run

| Status | Count |
|---|---|
| adaptation | 1 |
| faithful | 8 |

## prompt

| Metric | Score | 95% CI | n | Normalised | Fidelity |
|---|---:|---|---:|---:|---|
| DecodingTrustFairness | 1 | — | 6 | +1.000 | faithful |
| DecodingTrustStereotype | 0.5 | [0, 1] | 2 | +0.500 | faithful |
| FirstPersonFairness | 0.6 | [0.6, 0.6] | 5 | +0.600 | ADAPTATION |
| ImplicitAssociationTest | 0.995 | [0.995, 0.995] | 4 | +0.995 | faithful |
| LLMDecisionBias | 0.75 | [0.25, 1] | 4 | +0.500 | faithful |
| TrustLLMDisparagement | 1.38e-06 | — | 30 | +1.000 | faithful |
| TrustLLMPreference | 0.5 | [0, 1] | 2 | +0.500 | faithful |
| TrustLLMStereotypeAgreement | 0.25 | [0, 0.75] | 4 | +0.250 | faithful |
| TrustLLMStereotypeRecognition | 0.75 | [0.25, 1] | 4 | +0.250 | faithful |
