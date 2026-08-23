# Bias report — `bert-base-uncased`

Every score carries its interval and its fidelity status. **Scores from different metrics are not comparable and must not be averaged** — the library computes no composite score by design.

## Fidelity of the metrics run

| Status | Count |
|---|---|
| faithful | 4 |

## embedding

| Metric | Score | 95% CI | n | Normalised | Fidelity |
|---|---:|---|---:|---:|---|
| WEAT | 0.3688 | [-0.6195, 1.357] | 16 | +0.369 | faithful |

## probability

| Metric | Score | 95% CI | n | Normalised | Fidelity |
|---|---:|---|---:|---:|---|
| AUL | 50 | — | 6 | +0.000 | faithful |
| AULA | 50 | — | 6 | +0.000 | faithful |
| CrowSPairs | 66.67 | — | 6 | +0.333 | faithful |

## Not run

| Metric | Reason |
|---|---|
| DisCoMetric | requires caller-supplied stimuli (word lists, templates or a dataset); pass them via `inputs={metric: {...}}` |

## Protocol

| Field | Value |
|---|---|
| metric | `BiasSuite` |
| model_id | `bert-base-uncased` |
| dtype | `fp32` |
| seed | `42` |
| library_version | `0.1.1` |
| timestamp | `2026-08-23T17:22:36.347242+00:00` |
| hash | `e94234351a8b` |
| axis | `gender` |
| language | `en` |
