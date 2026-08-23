# Bias report — `bert-base-uncased`

Every score carries its interval and its fidelity status. **Scores from different metrics are not comparable and must not be averaged** — the library computes no composite score by design.

## Fidelity of the metrics run

| Status | Count |
|---|---|
| faithful | 2 |

## embedding

| Metric | Score | 95% CI | n | Normalised | Fidelity |
|---|---:|---|---:|---:|---|
| SEAT | 0.3688 | [-0.6195, 1.357] | 16 | +0.369 | faithful |
| WEAT | 0.3688 | [-0.6195, 1.357] | 16 | +0.369 | faithful |

## Not run

| Metric | Reason |
|---|---|
| CEAT | requires caller-supplied stimuli (word lists, templates or a dataset); pass them via `inputs={metric: {...}}` |

## Protocol

| Field | Value |
|---|---|
| metric | `BiasSuite` |
| model_id | `bert-base-uncased` |
| dtype | `fp32` |
| seed | `42` |
| library_version | `0.1.1` |
| timestamp | `2026-08-23T00:20:18.313784+00:00` |
| hash | `e94234351a8b` |
| axis | `gender` |
| language | `en` |
