# Bias report — `stub/embeddings`

Every score carries its interval and its fidelity status. **Scores from different metrics are not comparable and must not be averaged** — the library computes no composite score by design.

## Fidelity of the metrics run

| Status | Count |
|---|---|
| faithful | 1 |

## embedding

| Metric | Score | 95% CI | n | Normalised | Fidelity |
|---|---:|---|---:|---:|---|
| WEAT | -0.02389 | [-1.004, 0.9561] | 16 | -0.024 | faithful |

## Not run

| Metric | Reason |
|---|---|
| SEAT | BiasScopeError: SEAT: n must be positive, got 0. The metric scored no items, or does not report how many it scored. |

## Protocol

| Field | Value |
|---|---|
| metric | `BiasSuite` |
| model_id | `stub/embeddings` |
| seed | `42` |
| library_version | `0.1.1` |
| timestamp | `2026-08-22T22:46:59.754822+00:00` |
| hash | `bfea1f428504` |
| axis | `gender` |
| language | `en` |
