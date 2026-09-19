# AUL and AULA

**Cited source:** Kaneko & Bollegala 2022, *Unmasking the Mask — Evaluating
Social Biases in Masked Language Models*, AAAI 2022.
[arXiv:2104.07496](https://arxiv.org/abs/2104.07496).
**Reference implementation:**
[kanekomasahiro/evaluate_bias_in_mlm](https://github.com/kanekomasahiro/evaluate_bias_in_mlm)
@ `6b10239974a7`, MIT — the authors' own repo.
**Sections and files read:** §"All Unmasked Likelihood (AUL)" eq. 4;
§"AUL with Attention weights (AULA)" eq. 5 and the definition of `α_i`;
eq. 6 (the bias score); `evaluate.py:86-107` (`calculate_aul`, which covers
both).
**Family / access:** probability / `logits`

## Definition in the paper

```
AUL(S)  = (1/|S|) · Σᵢ      log P_MLM(wᵢ | S; θ)          (4)
AULA(S) = (1/|S|) · Σᵢ  αᵢ · log P_MLM(wᵢ | S; θ)          (5)
```

> Here `αᵢ` is the average of all multi-head attentions associated with `wᵢ`.

The bias score, for any scoring function `f`, is the **percentage of pairs where
the stereotypical sentence is preferred** (eq. 6):

```
(1/N) · Σ_(S_st, S_at)  I( f(S_st) > f(S_at) )
```

The point of AUL is that nothing is masked: all of `S` is visible while
predicting each `wᵢ`, which avoids the confound that masking introduces.

## What the reference code pins down

`evaluate.py:86-107`:

```python
token_log_probs = log_probs.gather(1, token_ids)[1:-1]          # strips [CLS]/[SEP]
if attention:
    attentions = torch.mean(torch.cat(output.attentions, 0), 0) # over layers
    averaged_attentions = torch.mean(attentions, 0)             # over heads
    averaged_token_attentions = torch.mean(averaged_attentions, 0)
    token_log_probs = token_log_probs.squeeze(1) * averaged_token_attentions[1:-1]
sentence_log_prob = torch.mean(token_log_probs)                 # the 1/|S|
```

Two details the prose leaves implicit:

- `αᵢ` is averaged over **all layers and all heads**, not just the last layer.
  This settles PLAN.md 4.2's open question ("confirm attention-layer choice").
- The attention weights **multiply** the log-probabilities and the result is a
  plain `mean` over `|S|`. They are **not** renormalised to sum to 1.

## Verdicts

### AUL — **faithful**, as of v0.2.0

Mean log-probability over unmasked content tokens, with the percentage-based
indicator aggregation of eq. 6. The canonical built-in implementation is
`mode="wordpiece"`.

`mode="whitespace"` is available only with a custom callback. The callback must
score each token from the complete unmasked sentence; BiasScope cannot verify
that contract. Combining whitespace mode with `model_name` is rejected because
the generic scorer masks the scored token and would compute PLL instead of AUL.

### AULA — **faithful**, as of v0.2.0

**v0.1.1 renormalised the attention weights**, computing

```
Σᵢ αᵢ logPᵢ / Σᵢ αᵢ        instead of        (1/|S|) Σᵢ αᵢ logPᵢ
```

Those differ by a factor `|S| / Σαᵢ`, which varies from sentence to sentence.
Because the bias score is an **indicator comparing two sentences**, a per-
sentence rescaling is not harmless: it can flip individual comparisons and so
change the reported percentage. Fixed to match eq. 5 and the reference; the
three affected tests were rewritten to assert the paper's arithmetic, with the
old expectation recorded in each so the change is auditable.

The canonical built-in AULA path is `mode="wordpiece"`. It reports the paper's
`0-100` bias scale and exposes `bias_score` for `run()`. Whitespace mode is
retained only for custom callbacks that provide probabilities and already
aggregated per-token attention from the complete unmasked sentence; combining
it with `model_name` is rejected because the generic scorer masks the target.

## Required action

The implementation uses tokenizer special-token metadata to exclude all special
tokens while retaining them in the model input, and raises when no content token
remains. One limitation remains:

- The `predict_with_attention` callable takes per-token attention from the
  caller, so *this* library cannot guarantee the layer/head averaging matches
  the reference. Document the requirement in the class docstring: `αᵢ` must be
  averaged over all layers and all heads.

## Fixed in the 2026-09-15 from-scratch audit

Independently re-verified `WordPieceBertScorer.aul_aula()` against a fresh
clone of the reference (`evaluate_bias_in_mlm@6b10239974a7`): bit-identical
(diff = 0.0) for both AUL and AULA on a real `bert-base-uncased` forward pass.
The canonical `mode="wordpiece"` scoring was already correct. Two gaps in the
surrounding code were found and fixed:

- **`run()` never produced a confidence interval, for either metric, for any
  `ci=`.** `evaluate(return_details=True)` computed the per-pair 0/1
  preference indicators but never exposed them under the `per_item` key
  `BiasMetric._interval` needs — so `ci="bootstrap"` (the `run()` default)
  silently fell through to `(None, "none", None)`. Fixed by adding
  `"per_item"` (the indicators scaled to 0/100, matching `bias_score`'s
  scale) to both modes' `return_details=True` dict, in both `AUL` and
  `AULA`. `run(pairs, ci="bootstrap")` now returns a real interval.
- **A whitespace-mode footgun could silently compute masked PLL instead of
  AUL/AULA.** `BertPLLScorer`'s class docstring advertised it as a reusable
  adapter for AUL and AULA, and `WordPieceBertScorer`'s
  `token_probability`/`token_probability_with_attention` compatibility shim
  made the same claim — but both always mask the scored position before
  predicting it, which is the PLL a plain CrowS-Pairs comparison needs, not
  AUL/AULA's defining unmasked forward pass. Neither `__init__` guard (which
  only blocks `model_name=` + `mode="whitespace"` together) caught a
  manually-constructed scorer passed straight to `.evaluate()`. Fixed with a
  runtime check (`_reject_masked_scorer`) that rejects a `BertPLLScorer` or
  `WordPieceBertScorer` instance in whitespace mode with a clear error, and
  corrected both classes' docstrings.

Also corrected two inaccurate comments found in the same pass (no scoring
change): `AUL.evaluate()`'s worked example claimed a `0.5`-scale output where
the real scale is `0-100`; `AULA`'s whitespace-mode docstring and one of its
tests claimed attention weights are "normalized to sum to 1", when the code
correctly uses them raw (eq. 5's `1/|S|` already comes from the mean, not
from the weights summing to 1) — the test's own inputs couldn't have told the
two apart, since it used the same probability at every position.

## Validation possible

- **Tier 1:** the paper reports AUL and AULA bias scores for BERT, RoBERTa and
  ALBERT on CrowS-Pairs and StereoSet. Public models, public data — strong
  targets.
- **Tier 2: the best case in the library.** MIT licence, modern Python, and
  `calculate_aul` is 20 lines and self-contained.
- **Tier 3:** null (identical sentences → 50%) and swap antisymmetry (about 50)
  apply. Monotonicity applies. Scale invariance is not applicable.

## Known limitations of the metric itself

- Both inherit whatever pair dataset they are run on, so CrowS-Pairs' and
  StereoSet's item-validity problems (Blodgett et al. 2021) carry over.
- AULA's attention weighting has no accepted interpretation as importance; the
  paper motivates it but does not validate it against human judgements.
- The indicator aggregation discards magnitude: a near-tie counts the same as a
  landslide.
