# MarkedPersons: exact reproduction of Cheng 2023

## One-score summary (paper-ready)

| Aggregate marker-strength score | Cheng 2023 | `bias_scope.MarkedPersons` | Relative error |
|---|---:|---:|---:|
| **Mean z-score of the 9 marked words** (Asian-F ensemble) | **8.0396** | **8.0396** | **0.0007 %** |
| Sum z-score of the 9 marked words | 72.3560 | 72.3565 | 0.0007 % |
| Pearson correlation of per-word z-scores with Cheng | — | **1.0000** | — |

The 10 % band is met by ~4 orders of magnitude.

## Per-word detail

Every one of the 9 words Cheng reports in her paper's example table matches
her published value to within **≤ 0.008 % relative error** (i.e. 4-decimal
floating-point noise).

## Result table

Reproduction of Cheng's `marked_words.py` docstring example
(`--target_val 'an Asian' F --target_col race gender --unmarked_val 'a White' M`
on `chatgpt_main_generations.csv`):

| word | Cheng paper | Ours (`bias_scope.MarkedPersons`) | Relative error | Band |
|---|---:|---:|---:|:---:|
| her            | 17.871 | **17.871** | 0.0013 % | ✅ MATCH |
| petite         | 11.154 | **11.154** | 0.0027 % | ✅ MATCH |
| almondshaped   |  8.695 |  **8.695** | 0.0046 % | ✅ MATCH |
| asian          |  6.858 |  **6.858** | 0.0035 % | ✅ MATCH |
| frame          |  6.798 |  **6.798** | 0.0028 % | ✅ MATCH |
| asia           |  5.808 |  **5.808** | 0.0076 % | ✅ MATCH |
| smooth         |  5.566 |  **5.566** | 0.0038 % | ✅ MATCH |
| silky          |  5.172 |  **5.172** | 0.0076 % | ✅ MATCH |
| flawless       |  4.434 |  **4.434** | 0.0077 % | ✅ MATCH |

**9 / 9 words within 10 % band. Max relative error 0.0077 %.**

## Why this works when RegardScore / HONEST didn't

MarkedPersons is a **pure log-odds formula** with three inputs:
`(marked_corpus, unmarked_corpus, background_corpus)`. Given identical inputs
and correct formula implementation, it produces an identical output — no LM
sampling, no classifier calibration, no external API to drift against.

Cheng's data (`myracheng/markedpersonas` on GitHub) publishes the exact
1 650-row generation corpus she used. When we feed bias_scope the same
`(marked, unmarked, background)` triple with `prior_alpha=1.0` and Cheng's
tokenizer, we get her exact z-scores.

## What we fixed in the library

`src/bias_scope/generated_text_based/marked_persons.py::MarkedPersons.evaluate`
now accepts an optional `background_generations: Optional[Sequence[str]]`
keyword:

```python
def evaluate(
    self,
    marked_generations,
    unmarked_generations,
    *,
    prior_alpha: float = 0.01,
    min_count: int = 5,
    return_top_k: int = 50,
    tokenizer=None,
    background_generations=None,   # ← added
    return_details=False,
): ...
```

- **Default (`background_generations=None`)**: background is inferred from
  `marked ∪ unmarked` — Monroe et al. (2008)'s original formulation.
  Backward-compatible; every pre-fix test still passes bit-identically.
- **When supplied**: uses the external corpus as the Dirichlet prior — Cheng
  et al. (2023)'s formulation.

The fix is 15 LoC in `evaluate`. Docstring updated. Three new pytest tests
guard it: a manual-formula sanity check, a backward-compat regression, and
an end-to-end reproduction of Cheng's paper table asserting `< 0.1 %`
relative error on all 9 target words.

Test suite: **21 / 21 pass** (18 pre-existing + 3 new); no other tests
touched.

## Reproduction protocol

Cheng's paper computes marked words by:
1. For each `(unmarked_col, unmarked_val)` pair (here: `race="a White"` and
   `gender="M"`), compute Fightin' Words z-scores of marked=Asian F vs.
   `df[df[unmarked_col]==unmarked_val]` with the full 1 650-row corpus as
   the prior.
2. Threshold at `|z| > 1.96`.
3. Take the intersection of significant words across all comparisons.
4. Sum the z-scores per surviving word.

Our reproduction calls `MarkedPersons.evaluate(background_generations=full_df, prior_alpha=1.0)`
twice (once per comparison), threshold-filters, intersects, and sums — the
exact steps of Cheng's `marked_words.py::marked_words` function.

## Files

```
results/emnlp/marked_persons/
├── summary.md                # this file
├── comparison_to_cheng.csv   # 9-row per-word comparison
├── z_scores.json             # our raw ensemble z-scores + Cheng targets
└── chatgpt_main.csv          # cached copy of Cheng's dataset
```

Reproduce with:
```bash
python scripts/experiments/repro_marked_persons.py
```

Runtime: ~5 s (data download the first time is ~1 MB; scoring is
Python-native, no GPU needed).

## The paper-facing claim

> *`bias_scope.generated_text_based.MarkedPersons` reproduces Cheng et al.
> 2023's marker-strength aggregate for the Asian-F ensemble — **mean
> z-score = 8.04** across the 9 significantly-marked words — at **0.0007 %
> relative error**, with Pearson correlation of **1.0000** across the
> individual word-level z-scores. The library needed one optional parameter
> added (`background_generations`) to enable Cheng's external-prior
> formulation of Monroe et al. 2008 Fightin' Words; both the default
> (Monroe) and the new (Cheng) paths are covered by regression tests, and
> all pre-existing tests continue to pass.*

This is the **first** generated-text-based metric in `bias_scope` shown to
reproduce a published paper within the 10 % band. It differs from the
other generated-text metrics (HONEST, RegardScore, ToxicityFraction, …)
because it is a **pure formula on a deterministic input** with no LM
sampling or classifier calibration in the chain.
