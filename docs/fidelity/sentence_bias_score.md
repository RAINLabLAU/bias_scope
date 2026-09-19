# SentenceBiasScore

**Cited source:** Dolci, T., Azzalini, F., & Tanelli, M. (2023). *Improving
Gender-Related Fairness in Sentence Encoders: A Semantics-Based Approach*.
Data Science and Engineering, 8, 177–195.
[doi:10.1007/s41019-023-00211-0](https://doi.org/10.1007/s41019-023-00211-0).
**Paper status: `read`, in full.** The article is Springer open access
(CC-BY-4.0); the PDF is at `biasscope papers/Sentencebiasscore.pdf`. It is
**not** paywalled — an earlier session (RL-029) could not locate it and left
the class `unaudited` on that assumption; that was wrong, and is corrected
here (see REVIEW_LATER.md).
**Reference implementation:** none. The paper's "Availability of data and
materials" section names only SNLI and SentEval; no code repository for bias
score itself is cited anywhere in the paper.
**Sections read:** §3 "Gender Bias Estimation" in full (3.1 Bias Score, 3.2
Gender Direction, 3.3 Gender Words, 3.4 Word Importance, 3.5 Examples), §6
"Extension to Transformer-Based Models".
**Family / access:** embedding / `embeddings`

## Definition in the paper

For a sentence `s`, with gender direction `g⃗`, gender-word lexicon `L`, and
per-word semantic importance `I_w`:

```
BiasScore_F(s)   = Σ_{w∈s, w∉L}  cos(vec_w, g⃗)|_{>0} × I_w
BiasScore_M(s)   = Σ_{w∈s, w∉L}  cos(vec_w, g⃗)|_{<0} × I_w
Abs-BiasScore(s) = Σ_{w∈s, w∉L}  | cos(vec_w, g⃗) × I_w |
```

`BiasScore_F` sums only positive-cosine terms (always ≥ 0); `BiasScore_M`
sums only negative-cosine terms (always ≤ 0); `Abs-BiasScore` (Eq. 3) is the
paper's own single-value summary, explicitly offered because it is "useful
... when sorting multiple sentences according to the total amount of
associated bias score." Words in `L` always contribute 0.

Three further procedures make the above computable, and are each their own
paper subsection:

- **§3.2 Gender direction `g⃗`**: PCA of the difference vectors of 10 (or up
  to 50) gender word pairs (`woman–man`, `girl–boy`, `she–he`, ...), oriented
  so female words get positive cosine.
- **§3.3 Gender words `L`**: 409 + 388 common nouns "selected starting from"
  Bolukbasi et al. 2016 and Zhao et al. 2018, in lower/capitalised ×
  singular/plural forms, plus 5765 U.S. Social Security given names — 6562
  words total. This exact merged list is not published.
- **§3.4 Word importance `I_w`**: percentage of the sentence encoder's
  max-pooled output dimensions for which the max-pooling operation selected
  that word's hidden state. Percentages are computed over *all* tokens
  (including gendered words and punctuation) and are not renormalised after
  `L`-exclusion (confirmed against the paper's own Table 2 worked example
  below, whose listed importances do not sum to 100% once the excluded
  gender word and an unlisted punctuation token are accounted for).

## Current BiasScope implementation

`src/bias_scope/embeddings_based/sentence_bias_score.py`.

**Eq. 1–3 (the scoring equations): faithful, verified.** `evaluate()`
computes exactly the formulas above. Feeding it vectors constructed to
reproduce the paper's own Table 2 example ("She likes the new pink dress")
gives `female_bias = 0.07550`, `male_bias = -0.01858` against the paper's
`0.07550` / `-0.01858` — agreement to the paper's own rounding
(`tests/test_embeddings/test_sentence_bias_score.py` pins the same equations
on synthetic known-answer cases).

**§3.2 and §3.4: now implemented.** `derive_gender_direction()` and
`derive_word_importance()` implement the paper's PCA and max-pooling
procedures respectively (added in the 2026-09 audit fix; previously absent
entirely — the class only executed the trivial final sum and required the
caller to have already derived `g⃗` and `I_w` by some unstated means).

- `derive_gender_direction`: PCA here means the top singular vector of the
  **uncentred** difference-vector matrix. Centring first (as
  `sklearn.decomposition.PCA` would, and as the paper's Fig. 2 scree-plot
  presentation might suggest) removes exactly the shared direction being
  sought, since gender-pair differences are expected to point roughly the
  same way; the paper publishes no code and its figure does not disambiguate
  centred from uncentred, so this is recorded as a `decide`, not a verified
  fact (REVIEW_LATER RL‑040).
- `derive_word_importance`: exact reproduction of the max-pooling-selection
  count, verified against the paper's own Fig. 3 ratio (`saxophone`: 1106 of
  4096 dimensions ≈ 27%).

**§3.3: still not implemented.** `build_gender_words_mask()` implements the
case-insensitive matching logic, but **the 6562-word lexicon itself is not
vendored**. It is not published by the paper, and reconstructing it from the
two cited source lists (Bolukbasi et al. 2016, Zhao et al. 2018) plus SSA
given-name data would not reproduce the authors' own curation ("selected
starting from" implies manual filtering the paper does not specify) — doing
so would replace one honest gap with a plausible-looking but silently wrong
lexicon. Callers must supply `gender_words_mask` (optionally built with
`build_gender_words_mask()` given their own lexicon). Tracked as
`REVIEW_LATER` RL‑041, tag `blocked`.

**`run()` fixed (was completely broken).** `evaluate(..., return_details=True)`
now includes a `bias_score` key (= `absolute_bias`, Eq. 3) so
`BiasMetric._split_result` can find a headline score, a `breakdown` dict
(`female_bias`, `male_bias`) and an integer `n` (= number of words scored).
Before this fix, `SentenceBiasScore().run(...)` raised `BiasScopeError`
unconditionally on every input — the framework's second public entry point
(PLAN.md §5.3) did not work at all for this metric.

**Citation was wrong.** The class docstring and `docs/api/embeddings/
sentence_bias_score.md` previously cited *"Dolci, M., Azzalini, D., & Tanelli,
M. (2023). Sentence-level bias detection in transformer models"* — wrong
author initials (T., F., M., not M., D., M.) and a title that does not match
the paper (`_metric_info.py`'s `reference` field had the venue right but the
docstring/docs did not). Both corrected.

## Verdict

**adaptation.** The scoring equations are faithful and verified; two of the
three upstream derivations (gender direction, word importance) are now
implemented and verified against the paper's own examples. The third
(the gender-word lexicon) is a genuine, disclosed, `blocked` gap — not a
silent deviation — so `fidelity="faithful"` would overstate what a caller
gets without supplying their own lexicon.

## Required action

None blocking. Two open `REVIEW_LATER` items: RL‑040 (`decide`, the
uncentred-PCA choice — revisit if reference code or an erratum ever
surfaces), RL‑041 (`blocked`, vendor a licensed gender-word lexicon with a
SHA-256, e.g. from Bolukbasi et al.'s `debiaswe` repo and Zhao et al.'s
released list, once their licenses are checked).

## Validation possible

- **Tier 1:** the paper's own worked examples (Table 2, Table 3, Table 13,
  the SNLI/STS-Benchmark/MultiNLI most-biased-sentence lists) are directly
  reproducible once a caller supplies InferSent/SBERT hidden states and the
  gender lexicon; the scoring-equation portion is already reproduced above.
- **Tier 2:** no reference implementation exists to compare against.
- **Tier 3:** null (identical embeddings for female/male-labelled words with
  the same cosine give equal-magnitude opposite bias — not directly
  applicable, this metric has no target/attribute group structure like
  WEAT), scale invariance (cosine similarity, holds), swap antisymmetry
  (swapping the sign convention of `g⃗` negates both outputs — holds by
  construction).

## Known limitations of the metric itself

- Depends entirely on the quality and completeness of the gender-word
  lexicon `L`; an incomplete lexicon systematically overcounts "bias" for
  words that should have been masked to zero.
- The gender direction is a single PCA axis from 10–50 word pairs; the paper
  itself cites reliability concerns about single-axis gender directions
  (Du, Fang & Nguyen 2021, cited in the paper as [21]).
- Word importance is tied to a specific max-pooling sentence encoder
  (InferSent in the main text, SBERT/BERT in §6); it has no defined meaning
  for encoders that don't use max-pooling (e.g. mean-pooling or `[CLS]`).
