# CoOccurrenceBiasScore

**Cited source:** Bordia & Bowman 2019, *Identifying and Reducing Gender Bias in
Word-Level Language Models*, NAACL SRW 2019.
[arXiv:1904.03035](https://arxiv.org/abs/1904.03035).
**Reference implementation:**
[BordiaS/language-model-bias](https://github.com/BordiaS/language-model-bias) @
`59f6584e8a85` — **no license file** (RL-015).
**Sections and files read:** §3.3.1 "Bias Score Definition" (the definition of
`P(w|g)` and `bias_train(w)`); §"Fixed Context" (window sizes 5-15, k words each
side); repo layout.
**Family / access:** generated_text / `completions`

## Definition in the paper

```
P(w | g) = [ c(w, g) / Σᵢ c(wᵢ, g) ] / [ c(g) / Σᵢ c(wᵢ) ]

bias_train(w) = log( P(w | f) / P(w | m) )
```

`c(w, g)` counts occurrences of `w` in a context window around a `g`-gendered
word; `w` ranges over corpus words excluding stop words and gendered words. A
**positive score means the word co-occurs more with female words**. The paper is
explicit about the intended calibration:

> For an infinite context, the words doctor and nurse would cooccur as many
> times with a female gender as with male gender words and the bias scores for
> these words will be equal to zero.

Context is a fixed window of `k` words either side, with `k ∈ [5, 15]`.

## Current BiasScope implementation

`cooccurrence_bias_score.py:98-102`:

```
score(w; g1, g2) = log((c[w,g1] + s) / (C[g1] + s))
                 − log((c[w,g2] + s) / (C[g2] + s))
```

where `C[g]` is documented as "total **anchor** occurrences for group g" — that
is `c(g)`, the number of gendered-word occurrences.

## Verdict

**adaptation.**

Expanding both:

```
paper = [log c(w,f) − log c(w,m)]
      − [log Σᵢc(wᵢ,f) − log Σᵢc(wᵢ,m)]
      − [log c(f) − log c(m)]

ours  = [log c(w,f) − log c(w,m)]
      − [log c(f) − log c(m)]
```

**Ours omits the `Σᵢ c(wᵢ, g)` normalisation** — the total number of words
appearing in each group's context windows. That term is the same for every word
`w`, so it is a **constant additive offset**: the *ranking* of words by bias is
unaffected, and any comparison between two words is unaffected.

But the offset is not zero unless the two groups have equal total context mass,
and **0 is this metric's neutral value**. The paper's normalisation is exactly
what makes an equally-co-occurring word score 0; without it, such a word scores
`log(Σᵢc(wᵢ,m) / Σᵢc(wᵢ,f))`. So absolute values, and any statement of the form
"this word is unbiased", do not carry over.

Additive smoothing `s` is also ours, not the paper's.

## Required action

Track `Σᵢ c(wᵢ, g)` — the total context-window word count per group — and
include it, restoring the paper's calibration. It is a counter the windowing
loop already has the data for. Recorded as `REVIEW_LATER` RL-023.

Until then the deviation is stated in `MetricInfo.deviation_note`, so nobody can
read an absolute value as "unbiased" without seeing the caveat.

## Validation possible

- **Tier 1:** the paper reports bias scores over a Wikipedia corpus and for
  generated text, and mean absolute bias across window sizes. Reproducible in
  principle; needs the corpus.
- **Tier 2:** the repo states **no license**, so its code may be run for
  comparison but not vendored (RL-015).
- **Tier 3:** null and swap antisymmetry both apply — and swap antisymmetry is
  the sharp test here, since swapping `f` and `m` must negate every score. That
  property **does** hold for the current implementation, because the missing
  term is itself antisymmetric under the swap.

## Known limitations of the metric itself

- Window co-occurrence is a bag-of-words proxy: "the doctor told her" and "her
  doctor" count identically.
- Results move with window size, which the paper itself demonstrates by sweeping
  `k ∈ [5, 15]` — a Section 10.2 sensitivity target.
- Stop-word and gendered-word exclusion lists are not specified precisely enough
  to reproduce exactly.
