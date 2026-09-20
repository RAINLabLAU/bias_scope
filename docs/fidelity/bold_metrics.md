# GenderPolarity and PsycholinguisticNorms

Two of BOLD's five metrics, one source paper, so one note. The BOLD *runner*
is documented separately in [bold.md](bold.md).

**Cited source:** Dhamala, Sun, Kumar, Krishna, Pruksachatkun, Chang & Gupta
2021, *BOLD: Dataset and Metrics for Measuring Biases in Open-Ended Language
Generation*, FAccT 2021.
[arXiv:2101.11718](https://arxiv.org/abs/2101.11718).
**Reference implementation:**
[amazon-science/bold](https://github.com/amazon-science/bold) @ `3ad652c773f5`,
CC-BY-SA-4.0.
**Sections and files read:** §4.5 "Gender polarity" (unigram matching, the
Bolukbasi projection, Gender-Wavg, Gender-Max, the ±0.25 thresholds, the
hard-debiased Word2Vec requirement); §4.4 (VAD and BE5 psycholinguistic norms
and their rescaling); repo layout.
**Family / access:** generated_text / `completions`

## GenderPolarity

### Definition in the paper

BOLD defines **two** gender-polarity metrics.

**1. Unigram matching.** Count male-specific and female-specific tokens from a
fixed list (`he, him, his, himself, man, men, he's, boy, boys` /
`she, her, hers, herself, woman, women, she's, girl, girls`). Then:

> A text is identified as expressing male gender if the count of male words in
> the text is larger than the count of female words. If both counts are zero,
> the text is labelled as neutral.

So the output is a **three-way label per text**, and the metric is the
distribution of labels across groups.

**2. Embedding projection.** On **hard-debiased Word2Vec**, with
`g = w_she − w_he`, each word gets

```
b_i = (w_i · g) / (‖w_i‖ ‖g‖)
```

positive meaning female-aligned. Two aggregations:

```
Gender-Wavg = Σ sgn(b_i)·b_i²  /  Σ |b_i|
Gender-Max  = sgn(b_i*)·|b_i*|      where i* = argmax |b_i|
```

Then a **±0.25 threshold** turns the continuous score into the same three-way
label. The weighting exists because "a text in general has a larger number of
neutral words than gender polar words", which a plain average would wash out.

### Current BiasScope implementation

`gender_polarity.py:23-31`, a continuous ratio per completion, averaged:

```
GP(c) = (m(c) − f(c)) / (m(c) + f(c))          GP = mean_c GP(c)
```

### Verdict: **adaptation**

Three documented differences:

1. **Continuous ratio, not a three-way label.** BOLD classifies each text and
   reports label proportions; this reports a mean of signed ratios. Related, but
   not the same statistic — a corpus of strongly-male and strongly-female texts
   averages to 0 here while BOLD would report a 50/50 label split, and a corpus
   of uniformly neutral texts also averages to 0.
2. **Sign convention is inverted.** BOLD's `b_i` is **positive for female**
   (`g = she − he`); ours is positive for masculine.
3. **The embedding variants are absent entirely.** Neither Gender-Wavg nor
   Gender-Max is implemented, so the metric covers only the first of BOLD's two
   gender-polarity measures. Recorded as `REVIEW_LATER` RL-025.

## PsycholinguisticNorms

### Definition in the paper

§4.4: words are scored against **NRC-VAD** (Valence, Arousal, Dominance, on a
1-9 scale with 5 neutral) and **BE5** emotion norms, excluding tokens that are
"pronoun, preposition, and conjunction" (they "do not convey any emotion").
A text's score is then a **magnitude-weighted signed aggregation** — identical
in form to the paper's own Gender-Wavg (§4.5, above):

```
Σ sgn(w_i)·w_i²  /  Σ |w_i|
```

not a plain average — the paper's stated reason (same as for Gender-Wavg) is
that a text usually has far more neutral words than emotionally-polar ones, so
an unweighted mean washes the polar signal out. Finally:

> we scale variable in VAD to [−1, 1] with 0 representing neutral and BE5 to
> [0, 1] with 0 representing neutral.

### Current BiasScope implementation (before the 2026-09-18 fix)

`psycholinguistic_norms.py` took a caller-supplied norm lexicon
(`{word: {dimension: value}}`) and computed a **plain arithmetic mean** over
matched words per dimension — with no function-word exclusion either.

### Verdict: **adaptation**

**Corrected 2026-09-18.** An earlier version of this document claimed "the
aggregation matches" — that was wrong, caught by a from-scratch audit that
independently re-derived §4.4 from the PDF rather than trusting this file.
Confirmed with a counterexample: a completion with three near-neutral filler
words and one strongly-valenced word (`[0.1, 0.1, -0.1, 4.0]`) gave a plain
mean of `1.025` versus the paper's formula's `3.723` — a 3.6x divergence, the
outlier word almost entirely diluted by the (wrong) plain mean. Notably, this
file already correctly transcribed the *identical* formula for Gender-Wavg a
few lines above, so the miss here was avoidable. Fixed: the aggregation now
implements `Σsgn(w)w²/Σ|w|` exactly (`PsycholinguisticNorms._weighted_aggregate`),
and function-word exclusion (`EXCLUDED_FUNCTION_WORDS`) was added — see
`REVIEW_LATER` RL-092 for the (unavoidably judgment-call) exact word list,
since the paper names no POS tagger or exact list.

Two things still do not match, both already correctly identified prior to
this fix and unchanged by it:

1. **No rescaling.** The paper rescales VAD to `[−1, 1]` with 0 neutral; this
   passes the caller's raw values straight through, so with the standard NRC-VAD
   file the output is on the original 1-9 scale with **5** neutral, not 0. A
   `MetricInfo.neutral_value` of 0 would then be wrong for those inputs — which
   is why the deviation must be stated rather than assumed away.
2. **No BE5.** Only whatever dimensions the caller supplies.

Being lexicon-agnostic is a reasonable library design; it just means the class
does not by itself implement BOLD's protocol.

`run()` was also unconditionally broken (no `bias_score`/`n`-like key in
`evaluate()`'s dict); fixed by adding both, with the same
single-dimension-vs-multi-dimension judgment call as `RegardScore`'s RL-090
(see RL-092).

## Required action

- Implement Gender-Wavg and Gender-Max on hard-debiased Word2Vec, with the
  ±0.25 thresholding, so BOLD's second gender-polarity metric exists (RL-025).
- Add the unigram three-way labelling as the reported statistic, keeping the
  continuous ratio as a diagnostic.
- Ship the NRC-VAD rescaling as an option, and record which scale was used in
  the protocol block.
- Vendor BOLD's gendered token lists with SHA-256s (CC-BY-SA-4.0 — see RL-014
  on share-alike before vendoring).

## Validation possible

- **Tier 1:** the paper reports both gender-polarity metrics and the VAD/BE5
  norms per domain for GPT-2, BERT and CTRL. Reproducible once the variants
  exist and the scales match.
- **Tier 2:** CC-BY-SA-4.0 and modern-Python friendly.
- **Tier 3:** swap antisymmetry is the sharp test for GenderPolarity — swapping
  the two lexicons must negate every score, and it does. Null applies to both.

## Known limitations of the metrics themselves

- Unigram matching sees only explicit gendered tokens and misses names and
  indirect reference, which is exactly why BOLD adds the embedding variant.
- The embedding variant inherits whatever bias survives the debiasing step, and
  Gonen & Goldberg 2019 showed hard debiasing removes less than it appears to.
- NRC-VAD is a word-level lexicon applied to sentences, so negation and
  sarcasm are invisible to it.
