# DemographicRepresentationBias → OccupationPronounSkew

**Cited source:** Zhao, Wang, Yatskar, Ordonez & Chang 2018, *Gender Bias in
Coreference Resolution: Evaluation and Debiasing Methods* (WinoBias), NAACL 2018.
[arXiv:1804.06876](https://arxiv.org/abs/1804.06876), read as
`sources/papers/demographicrepresentationbias.pdf`.
**Reference implementation:**
[uclanlp/corefBias](https://github.com/uclanlp/corefBias) @
`0bce984dd081bbc10b0622f326727a024c607895`, MIT.
**Sections and files read:** abstract and §1 (the metric: "an average difference
of 21.1 in F1 score" between pro- and anti-stereotypical conditions, `:17-19`,
`:91-93`); §"pro-stereotypical / anti-stereotypical" definitions (`:162-165`);
`WinoBias/wino/` layout (`anonymize.py`, `word_swapper.py`, `data/`).
**Family / access:** prompt / `completions`

## Definition in the paper

WinoBias is a **coreference resolution** benchmark. Each sentence has an
occupation, a participant, and a pronoun. In the **pro-stereotypical**
condition the pronoun's gender matches the occupation's stereotypical gender;
in the **anti-stereotypical** condition it does not. The two sets are otherwise
identical.

The metric is the **difference in coreference F1 between the two conditions**.
Zhao et al. report an average gap of **21.1 F1**. Zero means the system resolves
both conditions equally well.

Two sentence types (Type 1 requires world knowledge, Type 2 is syntactically
resolvable) are reported separately.

## Current BiasScope implementation (v0.1.1)

`src/bias_scope/prompts_based/demographic_representation_bias.py`. It generates
completions for occupation templates and **counts he/she/they pronouns**
(`:20-21`), reporting `representation_ratio` = male/female pronoun count
(`:90-91`), an L1 distance from a uniform pronoun distribution, and a
per-occupation breakdown.

## Verdict

**original**, as of v0.2.0 — renamed to `OccupationPronounSkew`.

**History: v0.1.1 was a `mismatch`.** It carried the name
`DemographicRepresentationBias` and cited Zhao et al. 2018, while computing
something else entirely. The audit below is of that version.

The paper measures **accuracy on a coreference task**; the implementation
measures **which pronouns a model chooses to emit**. There is no coreference
task, no pro/anti condition pairing, no gold labels, and no F1 anywhere in the
implementation. A model could produce a perfectly balanced pronoun distribution
and still resolve anti-stereotypical coreference far worse than pro-, or the
reverse — the two quantities are not related by any transformation.

Occupation-pronoun skew is a legitimate thing to measure, and it is what the
code measures well. The defect is the name and the citation.

## Resolution (v0.2.0)

1. Renamed to `OccupationPronounSkew`, `fidelity: original`. The statistic is
   **unchanged** — this is a pure rename, so `DemographicRepresentationBias`
   stays importable until 0.3.0 and raises a `DeprecationWarning` naming the new
   class (`bias_scope/_deprecation.py`). The alias is a subclass, so existing
   `isinstance` checks keep working.
2. The Zhao et al. citation is now recorded as inspiration, not as
   "implements", in `MetricInfo.reference` and `deviation_note`.

Still open: the faithful `WinoBias` (Phase 4, PLAN.md 7.2). Until it exists, the
library has **no** implementation of Zhao et al.'s metric — the rename removed a
false claim, it did not add the real thing.

## Required action (item 1 is now done)

1. ~~**Rename to `OccupationPronounSkew`**, status `original`, with a deprecating
   alias per PLAN.md Section 1.~~ Done in 0.2.0.
2. **Add a real `WinoBias`** (PLAN.md 7.2): pro/anti F1 gap via a prompt-based
   coreference question for chat models plus a logit version for encoders, with
   Tier 2 against `uclanlp/corefBias` scoring.
3. The reference repo ships the data and the swap lists under MIT, so both the
   dataset and the scorer are usable — this is one of the better-supported new
   metrics in Phase 4.

## Validation possible

- **Tier 1 (for the real WinoBias):** the 21.1 F1 average gap, and the
  per-system Type 1 / Type 2 tables. Note these are for 2018-era coreference
  systems, not LLMs, so a modern reproduction is a different protocol and must
  be recorded as such rather than compared to 21.1 directly.
- **Tier 1 (for `OccupationPronounSkew`):** none — it is `original`. Record
  `no_published_reference`.
- **Tier 2:** possible for the real WinoBias against `corefBias` (MIT).
- **Tier 3:** for the skew metric, null and swap antisymmetry apply cleanly
  (swapping the male and female word lists should invert the ratio).

## Known limitations of the metric itself

- Pronoun counting is binary-plus-they and cannot represent non-binary identity
  beyond a single bucket.
- The occupation list drives the result; WinoBias's own list comes from US
  Department of Labor statistics and is dated.
