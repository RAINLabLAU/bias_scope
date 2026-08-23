# SentenceBiasScore

**Cited source:** Dolci, Azzalini & Tanelli 2023, *Sentence-level bias score*,
Data Science and Engineering 8(2), Springer.
**Paper status: `not_located`.** Springer paywalled; no arXiv or other preprint
was found after the Section 4.0 search. `sources/SOURCES.yaml` records
`paper_status: not_located` rather than a guessed identifier — PLAN.md Appendix E
listed this row as "DOI via Springer; check for preprint", and there is none.
**Reference implementation:** none located — `code_status: none_found` with the
search log.
**Family / access:** embedding / `embeddings`

## Status: **unaudited**, and it will stay that way until the paper is obtained

This is the one metric in the library whose source could not be read at all.
Section 4.0 is unambiguous: no metric may be audited from memory, and a fidelity
status may not be assigned without reading the paper. So `SentenceBiasScore`
keeps `fidelity: unaudited`, and `MetricInfo.deviation_note` says why.

## What the implementation does

`sentence_bias_score.py` computes a gender direction from paired gendered words,
projects each token's embedding onto it, and combines the per-word scores with
importance weights, masking out explicitly gendered words. That is a coherent
and recognisable construction — it is close in spirit to Bolukbasi et al.'s
direct bias — but whether it matches Dolci et al.'s definition is exactly the
question that cannot be answered without the paper.

## Required action

**Obtain the paper.** Options, in order of preference:

1. Institutional access to Data Science and Engineering 8(2).
2. Email the corresponding author for an accepted manuscript — the usual and
   usually successful route.
3. If neither works, the honest outcome is to **relabel the class as `original`
   and stop citing Dolci et al. as its source**, because an unverifiable
   citation is worse than none. That is a `decide` for the maintainer, not for
   the audit.

Recorded as `REVIEW_LATER` RL-029.

## Validation possible

- **Tier 1:** unknown — the paper's reported values cannot be read.
- **Tier 2:** not possible; no code release located.
- **Tier 3: the only route currently available.** The five metamorphic
  properties can be checked without knowing the paper: null, swap antisymmetry
  (swapping the gender word lists should negate the score), permutation
  invariance, and scale invariance (an embedding metric's score should not
  change if all vectors are scaled). These should be run now, since they are
  objective evidence that the implementation is at least internally coherent,
  which is more than the library currently has for this metric.

## Note for the paper

This is worth one sentence in the write-up. "One of 43 metrics cites a source we
could not obtain, and we label it `unaudited` rather than assert a fidelity" is
a stronger statement about the library's integrity than silently claiming 43 of
43 were verified.
