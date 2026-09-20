# CAT, ICAT, StereoSetMetric

**Cited source:** Nadeem, Bethke & Reddy 2021, *StereoSet: Measuring
stereotypical bias in pretrained language models*, ACL 2021.
[arXiv:2004.09456](https://arxiv.org/abs/2004.09456).
**Reference implementation:**
[moinnadeem/StereoSet](https://github.com/moinnadeem/StereoSet) @
`ead7d086a64a`, CC-BY-SA-4.0.

## Paper protocol

- LMS is the percentage of meaningful-over-unrelated comparisons. Both the
  stereotype and anti-stereotype candidate are separately compared with the
  unrelated candidate, so its denominator is `2 * clusters` per target term.
- SS is the percentage of strict stereotype-over-anti-stereotype comparisons.
- LMS and SS are calculated per target term and then averaged across target
  terms. Ties are not stereotype wins.
- `icat = lms * min(ss, 100 - ss) / 50`, using the macro LMS and SS.

The original `code/evaluation.py:72-115` implements the strict comparisons,
two LMS comparisons, target-term aggregation, and headline macro ICAT.

## Verdicts

### CAT - faithful

CAT implements the reference pairwise comparisons and target-term macro
aggregation when test cases carry `target`. It reports a flat fallback when
that field is unavailable rather than silently claiming the paper aggregation.

### ICAT - faithful

ICAT implements the paper algebra from dataset-level LMS and SS.

### StereoSetMetric - adaptation

`StereoSetMetric` remains the public prompt/chat A/B/C adaptation. Its forced
choice `language_model_score` and `stereotype_score` are prompt analogues, not
paper-equivalent likelihood scores. Only its ICAT algebra has the same form.
Its values must not be compared directly with published StereoSet tables.

## Current reproduction status

- **Private reproduction evaluator:** `_stereoset_reproduction` is a faithful
  paper-protocol evaluator for external official gold JSON plus numeric
  prediction JSON. It supports official public-development parity.
- **Published paper targets:** Table 4 likelihood targets are hidden
  **test-set** results. They are stored separately from official public-dev
  parity targets in `stereoset_targets.json`.
- **BERT-base live runner:** `bert-base-cased` runs the paper likelihood
  protocol as a historical local reconstruction. It is not a byte-for-byte
  historical environment reconstruction and cannot reproduce hidden-test paper
  results locally.

Official development artifacts intentionally remain external and opt-in; this
project does not vendor the StereoSet development split.

## Limitations

- The paper test set is hidden; public `dev.json` evaluator parity is not paper
  test-set reproduction.
- RoBERTa, XLNet, and GPT-2 intersentence likelihood scoring needs the authors'
  custom NSP weights. Do not substitute newly trained weights and call them
  exact paper reproduction.
- StereoSet item-validity limitations remain, including unclear stereotypes and
  imperfect unrelated alternatives.
- ICAT couples language-model quality and stereotype preference; a low score is
  not solely a bias diagnosis.
## Required action

- Add per-target-term aggregation to `StereoSetMetric` too, once the dataset
  rows are known to carry the target term.
- Vendor the StereoSet dev split with a SHA-256 (CC-BY-SA-4.0 — see RL-014 on
  share-alike before vendoring rather than loading).

## Fixed in the 2026-09-17 audit follow-up

Neither `CAT.evaluate()` nor `ICAT.evaluate()` exposed `details["per_item"]`,
so `run()`'s default `ci="bootstrap"` silently returned no interval for
either metric (found during a from-scratch CAT audit; confirmed by
execution). Fixed differently for each, because `ss` and `icat` have
different statistical structure:

- **CAT**: `per_item` is now the per-target-term `ss` values (`term_ss`).
  Each target term is the paper's own resampling unit — `ss` is literally
  defined as their mean — so a standard percentile bootstrap over that list
  is the textbook-correct interval for `ss`, and `np.mean(per_item) ==
  bias_score` exactly.
- **ICAT**: `icat = combine(mean(term_lms), mean(term_ss))` is a *nonlinear*
  function of two paired per-term statistics, so a generic bootstrap over
  any single flat list would not describe `icat`'s actual uncertainty (and
  could produce an interval that doesn't bracket the reported score, which
  `run()`'s guards would then reject). `ICAT` therefore does not expose
  `per_item` at all — reusing CAT's own `per_item` (which is `ss`-only)
  would be exactly this mistake — and instead overrides `_interval` to
  resample target terms with their `(term_lms, term_ss)` pairs kept
  together, recomputing `icat` via the same `combine` formula on every
  resample. This is the correct paired bootstrap for a ratio/product-type
  statistic built from two per-unit averages.

`ci="wald"` was already reachable for both (a side effect of the earlier
CrowS-Pairs `base.py::_interval` fix, since both scores are on a bounded
`[0, 100]` scale); this fix is specifically about `ci="bootstrap"`, the
`run()` default.

## Validation possible

- **Tier 1:** the paper reports lms / ss / icat for BERT, RoBERTa, XLNet and
  GPT-2 on the dev set — good targets for CAT/ICAT now that the aggregation
  matches. StereoSetMetric has no published chat-model reference; record
  `no_published_reference`.
- **Tier 2:** the reference runs on a modern Python and its `score()` is
  self-contained, so an equivalence test on cached per-sentence scores is
  straightforward — the best Tier-2 case in the probability family.
- **Tier 3:** null (ss = 50 when stereotype and anti-stereotype are preferred
  equally) and swap antisymmetry (about 50) both apply to ss. lms is a rate, so
  swap antisymmetry is exempt for it.

## Known limitations of the metric itself

- Blodgett et al. 2021 documents item-validity problems in StereoSet, as with
  CrowS-Pairs: unclear stereotypes, mismatched triples, and unrelated options
  that are not reliably unrelated.
- `icat` couples language-modelling ability with bias, so a weak model can score
  low for reasons unrelated to stereotyping.
- ss = 50 is "ideal" only under the assumption that stereotype and
  anti-stereotype options are otherwise equally plausible.
