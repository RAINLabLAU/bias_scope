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
