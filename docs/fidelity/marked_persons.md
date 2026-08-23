# MarkedPersons

**Cited source:** Cheng, Durmus & Jurafsky 2023, *Marked Personas: Using Natural
Language Prompts to Measure Stereotypes in Language Models*, ACL 2023.
[arXiv:2305.18189](https://arxiv.org/abs/2305.18189).
**Reference implementation:**
[myracheng/markedpersonas](https://github.com/myracheng/markedpersonas) @
`9b3ae82ad262` — **no license file** (RL-015).
**Sections and files read:** §"We use the Fightin' Words method of Monroe et al.
(2008) with the informative Dirichlet prior, first computing the weighted
log-odds ratios ... and using the z-score to measure the statistical
significance"; the `z > 1.96` significance criterion; the per-group word tables;
repo layout.
**Family / access:** generated_text / `completions`

## Definition in the paper

Cheng et al. apply **Fightin' Words** (Monroe, Colaresi & Quinn 2008) with an
**informative Dirichlet prior**: for each word, compute the weighted log-odds
ratio between the marked group's generations and the unmarked group's, with the
prior drawn from a background corpus, then convert to a z-score. Words with
**z > 1.96** are reported as significantly distinguishing the group.

The paper's output is a **ranked word list per group**, not a single scalar.

## Current BiasScope implementation

`marked_persons.py` with `_helpers.py::compute_log_odds_with_prior`, which
returns `(delta, variance, z)` — the Monroe et al. quantities directly.

`evaluate` returns the per-term z-scores, the top marked and unmarked terms, and
a summary of vocabulary and token counts. **It reports no single headline
scalar**, which matches the paper: Cheng et al. do not define one.

PLAN.md 4.2's action was "document what the reported scalar is". The answer is
that there isn't one, and that is correct. The library's own reproduction
constructs two aggregates for comparison purposes — the mean and sum of the
z-scores of the words Cheng reports — and those are reproduction artefacts, not
the metric.

## Verdict

**faithful.**

Backed by the strongest reproduction evidence in the library.
`results/emnlp/marked_persons/` reproduces Cheng's published example table:

| | Cheng 2023 | BiasScope | Relative error |
|---|---:|---:|---:|
| Mean z-score of the 9 marked words (Asian-F) | 8.0396 | 8.0396 | **0.0007%** |
| Sum of those z-scores | 72.3560 | 72.3565 | 0.0007% |
| Pearson correlation of per-word z-scores | — | **1.0000** | — |

Every one of the nine words matches to ≤ 0.008% relative error — floating-point
noise. A correlation of exactly 1.0000 across the reported words means the
ranking is identical, not merely similar.

## Required action

None outstanding. One note for Phase 3: because the metric's output is a word
list rather than a scalar, its `BiasResult.score` needs a deliberate choice.
Reporting the count of significant words (z > 1.96) is the most defensible
scalar since it is the paper's own significance criterion, and should be
recorded as a `decide` when `run()` is wired up for this metric.

## Validation possible

- **Tier 1: already achieved**, and it is the cleanest reproduction in the
  library. It should be the lead example in the paper's validation table —
  a 0.0007% match on a metric a reviewer can check by eye against the paper's
  own table.
- **Tier 2:** the repo states **no license** (RL-015), so its code may be run
  locally but not vendored. Given the Tier-1 agreement, Tier 2 adds little here.
- **Tier 3:** null (identical corpora → all z ≈ 0) and swap antisymmetry
  (swapping marked and unmarked negates every z) both apply cleanly.

## Known limitations of the metric itself

- Fightin' Words compares word *frequencies*; it says nothing about whether a
  distinguishing word is harmful, only that it is distinguishing.
- The z-scores depend on the background corpus supplying the prior, and on
  `prior_alpha`; both belong in the protocol block.
- `z > 1.96` is an uncorrected threshold applied across a whole vocabulary, so
  the significant-word list will contain false positives by construction — the
  paper uses it as a ranking device rather than a hypothesis test.
