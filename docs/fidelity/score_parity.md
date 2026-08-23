# ScoreParity → MeanScoreGap

**Cited source:** Borkan, Dixon, Sorensen, Thain & Vasserman 2019, *Nuanced
Metrics for Measuring Unintended Bias with Real Data for Text Classification*,
WWW 2019 companion. [arXiv:1903.04561](https://arxiv.org/abs/1903.04561).
**Reference implementation:**
[conversationai/unintended-ml-bias-analysis](https://github.com/conversationai/unintended-ml-bias-analysis)
@ `1244018d92be`, Apache-2.0.
**Sections and files read:** §3 Definition 1 (Subgroup AUC, BPSN AUC, BNSP AUC,
eqs. 1-3) and Definition 2 (Positive / Negative Average Equality Gap, eq. 4);
§3.2.1 on the relationship to Mann-Whitney U.
**Family / access:** generated_text / `completions`

## Definition in the paper

Borkan et al. propose "a suite of **five** metrics, derived from ROC-AUC,
Equality Gap, and Mann-Whitney U metrics", all **threshold agnostic**. With
`D−`/`D+` the negative/positive background examples and `D−g`/`D+g` the
negative/positive examples in the identity subgroup:

```
Subgroup AUC = AUC(D−g + D+g)          (1)
BPSN AUC     = AUC(D+  + D−g)          (2)
BNSP AUC     = AUC(D−  + D+g)          (3)
Positive AEG = ∫₀¹ (y(t) − x(t)) dx(t) (4)   [and the negative analogue]
```

Every one of these requires **labels**: which examples are positive and which
are negative. The point of the suite, stated in §1, is that threshold-*dependent*
metrics "obscure the view of unintended bias and thus be misleading to
practitioners".

## Current BiasScope implementation

`mean_score_gap.py` (was `score_parity.py`). It takes two groups of texts, scores
them with a classifier, and reports:

```
difference = mean(scores_A) − mean(scores_B)
cohens_d   = (mean_A − mean_B) / pooled_std
```

No labels, no AUC, no threshold, no Mann-Whitney.

## Verdict

**original**, as of v0.2.0.

**History: v0.1.1 carried the name `ScoreParity` citing Borkan et al.** — an
overclaim. A mean-score gap is not one of Borkan's five metrics and is not
derivable from them: it is threshold-free but not threshold-*agnostic* in
Borkan's sense, and it cannot be computed at all in the setting Borkan cares
about, where the question is whether *mis-ordering* occurs between subgroup and
background examples of known polarity.

PLAN.md 4.2 anticipated this ("Borkan's metrics are AUC-based ... mean-score gap
is not one of them", preliminary: "likely original"). Confirmed.

## Resolution (v0.2.0)

Renamed to `MeanScoreGap`, `fidelity: original`, per PLAN.md 5.2 ("keep name
only if the AUC trio is implemented, else `MeanScoreGap`"). The statistic is
unchanged, so `ScoreParity` stays importable until 0.3.0 behind a
`DeprecationWarning`.

**The library still has no implementation of Borkan's metrics.** The rename
removed a false claim; it did not add the real thing.

## Required action

Implement the AUC trio plus the two AEGs as a separate metric, once a labelled
dataset is wired in. The reference repo is Apache-2.0 and its notebooks compute
all five, so Tier 2 is straightforward. This is a Phase 4 addition, not a fix.

## Validation possible

- **Tier 1 (for `MeanScoreGap`):** none — it is `original`. Record
  `no_published_reference`.
- **Tier 1 (for the real metrics):** Borkan's Table 2 reports the five metrics
  on the Civil Comments dataset per identity term.
- **Tier 2:** good — Apache-2.0, and the AUC computations are self-contained.
- **Tier 3:** null and swap antisymmetry apply cleanly to the gap (swapping the
  two groups negates it).

## Known limitations of the metric itself

- A mean gap is invisible to distribution shape: two groups with identical means
  but very different score spreads score 0.
- It inherits the classifier's own biases entirely.
