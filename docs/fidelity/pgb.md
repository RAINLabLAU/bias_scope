# PGB — Partial Gen Bias

**Cited source:** Smith et al. 2022, EMNLP,
[arXiv:2205.09209](https://arxiv.org/abs/2205.09209), §A.7. Read as
`sources/papers/fgb.pdf` (same PDF as [FGB](fgb.md)).
**Reference implementation:**
[facebookresearch/ResponsibleNLP](https://github.com/facebookresearch/ResponsibleNLP)
@ `0ec714eb084217f44cd9ac466d9e988c795302f9`, MIT — **does not implement this
metric**; see [fgb.md](fgb.md) for the search.
**Sections and files read:** §A.7 (PGB and SCGB definitions, the six style
clusters and how they were derived by agglomerative clustering);
`holistic_bias/src/bias_measurements.py:45-140`.
**Family / access:** generated_text / `completions`

## Definition in the paper

PGB is FGB restricted to one style cluster `C`:

```
PGB(C) = (1/T) · Σ_{t=1..T} Σ_{s∈C} Var_d ( (1/N_td) · Σ_{i=1..N_td} p_tdis )
```

The six clusters reported in Table 4 are SYMPATHY, ENVY, CURIOSITY, CONFUSION,
HATE, and CARE, obtained by agglomerative hierarchical clustering over the 217
style probability vectors and then ranked by PGB.

The paper immediately flags a defect in its own metric:

> "one issue with it is that it artificially deflates the bias in style clusters
> with many styles"

and therefore defines **SCGB**, which sums within the cluster before taking the
variance. A faithful implementation should offer both.

> **Correction to PLAN.md 4.2.** The row says PGB is "uncited in the paper".
> That is wrong — PGB is defined in §A.7 alongside FGB and reported in Table 4
> and Figure 4. Corrected in the same commit.

## Current BiasScope implementation (v0.1.1)

`src/bias_scope/generated_text_based/pgb.py:22`:

```
PGB = mean_{t,k} max(0, delta(t, k))     where delta = s_a − s_b
```

A one-sided mean gap: how much group A's scalar score exceeds group B's, on
average, counting only the positive direction.

## Verdict

**mismatch.**

The two share no structure. The paper's PGB is a per-style variance across many
descriptors summed over a style cluster; ours is a rectified mean difference
between two groups of scalars. Ours has no notion of styles, clusters, or
descriptors, and is asymmetric by construction (swapping A and B gives a
different number), whereas PGB is symmetric under descriptor relabelling.

## Required action

1. **Reimplement `PGB`** alongside FGB, sharing the style-vector machinery, and
   parameterised by a cluster. Ship the paper's six clusters as data with
   attribution. Add `SCGB` too, since the paper says PGB alone is misleading.
2. **Move the current behaviour** to `PairGapPositive`, status `original`, with
   a deprecating alias (PLAN.md Section 12).
3. Blocked on the same missing style classifier as FGB — `REVIEW_LATER` RL-013.

## Validation possible

- **Tier 1:** Table 4 gives PGB per cluster per model (×1000); same classifier
  dependency as FGB.
- **Tier 2:** not possible against this repo.
- **Tier 3:** as FGB — null and permutation invariance apply; swap antisymmetry
  is exempt with the same reason (a variance is symmetric under relabelling).

## Known limitations of the metric itself

- The paper's own stated defect: clusters with many low-probability styles get
  systematically smaller PGB, which is what SCGB exists to fix. Reporting PGB
  without SCGB reproduces a flaw the authors documented.
- The six clusters are derived from one model's outputs (BlenderBot 2.0 3B
  without bias-reduction tuning), so they are not model-neutral.
