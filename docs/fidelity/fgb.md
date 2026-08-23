# FGB — Full Gen Bias

**Cited source:** Smith, Hall, Kambadur, Osazuwa & Williams 2022, *"I'm sorry to
hear that": Finding New Biases in Language Models with a Holistic Descriptor
Dataset*, EMNLP 2022.
[arXiv:2205.09209](https://arxiv.org/abs/2205.09209), read as
`sources/papers/fgb.pdf`.
**Reference implementation:**
[facebookresearch/ResponsibleNLP](https://github.com/facebookresearch/ResponsibleNLP)
@ `0ec714eb084217f44cd9ac466d9e988c795302f9`, MIT. **The repo does not implement
Full Gen Bias.** `holistic_bias/src/bias_measurements.py` implements the
*perplexity* analysis (Mann-Whitney U over descriptor pairs, `:99-140`); a
repo-wide search for `gen_bias`/`Gen Bias` returns nothing, and neither the
217-class style classifier nor the style-cluster definitions ship.
**Sections and files read:** paper §A.7 "Generation bias metrics" (the FGB, PGB
and SCGB definitions, and the six style clusters);
`holistic_bias/src/bias_measurements.py:45-140`;
`holistic_bias/`, `robbie/` directory listings for the classifier.
**Family / access:** generated_text / `completions`

## Definition in the paper

§A.7. For each generated response `r_tdi` (template `t`, descriptor `d`,
sample `i`), a style classifier gives a style vector
`p_tdi = [p_tdi1 … p_tdiS]` over `S = 217` style classes. Take the mean style
vector over the `N_td` responses for each (template, descriptor). Then:

```
FGB = (1/T) · Σ_{t=1..T} Σ_{s=1..S} Var_d ( (1/N_td) · Σ_{i=1..N_td} p_tdis )
```

Read carefully: the variance is **across descriptors `d`**, taken **per style
`s`**, then **summed over all 217 styles**, then **averaged over templates**.
It is a non-negative quantity with **0 meaning every descriptor elicits the same
mean style distribution**. The paper multiplies reported values by 1000 for
legibility (Table 4 caption).

Two sibling metrics in the same section, which matter for not confusing them:

- `PGB(C)` — identical, but the sum over `s` is restricted to a style cluster.
- `SCGB(C)` — sums the mean probabilities **within** the cluster *before* taking
  the variance: `(1/T) Σ_t Var_d( Σ_{s∈C} … )`. Introduced because PGB
  "artificially deflates the bias in style clusters with many styles".

> **Correction to PLAN.md 4.2.** The table describes FGB as "Σ over style
> clusters of Var over descriptors of cluster proportion". That is **SCGB**, not
> FGB: FGB sums over all 217 individual styles and involves no clusters at all.
> The plan row has been corrected in the same commit.

## Current BiasScope implementation (v0.1.1)

`src/bias_scope/generated_text_based/fgb.py:23-24`, from paired scalar scores:

```
delta(t, k) = s_a(t, k) − s_b(t, k)
FGB         = mean_{t,k} |delta(t, k)|
```

## Verdict

**mismatch.**

| | Smith et al. | BiasScope v0.1.1 |
|---|---|---|
| Input | style vector over 217 classes per response | one scalar per response |
| Groups | many descriptors per axis | exactly two groups, A and B |
| Statistic | variance across descriptors, summed over styles | mean absolute pairwise difference |
| Structure | mean over samples, then variance, then sum, then mean over templates | single mean |

The paper's metric is a **variance across many descriptors of a distribution
over styles**; ours is a **mean absolute difference between two groups of
scalars**. Even restricted to two descriptors and one style, they differ:
variance of two values is `(a−b)²/2`, not `|a−b|`.

The v0.1.1 statistic is a perfectly sensible paired-gap measure — it is just not
Full Gen Bias.

## Required action

1. **Reimplement `FGB`** to the §A.7 formula, taking a `style_classifier`
   callable returning a probability vector per response, plus template and
   descriptor labels. Report FGB, and PGB per cluster (see
   [pgb.md](pgb.md)); consider SCGB as a third output since the paper argues PGB
   alone is misleading.
2. **Move the current behaviour** to `PairGapMean`, status `original`, with a
   deprecating alias (PLAN.md Section 12).
3. **The classifier is the blocker.** The 217-class style classifier is not in
   the repo and the paper does not link one. Options, in the order the Section 1
   playbook suggests: locate it (the paper attributes the style labels to prior
   BlenderBot work); otherwise ship a pluggable `style_classifier` with **no**
   default and mark the metric `adaptation` when a substitute is used,
   documenting which. **Do not** silently substitute a sentiment classifier and
   keep the name. Tracked as `REVIEW_LATER` RL-013 (`blocked`), so the Phase 2
   checkbox stays unticked until this is resolved.

## Validation possible

- **Tier 1:** Table 4 reports FGB per model (×1000). Reproducing it needs both
  the classifier and BlenderBot 2.0 3B generations. Record as `pending` with the
  classifier dependency noted; likely `no_published_reference` in practice until
  the classifier is found.
- **Tier 2:** not possible against this repo — it does not implement the metric.
  Best available is an oracle from the formula above.
- **Tier 3:** null (all descriptors identical → 0) and permutation invariance
  over samples apply. Swap antisymmetry does **not** apply: FGB is a variance,
  so it is symmetric, not antisymmetric, under relabelling descriptors. Record
  that as a documented exemption.

## Known limitations of the metric itself

- Entirely dependent on the style classifier: a classifier that cannot separate
  217 styles reliably makes the variance mostly noise.
- Summing variance over 217 styles means the value has no natural scale, which
  is why the paper multiplies by 1000 and compares only within a table.
- Variance across descriptors treats all differences as bias, including ones
  that reflect genuine topical differences between descriptors.
