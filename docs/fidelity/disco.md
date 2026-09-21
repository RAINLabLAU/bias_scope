# DisCoMetric — Discovery of Correlations

**Cited source:** Webster, Wang, Tenney, Beutel, Pitler, Pavlick, Chen, Chi &
Petrov 2020, *Measuring and Reducing Gendered Correlations in Pre-trained
Models*. [arXiv:2010.06032](https://arxiv.org/abs/2010.06032), read as
`sources/papers/discometric.pdf`.
**Reference implementation:** none located. `code_status: none_found` in
`sources/SOURCES.yaml` with the search log. The paper states no code URL; the
templates are given in its Appendix, and the two word lists are external (US SSA
baby-name statistics; Zhao et al. 2018a's gendered noun list).
**Sections and files read:** §"Discovery of Correlations (DisCo)" (the full
definition: templates, two slots, top-three fills, χ², Bonferroni, aggregation);
Table 2 (reported values for ALBERT and BERT).
**Family / access:** probability / `logits`

## Definition in the paper

Templates have two slots, e.g. `"[PERSON] studied [BLANK] at college."`

1. **`[PERSON]` is filled manually** from a gender-labelled word list. Two
   variants are reported:
   - **Names** — US SSA names with >80% of counts in one gender.
   - **Terms** — noun phrases `"the NOUN"` from Zhao et al. 2018a's gendered
     noun list.
   The paper notes the labels are binary here but that "DisCo can accommodate
   word lists with any number of label values".
2. **`[BLANK]` is filled by the model.** A candidate fill counts as "supplied"
   if it appears among the model's **top-three highest scoring fills**. The
   paper justifies the small number: "the probability distribution shape can
   differ substantially between models".
3. **Significance.** A fill is preferentially associated with one gender when
   "the χ² metric rejects a null hypothesis of equal prediction rate", with a
   **Bonferroni correction to the standard p-value of 0.05**, "since our
   procedure runs many significance tests".
4. **Aggregation.** "we define the metric to be the number of fills
   significantly associated with gender, averaged over templates."

So DisCo is a **count of significant fills per template, averaged over
templates**. It is `≥ 0`, **0 means no fill is significantly gendered**, and it
is bounded above by the number of distinct fills the top-3 sets can produce.

This settles the open question in PLAN.md 4.2 ("χ² test (p<0.05 with the paper's
correction if any)"): **the correction is Bonferroni, and it is explicit.**

The paper also says templates include "multiple related variants of each
template (e.g. by inserting 'often' and 'always')" for robustness, and that 276
templates were collected.

## Current BiasScope implementation (v0.1.1)

`src/bias_scope/probability_based/disco.py`. `evaluate` takes **two prompt
strings** and returns, per `:32-38`:

```
DisCo score = |T_A Δ T_B|      # symmetric difference of the two top-k sets
                               # 0 -> identical, 2k -> disjoint
```

## Verdict

**faithful**, as of v0.2.0.

**History: v0.1.1 was a `mismatch`.** The audit below is of that version.

| | Webster et al. | BiasScope v0.1.1 |
|---|---|---|
| Input | many templates × a gender-labelled word list | two prompt strings |
| Fill cutoff | top **three**, fixed and justified | top `k`, caller-chosen |
| Test | χ² on prediction rates, Bonferroni-corrected | none |
| Statistic | mean count of *significant* fills per template | size of a set symmetric difference |
| Range | ≥ 0, 0 = no significant association | 0 … 2k |

There is no significance test anywhere in the v0.1.1 code, so the central idea
of the metric — that a difference in fills counts only when it is unlikely under
equal prediction rates — is absent. The symmetric difference will report a large
value for two top-k sets that differ purely by sampling noise, which is exactly
what the χ² test exists to prevent.

The v0.1.1 statistic is a reasonable descriptive measure of top-k sensitivity;
it is just not DisCo.

## Resolution (v0.2.0)

1. Implemented from the paper's §"Discovery of Correlations": templates with
   `[PERSON]`/`[BLANK]`, top-3 fills, a per-candidate-fill chi-square against
   equal prediction rates, **Bonferroni-corrected** α = 0.05, and the mean count
   of significant fills per template.
2. `chi_square_2xk` and `chi_square_p_value` are written out rather than pulled
   from scipy, so the core install stays light. **Validated against
   `scipy.stats.chi2_contingency` on five contingency tables — exact agreement
   to 1e-9 on both the statistic and the p-value.**
3. The v0.1.1 top-k symmetric difference is preserved as `TopKFillDivergence`
   (`fidelity: original`), per PLAN.md Section 12.
4. `DisCoMetric` no longer needs torch: the caller supplies the fills, so the
   metric is pure Python and installs with the core package.

The test that most clearly separates the two: with one name per gender and
completely disjoint top-3 fills, the old statistic reported 6 and DisCo reports
**0** — one name pair cannot be distinguished from chance, which is exactly what
the chi-square is for.

## Required action (now largely done)

1. **Reimplement `DisCoMetric`** from the definition above: templates, a
   gender-labelled word list for `[PERSON]`, top-3 fills, per-candidate-fill χ²
   against equal prediction rate, Bonferroni-corrected α = 0.05, then the mean
   count of significant fills per template.
2. **Move the current behaviour** to `TopKFillDivergence`, status `original`,
   with a deprecating alias (PLAN.md Section 12).
3. Vendor the templates from the paper's Appendix with attribution, and
   implement loaders for the two external word lists. Record SHA-256s.
4. **Two details the paper leaves open**, to be decided and logged as `decide`
   when implementing: the exact contingency table the χ² is computed over
   (per-fill 2×2 of supplied/not-supplied by gender is the natural reading), and
   whether any expected-count threshold or Yates correction is applied. PLAN.md
   Section 12's pre-decided default applies: implement the simpler reading,
   expose the other only if it is a one-line difference.

## Validation possible

- **Tier 1: available and precise.** Table 2 reports, for ALBERT Base / ALBERT
  Large / BERT Base / BERT Large (uncased):
  - DisCo (Terms): 0.4, 0.0, 0.8, 1.0
  - DisCo (Names): 3.7, 3.1, 3.7, 3.4

  All four models are public. This is one of the better Tier-1 targets in the
  library and should get registry entries naming Table 2 and the exact cell.
- **Tier 2:** not possible — no reference implementation exists. Validate with
  Tier 1 plus an oracle instead.
- **Tier 3:** null (identical fills for both genders → 0) and permutation
  invariance apply. Swap antisymmetry does **not**: DisCo is a count, so it is
  symmetric under swapping the two gender labels. Document as an exemption.

## Fixed in the 2026-09-18 audit follow-up

Independently re-verified the chi-square/p-value implementation against
`scipy.stats.chi2_contingency` fresh (bit-exact to displayed precision on 5
tables, including a degenerate zero-variance case scipy itself errors on)
and re-confirmed `run()` end-to-end by execution — both were already
correct, not new fixes. Two things found and fixed:

- **The shipped example was still the v0.1.1 API and crashed immediately.**
  `examples/probability_based/disco.py` called
  `metric.evaluate(template=..., attr_a=..., attr_b=..., k=5)` — the old
  two-prompt symmetric-difference signature, now `TopKFillDivergence`'s —
  against the reimplemented `DisCoMetric`, whose signature is
  `evaluate(templates, person_words, top_k_fills, ...)`. It had never been
  updated when the two classes were split. Rewritten to demonstrate the
  current chi-square/Bonferroni API with an offline `top_k_fills` callback.
- **Stale `validation/registry.yaml` notes.** All eight DisCo Tier-1 rows
  said "Blocked until DisCoMetric is reimplemented ... the current class
  computes a different statistic" — true for v0.1.1, false since the
  reimplementation. Corrected to say what's actually still pending
  (running the reproduction against the cited model/templates/word list),
  not a nonexistent implementation gap.

## Known limitations of the metric itself

- The paper states its own upper-bound problem: restricting to top-3 fills caps
  how many correlations can be discovered per template.
- Binary gender in both released word lists, though the definition generalises.
- Bonferroni over many tests is conservative; a model with real but weak
  correlations can score 0, as ALBERT Large does on Terms.
