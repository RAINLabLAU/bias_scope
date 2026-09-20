# CounterfactualSentimentBias and SocialGroupSubstitution

Two classes, one source paper, so one note.

**Cited source:** Huang, Zhang, Elsayed, Juefei-Xu, Reid, Kohli et al. 2020
(DeepMind), *Reducing Sentiment Bias in Language Models via Counterfactual
Evaluation*, Findings of EMNLP 2020.
[arXiv:1911.03064](https://arxiv.org/abs/1911.03064).
**Reference implementation:** none located — `code_status: none_found` with the
search log in `sources/SOURCES.yaml`.
**Sections and files read:** §on the Wasserstein-1 formulation (eq. 1, the
`E_τ|p(S(x)>τ) − p(S(x̃)>τ)|` identity); eq. 2 (the counterfactual fairness
condition); §"Individual Fairness Metric" (Average I.F. over templates);
§"Group Fairness Metric" (Average G.F., subgroup vs whole evaluation set).
**Family / access:** generated_text / `completions`

## Definition in the paper

Counterfactual sentiment bias is the **Wasserstein-1 distance** between the
sentiment distributions of an input and its counterfactual:

```
W1(P, Q) = E_{τ~U[0,1]} | p(S(x) > τ) − p(S(x̃) > τ) |        (1)
```

The paper notes explicitly that W1 "does not require assumptions on their shape
(e.g. symmetry)" — the reason for using a distance between distributions rather
than comparing means.

Two aggregate metrics are then defined:

- **Average Individual Fairness (I.F.)** — the mean W1 between each evaluation
  sentence's sentiment distribution and each of its counterfactuals, across all
  M templates and all pairs of sensitive-attribute values.
- **Average Group Fairness (G.F.)** — for each subgroup `a`, the W1 between that
  subgroup's sentiment distribution `P^a_S` and the distribution over the
  **entire evaluation set** `P*_S`; then the mean over subgroups.

## CounterfactualSentimentBias — **faithful**, as of v0.2.0

**v0.1.1 was a `mismatch`:** it computed `mean_{t,k}(s_a − s_b)`, the signed
mean of paired sentiment differences. That is not a distance and behaves
differently in the case that matters: a model biased toward group A on some
templates and toward group B on others has those differences **cancel**, scoring
near 0, while W1 is positive for every template that differs.

Now computes W1 per template over that template's K samples, averaged over
templates — the paper's I.F. The Wasserstein-1 implementation lives in
`bias_scope/stats.py::wasserstein_1` and was **validated against
`scipy.stats.wasserstein_distance` on seven cases including unequal sample
sizes, matching to 1e-9**.

The v0.1.1 quantity is retained in `details["signed_mean_difference"]`, where it
is genuinely useful: it says *which* group is favoured, which a distance cannot.

### Fixed in the 2026-09-18 audit follow-up

A from-scratch audit re-derived `wasserstein_1` independently (bit-exact
against `scipy.stats.wasserstein_distance` over 200 random trials — no
computational defect) and found three documentation-level issues:

1. **Fixed: a stale, impossible sign claim.** The docstring's
   "Interpretation" section, the shipped example's printed output, and its
   copied `docs/api` page all claimed `csb_score` ("CSB") is signed — "CSB
   < 0: group B receives more positive sentiment" — left over from before
   the v0.1.1→v0.2.0 fix above. Since `csb_score` is a Wasserstein-1
   distance, it is **always >= 0**; a group-A-all-negative,
   group-B-all-positive counterexample gives `csb_score = 1.6` (positive),
   not negative. All three copies now correctly state `csb_score` has no
   sign and point to `signed_mean_difference` for direction.
2. **Documented, not changed: the sentiment score domain.** Huang et al.
   define `S ∈ [0, 1]` (§3); this class validates `[-1, 1]` instead, and the
   example uses `[-1, 1]`-scaled scores (e.g. a raw VADER compound score).
   `wasserstein_1` is domain-agnostic so this isn't a computation bug, but
   `csb_score` is only numerically comparable to the paper's own reported
   I.F. values (Figures 4-17, Tables 5-6) when scores are actually scaled to
   `[0, 1]`. Now stated explicitly in the class docstring and
   `deviation_note`.
3. **Documented, not changed: two-group-only scope.** This class computes
   one pairwise term of eq. 3 — exactly the paper's I.F. for a binary
   attribute (Name), but for a >2-valued attribute (Country: 10 values,
   Occupation: 29) reproducing eq. 3 requires calling `evaluate()` once per
   unordered pair and averaging yourself; a single call is not itself the
   paper's I.F. for such attributes. Now stated explicitly in the docstring.

All three logged/tracked as `REVIEW_LATER` RL-091.

## SocialGroupSubstitution — **adaptation**

The class has the right structure — it reports `individual_unfairness` and
`group_disparity`, mirroring the paper's I.F. and G.F. — but both are computed
as a **range (max − min)** across substituted values rather than a Wasserstein-1
distance between distributions.

A range is determined entirely by the two extreme groups and ignores everything
between them; W1 uses the whole distribution. With three or more groups the two
can order models differently.

`group_disparity` also compares subgroups **to each other**, whereas the paper's
G.F. compares each subgroup **to the whole evaluation set**.

Left as `adaptation` rather than converted: unlike
`CounterfactualSentimentBias`, this class's API is built around per-substitution
scalar scores rather than distributions, so the change is a signature change,
not a formula swap. Recorded as `REVIEW_LATER` RL-027.

## Required action

- Convert `SocialGroupSubstitution` to W1 over distributions, and make G.F.
  compare each subgroup against the full evaluation set (RL-027).
- No published reference implementation exists, so an oracle plus Tier 1 is the
  only validation route.

## Validation possible

- **Tier 1:** Huang et al. report I.F. and G.F. for GPT-2 across country, name
  and occupation attributes. Reproducible in principle; the sentiment model is
  the uncertain part.
- **Tier 2:** not possible — no code release located after all six search steps.
- **Tier 3:** null (identical distributions → W1 = 0) applies and is exact.
  Swap antisymmetry does **not** apply to `CounterfactualSentimentBias` any
  more: W1 is a distance, so it is symmetric, not antisymmetric, under swapping
  the groups. That is a change from v0.1.1's signed statistic and should be
  recorded as a documented exemption in the property tests.

## Known limitations of the metrics themselves

- Both depend entirely on the sentiment scorer, and Huang et al. show results
  shift with it.
- W1 is unbounded above in principle; on sentiment scores confined to [−1, 1] it
  is bounded by 2, but the scale is not intuitive and only comparable across
  runs that use the same scorer and the same K.
- Counterfactual substitution assumes the swapped token is the only thing that
  changed, which name-based substitution does not always guarantee.
