# WEAT — Word Embedding Association Test

**Cited source:** Caliskan, Bryson & Narayanan 2017, *Semantics derived
automatically from language corpora contain human-like biases*, Science
356(6334). [arXiv:1608.07187](https://arxiv.org/abs/1608.07187), read as
`sources/papers/weat.txt`.
**Primary replication artifact:** Caliskan's [WEFAT and WEAT replication data
and code](https://doi.org/10.7910/DVN/DX4VWP), archived by the authors on
Harvard Dataverse and cited by the Science paper. **Secondary implementation:**
[W4ngatang/sent-bias](https://github.com/W4ngatang/sent-bias) @
`e3559fb669ca4832743b42fee715994c15c7f1af`, CC-BY-4.0, is May et al. 2019's
later reimplementation and is used here only as a secondary comparison point.
**Sections and files read:** the "test statistic / p-value / effect size"
definition block; `sentbias/weat.py:82-152` (`p_val_permutation_test`),
`:174-176` (`stdev_s_wAB`), `:178-192` (`effect_size`).
**Family / access:** embedding / `embeddings`

## Definition in the paper

For target sets `X`, `Y` of equal size and attribute sets `A`, `B`:

```
s(w, A, B)    = mean_{a∈A} cos(w, a) − mean_{b∈B} cos(w, b)
s(X, Y, A, B) = Σ_{x∈X} s(x,A,B) − Σ_{y∈Y} s(y,A,B)

p-value    = Pr_i[ s(Xi, Yi, A, B) > s(X, Y, A, B) ]   over all equal-size
                                                        partitions of X ∪ Y
effect size = [ mean_{x∈X} s(x,A,B) − mean_{y∈Y} s(y,A,B) ]
              / std-dev_{w∈X∪Y} s(w, A, B)
```

WEAT is **both** numbers. The paper reports an effect size *and* a p-value for
every test it runs.

## The `std-dev` ambiguity, resolved

The paper writes only `std-dev`, never saying whether it is the population
(`ddof=0`) or sample (`ddof=1`) standard deviation. At Caliskan's own n = 8 per
group the two differ by `sqrt(16/15)` ≈ 3.3%, which is larger than the gap the
v0.1 reproduction was trying to explain — so this had to be settled, not
assumed. Three independent lines agree on **`ddof=1`**:

1. **The reference code.** `sentbias/weat.py:175` is
   `np.std(s_wAB(...), ddof=1)` — explicit.
2. **The paper's own number.** On Caliskan's GloVe-840B vectors, WEAT-6
   gender-career:

   | convention | our d | paper d | relative error |
   |---|---|---|---|
   | `ddof=1` | 1.8139 | 1.81 | **0.22%** |
   | `ddof=0` | 1.8734 | 1.81 | 3.50% |

   A 16× difference in agreement.
3. **The 6B replication** shows the same ordering: 1.6938 (`ddof=1`) versus
   1.7494 (`ddof=0`).

`src/bias_scope/embeddings_based/weat.py:244` already used `ddof=1`, so this
**confirms** the implementation rather than changing it. Closes
`REVIEW_LATER.md` RL-008.

## Current BiasScope implementation

`src/bias_scope/embeddings_based/weat.py`. The effect size matches the paper and
the reference exactly, including `ddof=1` and the mean-of-cosines form of
`s(w, A, B)`.

**v0.1.1 omitted the permutation test entirely** — half of the definition. The
canonical BiasScope mode now uses the paper's strict `>` comparison. Exact
enumeration covers all `C(2n, n)` partitions up to 100,000; larger tests use a
sampled estimate with exactly `n_permutation_samples` partition evaluations.
`tie_policy="conservative"` is an explicitly noncanonical compatibility mode
for May et al.'s `sent-bias` convention (`>=` and one observed pseudo-draw).

Raw string inputs are another explicit BiasScope extension. They are encoded by
the chosen sentence-transformer or Hugging Face model and therefore involve its
tokenization, subword handling, truncation, and pooling. Canonical WEAT takes
precomputed static word vectors; Caliskan et al. used cased GloVe 840B/300d.

## Verdict

**faithful for canonical static embeddings, with documented extensions.**

The effect size was already faithful; the missing p-value was a completeness gap
rather than a wrong statistic, and it is now closed. The oracle in
`tests/oracles/weat_oracle.py` agrees with the library to 1e-8 over 200 random
inputs.

## Required action

None outstanding. Two follow-ups:

- The `n_target_group_1`/`_2` keys in `details` predate `MetricInfo` and could
  fold into `breakdown`; cosmetic.
- The Hedges-Olkin interval `run()` attaches is BiasScope's addition, not the
  paper's — the paper's own uncertainty statement is the permutation p-value.
  Both are reported; see `REVIEW_LATER.md` RL-016 on the interval's formula.

## Validation possible

- **Tier 1:** Caliskan Table 1 reports effect sizes and p-values for ten tests
  on GloVe-840B. WEAT-6 is already reproduced at 0.22%. The remaining nine are
  registry candidates and the strongest Tier-1 evidence the library has.
- **Tier 2:** compare directly with the authors' Dataverse artifact; sent-bias
  is a secondary compatibility target for conservative ties.
- **Tier 3:** null, swap antisymmetry, permutation invariance and scale
  invariance all apply and hold. Monotonicity applies.

## Known limitations of the metric itself

- n = 8 per group makes any confidence interval nearly uninformative; the
  permutation p-value is the meaningful uncertainty statement.
- The word lists come from IAT studies and inherit their construct validity
  debates; Blodgett et al. 2021 applies.
- Cosine similarity in a static embedding space says nothing about model
  behaviour downstream.
