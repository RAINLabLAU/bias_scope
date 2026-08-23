# CEAT — Contextualized Embedding Association Test

**Cited source:** Guo & Caliskan 2021, *Detecting Emergent Intersectional
Biases: Contextualized Word Embeddings Contain a Distribution of Human-like
Biases*, AIES 2021. [arXiv:2006.03955](https://arxiv.org/abs/2006.03955).
**Reference implementation:**
[weiguowilliam/CEAT](https://github.com/weiguowilliam/CEAT) @
`497e2958a152ad70004ef6f3b613b98efdd26389` — **no license file** (RL-015).
**Sections and files read:** §"combined effect size (CES)" (the weighted-mean
definition and the inverse-variance weights); §"Random-Effects Model" (Hedges &
Vevea 1998, DerSimonian & Laird 1986); §on sampling N contexts (N = 1,000 and
N = 10,000 both reported).
**Family / access:** embedding / `embeddings`

## Definition in the paper

A contextualized embedding gives a *distribution* of WEAT effect sizes rather
than one value, so CEAT samples N contexts, computes a WEAT effect size in each,
and combines them with a **random-effects meta-analysis**:

```
CES(X, Y, A, B) = Σᵢ vᵢ · ESᵢ / Σᵢ vᵢ
```

> where `vᵢ` is the inverse of the sum of in-sample variance `Vᵢ` and
> between-sample variance in the distribution of random effects `σ²_between`.

The paper is explicit that a fixed-effects model would be wrong here: "since the
effect sizes calculated with the CWE in different contexts are expected to vary,
we cannot assume a fixed-effects model". It reports results at both **N = 1,000
and N = 10,000** sampled contexts.

## Current BiasScope implementation

`src/bias_scope/embeddings_based/ceat.py` with
`_helpers.py::_compute_random_effects_weights`. The estimator is the real
DerSimonian-Laird procedure, not a shortcut:

```python
V_i     = 2/n + ES_i² / (4n − 4)            # within-sample variance
w_fixed = 1 / V_i
Q       = Σ w_fixed (ES_i − ES̄_fixed)²
C       = Σ w_fixed − Σ w_fixed² / Σ w_fixed
τ²      = max(0, (Q − df) / C)
w_i     = 1 / (V_i + τ²)                    # random-effects weights
```

which is exactly `vᵢ = 1 / (Vᵢ + σ²_between)`.

Two fixes in 0.2.0:

- The helper's docstring claimed `w_i = 1/(var + tau^2 + epsilon)` with "tau^2:
  between-sample variance (max(0, var - 0))" — a description of a simpler
  estimator than the code actually implements. Corrected to state the real
  formula and cite the paper.
- Default pooling changed from `"mean"` to `"cls"`, matching the reference
  protocol (see [seat.md](seat.md) for the position-0 detail).

## Verdict

**faithful.**

The combined effect size and the random-effects weighting match the paper.

## Required action

**One open question, `decide`:** the default `n_samples` is **100**, while the
paper reports N = 1,000 and N = 10,000. CES is a weighted mean over sampled
contexts, so N controls its variance directly, and 100 will be noisier than
anything the paper reports. Either raise the default to 1,000 or document that
the default is a fast setting and reproductions must set N explicitly. Recorded
as `REVIEW_LATER` RL-021.

## Validation possible

- **Tier 1:** Guo & Caliskan report CES per test per model at both N values.
  Reproducible in principle; needs the contextual embedding pipeline.
- **Tier 2:** the repo states **no license**, so its code may be run for
  comparison but not vendored (RL-015).
- **Tier 3:** null, swap antisymmetry, permutation invariance and scale
  invariance apply, as for WEAT. Note CEAT is stochastic in N, so Tier-3 tests
  need a fixed seed — the class already takes `random_seed`.

## Known limitations of the metric itself

- The result depends on the corpus the N contexts are sampled from; the paper
  uses Reddit, which is not neutral.
- CES inherits WEAT's small word-list construct-validity issues on top of the
  sampling variance.
