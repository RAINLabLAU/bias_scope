# CEAT — Contextualized Embedding Association Test

**Cited source:** Guo & Caliskan 2021, *Detecting Emergent Intersectional
Biases: Contextualized Word Embeddings Contain a Distribution of Human-like
Biases*, AIES 2021. [arXiv:2006.03955](https://arxiv.org/abs/2006.03955).
**Reference implementation:**
[weiguowilliam/CEAT](https://github.com/weiguowilliam/CEAT) @
`497e2958a152ad70004ef6f3b613b98efdd26389` — **no license file** (RL-015).
Cloned and read directly for the 2026-09 audit (superseding the earlier
repo-layout-only read).
**Sections and files read:** §"Contextualized Embedding Association Test" (CES
definition); §"Random-Effects Model"; Appendix "Random-Effects Model Details"
(the full `Q`/`c`/`τ²`/`v_i`/`SE` estimator and the two-sided p-value formula);
§on sampling N contexts (N = 1,000 and N = 10,000 both reported); `code/ceat.py`
in full (`associate`, `effect_size`, `ceat_meta`).
**Family / access:** embedding / `embeddings`

## Definition in the paper

A contextualized embedding gives a *distribution* of WEAT effect sizes rather
than one value, so CEAT samples N context-combinations, computes a WEAT effect
size in each, and combines them with a **DerSimonian-Laird random-effects
meta-analysis** (Appendix "Random-Effects Model Details"):

```
ES_i = (mean_x s(x,A,B) - mean_y s(y,A,B)) / std_dev_{X∪Y} s(w,A,B)     # WEAT's d, ddof=1
V_i  = ( std_dev_{X∪Y} s(w,A,B) )²                                      # "the square of std_dev"
W_i  = 1/V_i ;  c = ΣW_i - ΣW_i²/ΣW_i ;  Q = ΣW_iES_i² - (ΣW_iES_i)²/ΣW_i
σ²_between = max(0, (Q-(N-1))/c)
v_i  = 1/(V_i + σ²_between)
CES  = Σ v_i ES_i / Σ v_i                     SE(CES) = sqrt(1/Σ v_i)
P_combined = 2 × [1 - Φ(|CES/SE(CES)|)]       # explicitly two-sided
```

Sampling (main text): "for each stimulus that appears in at least N sentences,
we randomly sample one of its CWE vectors **without replacement**. If a
stimulus occurs in less than N sentences ... we randomly sample ... **with
replacement**." N = 1,000 and N = 10,000 are both reported and agree to
within 0.1.

## Definition in the reference code (`code/ceat.py`)

```python
def effect_size(X, Y, A, B):
    delta_mean = mean(assoc(X)) - mean(assoc(Y))
    s = [assoc(w) for w in concat(X, Y)]
    std_dev = np.std(s, ddof=1)
    return delta_mean / std_dev, std_dev**2          # (ES_i, V_i) -- V_i = std**2, matches the paper

for i in range(N):
    X = [weat_dict[wd][np.random.randint(0, len(weat_dict[wd]))] for wd in ...]   # WITH replacement, unconditionally
    ...
# Q, c, tao_square: identical to the paper's formulas
z = pes / np.sqrt(v)
p_value = scipy.stats.norm.sf(z)                     # ONE-sided on signed z, no abs()
```

Two places where the reference **code** disagrees with the **paper text**:

1. **Sampling.** The paper's prose says "without replacement" once a stimulus
   has ≥ N contexts; the code always calls `np.random.randint` (with
   replacement), for every stimulus, every iteration, regardless of how many
   contexts it has.
2. **P-value.** The paper's Appendix formula is two-sided on `|CES/SE|`, and
   Table 1's own reported p-values are only consistent with that (a negative
   CES with `|z|≈1.6` gets p≈0.11 in Table 1, not the ≈0.95 that a one-sided
   `sf(z)` on a negative z would give). The shipped script uses
   `scipy.stats.norm.sf(z)` — one-sided, signed, no `abs()` — which for a
   positive CES returns exactly half the paper's value and for a negative CES
   returns ≈1. This looks like a bug in the script that the paper's own
   reported numbers do not exhibit.

## Current BiasScope implementation

`src/bias_scope/embeddings_based/ceat.py` with `_helpers.py::_weat_effect_components`
(per-sample ES and pooled SD) and `_helpers.py::_ceat_random_effects` (the
DerSimonian-Laird pooling). Verified this audit against the cloned reference,
differentially, on random inputs:

- per-sample `ES_i` and `V_i = np.std(s, ddof=1)**2` agree with
  `code/ceat.py::effect_size` to **1e-16**.
- `W_i`, `Q`, `c`, `τ²`, `CES`, `SE(CES)` agree with a fresh port of
  `code/ceat.py::ceat_meta`'s meta-analysis to **1e-16**.
- default `n_samples = 10_000`, matching the paper's main N. (`REVIEW_LATER`
  RL-021 previously flagged the default as 100 with an open decision to raise
  it; the code was already reworked to 10,000 by the time RL-021 was written
  down, and that entry is closed as stale — see RL-021 in `REVIEW_LATER.md`.)

**BiasScope follows the paper text, not the reference code, on the two points
above** — this was undocumented before the 2026-09 audit and is now recorded
here and in `MetricInfo.deviation_note`:

- **Sampling** (`_sample_context_indices`): `rng.choice(n_contexts, size=n_samples,
  replace=n_contexts < n_samples)` — without replacement once a stimulus has
  enough contexts, exactly as the paper's prose describes, unlike the reference
  script's unconditional `randint`. This means BiasScope's numbers will not
  exactly reproduce a Tier-2 run against the reference script for any stimulus
  with `n_contexts >= n_samples`; the divergence shrinks as `n_contexts` grows
  past `n_samples` (fewer birthday collisions in the reference's
  with-replacement draws) and vanishes when `n_contexts < n_samples` (both
  schemes sample with replacement there). **`verify`**: which scheme the
  authors actually ran to produce Table 1 is not settled by reading the script
  alone; the prose and the script disagree with each other.
- **P-value** (`evaluate`): `p_value = math.erfc(abs(z_score)/math.sqrt(2))`
  = `2·[1-Φ(|z|)]` — the paper's own two-sided formula, chosen because it
  matches the paper's stated formula *and* Table 1's reported numbers, where
  the reference script's literal `norm.sf(z)` does not. **`verify`**: flagged
  in case the authors' actual Table 1 pipeline differs from the script in
  `ceat.py` (e.g. a later, unpublished fix).

**Fixed this audit (`run()` was materially broken for CEAT):**

- `CEAT.run(seed=...)` did not thread `seed` into `random_seed`, and
  `random_seed=None` fell back to OS entropy — so `run()` was **not
  reproducible** even with a fixed `seed` (two calls at `seed=42` could return
  CES of opposite sign). `CEAT.run` now threads and records `random_seed` the
  way `WEAT.run` / `SEAT.run` do; `evaluate(random_seed=...)` was already
  correct on its own.
- `run()` attached a Hedges-Olkin interval built from `|X|`, `|Y|` (the
  *stimulus* counts, e.g. 3 vs 3) instead of CEAT's own `SE(CES)` — on one
  test case this produced a CI **13x wider** than the true random-effects
  interval, under the wrong label (`ci_method="hedges_olkin"`). `CEAT`
  now overrides `_interval` to return `CES ± Z_95 · SE(CES)` with
  `ci_method="random_effects"`, and `_count_items` to report `n = n_samples`
  (the meta-analysis sample size) rather than `|X|+|Y|`.

## Verdict

**faithful for `evaluate()` (the CES and its random-effects pooling are an
exact reproduction of the reference); `run()` is now fixed to report CEAT's
own interval, reproducibly.**

## Required action

None open. RL-021 (stale: cited default `n_samples=100`, code already reads
10,000) closed. Two `verify` items recorded above (sampling-scheme choice,
p-value formula) pending a byte-for-byte comparison against a real Reddit +
neural-LM run, which needs the CWE-extraction pipeline this class does not
implement (see below).

## Validation possible

- **Tier 1:** Guo & Caliskan report CES per test per model at both N values.
  Reproducible in principle; needs the contextual embedding pipeline
  (`code/generate_ebd_*.py`: last-subtoken CWE from Reddit sentences), which
  BiasScope does not implement — callers supply the
  `{stimulus: (n_contexts, dim)}` mapping directly.
- **Tier 2:** the repo states **no license**, so its code may be run for
  comparison but not vendored (RL-015). A byte-for-byte comparison must also
  resolve the sampling-scheme and p-value `verify` items above, or it will
  report spurious disagreement.
- **Tier 3:** null, swap antisymmetry, permutation invariance and scale
  invariance apply, as for WEAT, and hold numerically. CEAT is stochastic in
  N, so Tier-3 tests need a fixed seed — `run(seed=...)` now actually provides
  one.

## Known limitations of the metric itself

- The result depends on the corpus the N contexts are sampled from; the paper
  uses Reddit, which is not neutral.
- CES inherits WEAT's small word-list construct-validity issues on top of the
  sampling variance.
