# DECISIONS

Design choices that shape the codebase, with the reason. Narrower than
`REVIEW_LATER.md`: this file records what the code *is*, that file records what
the maintainer should *check*. Newest at the bottom.

---

## 2026-08-22 · New dependencies: `pyyaml` and `pypdf`, dev extra only

Section 1 forbids new runtime dependencies without a note here.

Phase 0 introduces three YAML manifests (`sources/SOURCES.yaml`,
`validation/registry.yaml`, `verification/ledger.yaml`) and a source fetcher
that extracts PDF text. `pyyaml` and `pypdf` were added to
`[project.optional-dependencies] dev` **only**.

The installed library's runtime dependencies are unchanged: `numpy` and
`requests`. Nothing under `src/bias_scope/` imports either package; only
`scripts/sources/`, `scripts/verification/`, and CI do. JSON was considered and
rejected — the manifests are hand-edited, and every example in `PLAN.md` is
YAML. See `REVIEW_LATER.md` RL-003.

## 2026-08-22 · `ruff` gate scope: `src` and `tests`

The lint gate in CI is `ruff check src tests`, as `PLAN.md` Section 2 specifies.
`scripts/experiments/` is not linted; it holds the v0.1 reproduction drivers and
has 53 violations. The Phase 0 scripts under `scripts/sources/`,
`scripts/verification/`, and `scripts/validation/` are ruff-clean and should
stay that way. See `REVIEW_LATER.md` RL-009.

## 2026-08-22 · Evidence in `ledger.yaml` is verified, not trusted

`scripts/verification/render_ledger.py` checks every non-empty evidence value
before rendering it: a `path::test` must exist *and* be collected by pytest, a
bare path must exist, a `FILE.md#anchor` must contain the anchor, and
`exempt: reason` requires the reason. Anything else renders as `BROKEN` and the
script exits non-zero.

Section 1 makes the ledger the definition of "done", so a ledger that repeats
claims it has not checked would be worse than none at all.

## 2026-08-22 · Goldens regenerate only through a script that demands a reason

`scripts/verification/regen_golden.py` requires `--reason`, writes it into the
JSON as `regenerated_because`, prints the old and new score, and tells the
caller to add a `CHANGELOG` entry. `tests/golden/test_golden.py` fails if any
golden lacks that field, so hand-editing a golden to make a test pass is itself
a test failure.

## 2026-08-23 · `fairlearn` in the `dev` extra only

**Decision.** `fairlearn>=0.10.0` is added to the `dev` extra. It is **not** a
runtime dependency, and `bias_scope` never imports it.

**Why.** `DecodingTrust`'s own scorer calls
`fairlearn.metrics.demographic_parity_difference` and
`equalized_odds_difference` directly, so fairlearn *is* the reference
implementation for those two numbers. PLAN.md Section 1's light-core rule keeps
it out of the install, which means BiasScope reimplements them — and a
reimplementation has to be *shown* equal, not assumed equal.
`tests/oracles/test_fairlearn_oracle.py` compares the two across 2000 seeded
random draws and asserts exact agreement to 1e-12. Putting fairlearn in `dev`
is what makes that check run in CI rather than silently skipping.

**Consequence if removed.** The oracle test `importorskip`s and passes
vacuously; the equivalence claim in `docs/fidelity/decodingtrust.md` would then
rest on nothing a reader could re-run.

**Same pattern as** the existing scipy usage: a dev-only oracle for a formula
the library implements itself.


## 2026-09-09 · SEAT uses May et al.'s non-strict permutation p-value by default

**Decision.** `SEAT.evaluate` delegates to `WEAT` with `tie_policy="conservative"`
and `n_permutation_samples=100_000`, and exposes `tie_policy`,
`n_permutation_samples`, `permutation_seed` as parameters (defaults as stated).
`SEAT` keeps `fidelity="faithful"` with a populated `deviation_note`.

**Why.** May et al. 2019 Appendix A is explicit that SEAT departs from Caliskan
on the permutation test: it counts `s(Xi,Yi,A,B) >= s(X,Y,A,B)` ("the more
conservative non-strict inequality", because "the equality has positive
probability" in the nonparametric version), and samples 99,999 partitions +
1 hallucinated hit so "we can never observe a p-value less than 1e-5". The
authors' own `sent-bias/sentbias/weat.py` returns `(total_true + total_equal)
/ total`. BiasScope's `SEAT` was inheriting `WEAT`'s Caliskan default (`>`) and
exposed no way to change it, and `docs/fidelity/seat.md` wrongly said the
p-value was "unchanged". WEAT's existing `tie_policy="conservative"` branch
already implements exactly the `>=` count plus the "+1 observed partition", so
100,000 total evaluations reproduce the 1e-5 floor. The effect size is
untouched (Appendix A: "we compute the effect size identically").

**Fidelity label.** After this change SEAT matches the reference for both the
effect size and the p-value on precomputed embeddings, so `faithful` (as for
`WEAT`, which also carries a `deviation_note` for its raw-text path) rather than
`adaptation`. The residual gap — SEAT's bleached templates and per-encoder
pooling table are not reproduced — is an input-construction concern recorded in
the `deviation_note` and `docs/fidelity/seat.md`, not a scoring deviation.

**Most reversible option.** Callers who want the old behaviour pass
`tie_policy="strict"`; nothing is removed.

## 2026-09-15 · CEAT `run()` fixes; `make_protocol` gains `random_seed`

**Decision.** `CEAT.run()` now threads and records `random_seed` (mirroring
`WEAT.run`/`SEAT.run`), and overrides `_interval`/`_count_items` to report its
own random-effects `CES ± Z_95·SE(CES)` (`ci_method="random_effects"`) and
`n = n_samples`, instead of a Hedges-Olkin interval built from `|X|`,`|Y|`.
`bias_scope.result.make_protocol` gained a new `random_seed` field (alongside
the existing `permutation_seed`), since CEAT's Monte-Carlo context resampling
is not a permutation test and deserves its own protocol field rather than
overloading `permutation_seed`'s name.

**Why.** `CEAT().run(seed=42)` was non-reproducible: `evaluate()`'s own
`random_seed=None` default falls back to OS entropy, and nothing in `run()`
passed a seed through, so two calls returned effect sizes of opposite sign.
Separately, `run()`'s Hedges-Olkin interval — inherited from
`EmbeddingMetric._interval`, which is right for WEAT/SEAT effect sizes on
`|X|`/`|Y|` stimuli — is the wrong quantity for CEAT: CEAT's uncertainty comes
from the 10,000-sample random-effects model (paper Appendix), not from the
handful of target stimuli. On one test case the wrong interval was ~13x wider
than the true one. Found during a from-scratch CEAT audit (2026-09-15),
verified against the reference `weiguowilliam/CEAT@497e2958` for the scoring
math (which is unaffected — only `run()`'s framework glue was wrong).

**Not changed (documented instead, see `REVIEW_LATER.md` RL-038, RL-039):**
CEAT's context sampling follows the paper's prose (without replacement once a
stimulus has `n_samples` contexts) rather than the reference script (which
always samples with replacement); its p-value follows the paper's two-sided
Appendix formula rather than the reference script's one-sided signed formula
(which cannot reproduce the paper's own Table 1). Both are judgment calls
where the paper and its own reference script disagree with each other, so
"follow the code" is not a clean tiebreaker — recorded as `verify`, not
silently resolved.

**Consequence if reverted.** `CEAT().run()` regresses to non-reproducible
results and a misleadingly narrow-or-wide, wrong-basis interval; `evaluate()`
is untouched either way.
