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

