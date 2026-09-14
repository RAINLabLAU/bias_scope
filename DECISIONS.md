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

## 2026-09-12 · `anthropic` and `huggingface_hub` in a new `agent` extra

**Decision.** `anthropic>=0.40.0` and `huggingface_hub>=0.24.0` are added to a
new `[project.optional-dependencies] agent` extra (folded into `all`, like
every other extra). `src/bias_scope/` still depends on nothing but `numpy` and
`requests` — this extra is consumed only by the new `src/bias_scope_agent/`
package.

**Why.** `bias_scope_agent` is a single-LLM tool-calling agent (the `anthropic`
SDK drives its tool-use loop) that can optionally live-inspect a target model
identifier by fetching `config.json` from the Hugging Face Hub
(`huggingface_hub.hf_hub_download`, behind the `BIASSCOPE_AGENT_INSPECT_LIVE`
flag). Both imports are deferred to inside the specific functions that need
them (`bias_scope_agent.loop.AgentLoop.__init__`,
`bias_scope_agent.introspection._download_hf_config`), so importing
`bias_scope_agent` itself does not require either package unless a real
Anthropic client or a live Hub lookup is actually used — `AgentSession`,
`tools.py`, and the test suite all run with a `StubBackend` and no client at
all.

**Consequence if removed.** `bias_scope_agent.loop.AgentLoop()` (with no
`client=` override) and `bias_scope_agent.introspection.inspect_model(...,
live=True)` on a Hub identifier would raise `ImportError` naming the missing
package. Every unit test in `tests/test_bias_scope_agent/` passes a fake
client or mocks the Hub call, so the fast suite does not depend on this extra
being installed; `tests/integration/test_bias_scope_agent_tiny_model.py` does
not need it either (it exercises `HuggingFaceBackend`, not `huggingface_hub`,
and uses `WEAT` with raw embedding arrays rather than a live model call — see
`REVIEW_LATER.md` RL-041).

**Not litellm.** `bias_scope` already depends on `litellm` (the `llm` extra)
for target-model backends, and `bias_scope_agent.introspection` also uses it
opportunistically to probe an API-endpoint identifier. But the agent LLM
itself (the model running the tool-calling loop, as opposed to the model being
bias-tested) uses the `anthropic` SDK directly rather than going through
litellm, per an explicit decision made before implementation: v1 targets one
provider (Claude) with native tool-use, not a multi-provider abstraction.

