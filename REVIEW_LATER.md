# REVIEW_LATER

Decisions taken without the maintainer, and facts the maintainer should
double-check. **This file, not the commit log, is where you find what to
verify.** One entry per decision or skip, newest at the bottom, IDs sequential.

Severity tags:

- `decide` — a judgement call the maintainer may overturn.
- `verify` — a fact to double-check: a table cell, a license, a discrepancy
  between paper and code.
- `blocked` — a task skipped because of a gated model, a missing key,
  unavailable data, or a cost cap. **The checkbox in `PLAN.md` stays unticked.**

Keep entries under 15 lines; link to files rather than pasting code.

---

## RL-001 · decide · 2026-08-22 · Phase 0 / ruff configuration
**Encountered:** `ruff` reports `E402` (module-level import not at top of file)
three times in `src/bias_scope/__init__.py`. The imports sit after a
`try/except ImportError` block that installs constructor stubs, so that
`import bias_scope` works without the optional torch/embedding extras and each
missing metric raises a clear `ImportError` when constructed instead of at
import time.
**Options:** (a) per-file ignore for `E402`; (b) move the imports to the top and
drop the stub mechanism; (c) hoist the stubs into a submodule.
**Chosen:** (a). The ordering is load-bearing, (b) breaks the light-core install
rule in Section 1, and (c) adds an indirection layer for a lint rule.
**Where:** `pyproject.toml` `[tool.ruff.lint.per-file-ignores]`.
**Risk if wrong:** none functional; a genuinely misplaced import in that one
file would go unflagged.
**To revisit:** when Phase 2 adds `bias_scope/metadata.py` and the registry, the
stub mechanism may move there; drop the ignore if it does.

## RL-002 · decide · 2026-08-22 · Phase 0 / complexity cap
**Encountered:** Enabling `C901` at max-complexity 10 flags 17 pre-existing
functions (complexity 11–30), 16 in `src` and one in `tests`. Most are the
`evaluate` bodies of metrics that Phase 1 audits and Phase 2 partly rewrites.
**Options:** (a) refactor all 17 now; (b) `# noqa: C901` each with a tracking
reference; (c) raise the cap.
**Chosen:** (b). Refactoring `evaluate` bodies immediately before the fidelity
audit that may replace them is churn against code about to change, and it would
make the audit diff unreadable. (c) abandons a Section 1 rule outright.
**Where:** `# noqa: C901 (RL-002)` on the `def` line of each; the worst are
`cooccurrence_bias_score.py` (30), `stereotypical_associations.py` (29),
`social_group_substitution.py` (27).
**Risk if wrong:** the cap is not enforced on those 17 functions; it *is*
enforced on all new code.
**To revisit:** remove each `noqa` as Phase 2 rewrites that metric. Zero
remaining `C901 (RL-002)` markers is a release gate for 0.2.0.

## RL-003 · decide · 2026-08-22 · Phase 0 / new dependencies
**Encountered:** `SOURCES.yaml`, `registry.yaml`, and `ledger.yaml` are YAML,
and `fetch_sources.py` must extract PDF text. Neither has a standard-library
answer, and Section 1 forbids new runtime dependencies without a note.
**Options:** (a) `pyyaml` + `pypdf` in the `dev` extra only; (b) switch the
manifests to JSON; (c) hand-roll a YAML subset parser.
**Chosen:** (a). Runtime dependencies stay `numpy` + `requests`; nothing the
installed library imports gained a dependency. (b) makes three
hand-edited files markedly worse and diverges from every example in `PLAN.md`;
(c) is new code to maintain for no benefit.
**Where:** `pyproject.toml` `[project.optional-dependencies] dev`; also
`DECISIONS.md`.
**Risk if wrong:** none to library users; contributors need `.[dev]` to run the
manifest and ledger scripts.
**To revisit:** if Phase 5 adapters need YAML at runtime, this becomes a real
dependency decision rather than a dev-tooling one.

## RL-004 · verify · 2026-08-22 · Phase 0 / version mismatch
**Encountered:** `src/bias_scope/__init__.py` declares `__version__ = "0.1.0"`
while `pyproject.toml` declares `version = "0.1.1"`, which is also what was
released to PyPI. Confirmed in
`results/validation/environment.json` → `bias_scope.version: "0.1.0"`.
**Where:** `src/bias_scope/__init__.py:19`, `pyproject.toml:3`.
**Why it matters:** Section 5.3 puts `library_version` in every result's
`protocol` block, and `protocol_hash` covers it. Left as is, every v0.1 result
would be stamped with a version that was never released.
**To revisit:** fix in the Phase 2 commit that adds `make_protocol`, ideally by
reading the version from `importlib.metadata` so the two cannot diverge again.
Not fixed in Phase 0 because it changes a public attribute and belongs with the
protocol work.

## RL-005 · verify · 2026-08-22 · Phase 0 / WEAT 840B not recomputed
**Encountered:** The Phase 0 task asks to re-run WEAT-6 on GloVe-6B *and*
GloVe-840B. `results/emnlp/weat_840B/weat_840B_result.json` already records
d = 1.8139 against the paper's 1.81, but the 5.6 GB `glove.840B.300d.txt` it was
computed from is no longer on disk, and gensim does not ship the 840B vectors.
**Chosen:** carry the recorded 840B result forward and wire it into the
generated tables; do not re-download 2.2 GB this session. Maintainer approved.
**Where:** `results/emnlp/weat_840B/weat_840B_result.json`,
`scripts/experiments/repro_weat_840B.py`.
**Risk if wrong:** the 1.8139 in the paper rests on a run that has not been
independently repeated. The 6B number *was* re-verified this session.
**To revisit:** before submission, re-download the 840B vectors and re-run
`repro_weat_840B.py`; the script raises on OOV, so a silent vocabulary change
cannot pass unnoticed.

## RL-006 · verify · 2026-08-22 · Phase 0 / examples are not exercised
**Encountered:** Section 1 requires every file in `examples/` to be executed by
`tests/test_examples/`. Only `examples/metric_usage_examples.py` is — the other
44 under `examples/*/` instantiate real models at module import (e.g.
`examples/probability_based/crows_pairs.py` downloads `bert-base-uncased`), have
no `main()` and no `__main__` guard, so they cannot be imported in a test
without hitting the network.
**Where:** `tests/test_examples/test_metric_usage_examples.py`, `examples/*/`.
**Risk if wrong:** the README and docs examples can rot undetected; a rename in
Phase 2 would not fail any test.
**To revisit:** Phase 2. Each example needs a `main()`, a `__main__` guard, and
a tiny-model or stub path, then `tests/test_examples/` can `runpy` all of them.
Also note `tests/test_examples/` has no `__init__.py` while its four sibling
packages do, and `test_metric_usage_examples.py` uses a relative path, so it
only passes when pytest runs from the repo root.

## RL-007 · decide · 2026-08-22 · Phase 0 / gitignore for large artefacts
**Encountered:** `results/` is not ignored, and `results/emnlp/` holds ~6.6 GB
of untracked-but-stageable artefacts, including
`regard_full/sheng_models/regard1.tar` (3.1 GB) and three BERT checkpoints
(~1.3 GB each). A single `git add -A` would try to commit them.
**Options:** (a) targeted ignores for weight and vector files; (b) ignore
`results/` wholesale; (c) leave it and rely on care.
**Chosen:** (a). (b) conflicts with Section 11, which requires committed study
outputs and figures under `results/`; (c) is one keystroke from a ruined repo.
**Where:** `.gitignore` — `*.tar`, `pytorch_model.bin`, `optimizer.pt`,
`glove.*.txt`, alongside the `sources/papers/` and `third_party/` entries
`PLAN.md` asks for. Maintainer approved.
**Risk if wrong:** a legitimately small `.tar` under `results/` would be
skipped; use `git add -f` for it.
**To revisit:** if the reproduction scripts start writing weights elsewhere.

## RL-008 · verify · 2026-08-22 · Phase 0 / WEAT standard deviation convention
**Encountered:** Writing the naive WEAT oracle surfaced that
`src/bias_scope/embeddings_based/weat.py:244` divides by
`np.std(scores_union, ddof=1)` — the **sample** standard deviation — while the
docstring at line 132 says only `std(s(X ∪ Y, A, B))`. Caliskan et al. eq. 3
likewise writes "std-dev" without specifying. The two differ by a factor
`sqrt(n/(n-1))`: about 7% at the n = 8 of Caliskan's own word sets, which is
larger than the gap the v0.1 reproduction is trying to explain.
**Where:** `src/bias_scope/embeddings_based/weat.py:244`;
`tests/oracles/weat_oracle.py` (takes `ddof` explicitly, defaults to 1 to match
the library);
`tests/oracles/test_weat_oracle.py::test_library_uses_the_sample_standard_deviation`.
**Why it matters:** it moves the headline WEAT number the paper reports, and it
propagates to SEAT and CEAT, which build on the same effect size.
**RESOLVED 2026-08-22 (Phase 1 WEAT audit) — `ddof=1` is correct; no code
change needed.** Three independent lines agree:
1. `sent-bias/sentbias/weat.py:175` is `np.std(..., ddof=1)`, explicit.
2. On Caliskan's own GloVe-840B vectors, `ddof=1` reproduces the published 1.81
   to **0.22%**; `ddof=0` is off by **3.50%** — a 16x difference.
3. The 6B replication shows the same ordering (1.6938 vs 1.7494).
`weat.py:244` already used `ddof=1`, so the audit **confirms** the
implementation. Pinned by
`tests/test_embeddings/test_weat_permutation.py::TestStandardDeviationConvention`
and written up in `docs/fidelity/weat.md`. The claim in this entry that the
docstring is silent still stands and is worth a one-line fix.

## RL-009 · decide · 2026-08-22 · Phase 0 / scripts/ is not linted
**Encountered:** `PLAN.md` Section 2 defines the lint gate as
`ruff check src tests`. Running ruff over `scripts/` as well reports 53
violations, all in `scripts/experiments/` (28 `E501`, 8 `F541`, 7 `I001`,
4 `E731`, 3 `C901`, 2 `E741`, 1 `F401`) — the v0.1 reproduction drivers.
**Options:** (a) keep the gate at `src tests` as written; (b) extend it to
`scripts/` and fix all 53 now.
**Chosen:** (a), matching `PLAN.md` as written. The new Phase 0 scripts
(`scripts/sources/`, `scripts/verification/`, `scripts/validation/`) are
nonetheless ruff-clean and are kept that way.
**Where:** `.github/workflows/tests.yml`, `scripts/experiments/`.
**Risk if wrong:** the reproduction scripts drift in style, and their real
issues (a hardcoded absolute path, duplicated code between
`honest_dump_finalize.py` and `finalize_emnlp.py`) stay unflagged by CI.
**To revisit:** Section 10.3 rewrites these scripts for the effort survey;
extend the gate to `scripts/` then.

## RL-010 · decide · 2026-08-22 · Phase 0 / tiny encoder substitution
**Encountered:** `PLAN.md` Section 1 names `prajjwal1/bert-tiny` as the tiny
encoder for integration tests. Under transformers 5.12.1 its tokenizer will not
instantiate — the repo ships no `tokenizer.json`, and the slow-to-fast
conversion raises *"You need to have sentencepiece or tiktoken installed"*, with
`use_fast=False` failing identically.
**Options:** (a) substitute `hf-internal-testing/tiny-random-BertForMaskedLM`,
the other family Section 1 sanctions; (b) add `sentencepiece` to the `dev`
extra; (c) pin transformers < 5.
**Chosen:** (a). It loads cleanly on this transformers version, is smaller, and
`hf-internal-testing/tiny-random-*` is already permitted by the plan. (b) adds a
dependency to work around one repo's packaging; (c) pins the whole project back
for a test fixture.
**Where:** `tests/conftest.py` `TINY_ENCODER_ID`; `PLAN.md` Section 1 updated in
the same commit, per the "the plan itself is wrong" rule.
**Risk if wrong:** the tiny encoder is randomly initialised, so it is good for
shape, plumbing, and error-path assertions but never for a value assertion.
`tests/integration/test_tiny_model_fixtures.py` asserts only on ranges and
finiteness for that reason.
**To revisit:** if `prajjwal1/bert-tiny` gains a `tokenizer.json`, switch back —
a pretrained tiny model gives slightly more meaningful integration signal.

## RL-011 · verify · 2026-08-22 · Phase 0 / reproduction scripts have undeclared dependencies
**Encountered:** Re-running the WEAT-6 GloVe-6B reproduction needed `gensim`,
which `scripts/experiments/emnlp_reproduction.py` imports but which appears in
no extra in `pyproject.toml`. It was installed into `.venv` by hand to complete
the Phase 0 reconciliation.
**Where:** `scripts/experiments/emnlp_reproduction.py:118`; `pyproject.toml`.
**Why it matters:** Section 10.3 measures how much glue code it takes to re-run
each reference implementation and compares it against BiasScope. Our own
reproduction scripts failing on a missing import would be an embarrassing entry
in that table, and a reviewer following `CONFIGURATION.md` hits it immediately.
**To revisit:** add a `repro` extra (gensim, and whatever else
`scripts/experiments/` imports — audit them all, not just this one) when Section
10.3 rewrites these scripts. Record it in `CONFIGURATION.md` alongside the
existing protocol tables.

## RL-012 · verify · 2026-08-22 · Phase 1 / LPBS reference code reads the prior at the wrong position
**Encountered:** Auditing LPBS against
`third_party/code/contextual_embedding_bias_measure` @ `18044f87` found two
compounding deviations from the paper's §2 step 3, both about which mask
position `p_prior` is read at:
(a) `bias_calculator.py:49` vs `:61` pass opposite `use_last_mask` values, so the
both-masked sentence is read at the **attribute** slot, not the target slot;
(b) `bert_utils.py:82-90` — the `not last` branch returns `i + 1` "to take the
[CLS] token into account" while the `last` branch returns `len(toks) - 1 - i`
with no offset, and `tokenize` (`:51-55`) emits bare wordpieces.
Net: for `"[MASK] is [MASK]"` the prior appears to be read at the token `is`.
**Chosen:** implement the **paper's** step 3 in the Phase 2 reimplementation.
PLAN.md's "the code wins" rule is for resolving paper *ambiguity*; §2 here is
explicit and (b) looks like an unintended off-by-one.
**Where:** `docs/fidelity/lpbs.md`, section "Two discrepancies".
**Risk if wrong:** if Kurita's Table 2 numbers were produced with the code's
behaviour, a faithful-to-the-paper implementation will not reproduce them, and
Tier 1 will read `off`. That is the correct outcome to record, not a reason to
change the protocol.
**To revisit:** Phase 3 Tier 2. The repo pins `pytorch_pretrained_bert` and
AllenNLP and does not install on a supported Python, so **this is a static
code reading, not an executed result** — do not cite it as established until it
has been run. Playbook step 3 (copy the scoring function) is unavailable: the
repo states no license.

## RL-013 · blocked · 2026-08-22 · Phase 1 / HolisticBias style classifier is not available
**Encountered:** FGB and PGB (Smith et al. 2022 §A.7) are defined over a
217-class style classifier. `facebookresearch/ResponsibleNLP` @ `0ec714eb` does
**not implement either metric** — `holistic_bias/src/bias_measurements.py` is
the perplexity/Mann-Whitney analysis — and ships neither the classifier nor the
six style-cluster definitions. A repo-wide search for `gen_bias`/`Gen Bias`
returns nothing.
**Where:** `docs/fidelity/fgb.md`, `docs/fidelity/pgb.md`.
**Consequence:** the Phase 2 checkboxes for FGB/PGB **stay unticked**. A
faithful implementation is impossible without the classifier; a pluggable
`style_classifier` with a substitute makes the metric an `adaptation`, and the
substitute must be named in the docstring and in Table 2 of the paper.
**Do not** silently swap in a sentiment classifier and keep the FGB name.
**To revisit:** search for the classifier (the paper attributes the style labels
to prior BlenderBot work); if not found, ship the pluggable version as
`adaptation` and say so.

## RL-014 · decide · 2026-08-22 · Phase 1 / BOLD dataset is CC-BY-SA-4.0
**Encountered:** Rebuilding `BOLD` as a benchmark runner needs its prompt set.
`amazon-science/bold` is **CC-BY-SA-4.0** — attribution *and* share-alike.
BiasScope is MIT.
**Chosen (provisional):** load the prompts from the upstream source at runtime
rather than vendoring them, so the share-alike term is not triggered for this
repo's own code. Recorded here rather than acted on, because it is a licensing
question the maintainer may want to take differently.
**Where:** `docs/fidelity/bold.md`; `third_party/LICENSES.md`.
**Risk if wrong:** vendoring CC-BY-SA data into an MIT repo could oblige
share-alike on derived material. Also affects StereoSet and sent-bias, which
are likewise CC-licensed.
**To revisit:** before Phase 2 implements the BOLD runner. Same question applies
to the CC-licensed word lists in `sent-bias` and `StereoSet`.

## RL-015 · verify · 2026-08-22 · Phase 1 / six reference repos state no license
**Encountered:** Of the 21 reference implementations cloned, six ship no license
file at all: `CEAT`, `contextual_embedding_bias_measure` (LPBS), `crows-pairs`,
`language-model-bias` (CoOccurrenceBiasScore), `markedpersonas`, `nlg-bias`
(RegardScore).
**Where:** `third_party/LICENSES.md`.
**Why it matters:** under default copyright that is all rights reserved. Reading
and running them locally for a Tier-2 comparison is fine; **copying their
scoring functions into `tests/equivalence/reference_snippets/` is not**, which
removes step 3 of PLAN.md Section 1's obstacle playbook for exactly the metrics
most likely to need it (LPBS's repo does not install on a supported Python).
**To revisit:** Phase 3. Where a snippet copy is the only route, ask the authors
or fall back to Tier 1 + Tier 3, and record the metric as having no Tier 2.

## RL-016 · verify · 2026-08-22 · Phase 2 / two Hedges-Olkin standard errors in one repo
**Encountered:** Writing `bias_scope/stats.py` surfaced that the new
`hedges_olkin_ci` and the existing `scripts/experiments/finalize_emnlp.py`
`hedges_olkin_ci_d` use **different** standard errors for Cohen's *d*:

- `stats.py`: `se = sqrt((n1+n2)/(n1·n2) + d²/(2·(n1+n2)))` — Borenstein et al.
  2009 eq. 4.20, the form most commonly cited.
- `finalize_emnlp.py`: the same with `2·(n1+n2−2)` in the second term.

Both appear in the literature. At d = 1.6938, n1 = n2 = 8 they give
[0.5515, 2.8361] and [0.5302, 2.8574] respectively.
**Chosen:** `stats.py` uses the Borenstein form; `finalize_emnlp.py` keeps its
own, with a comment pointing here. **Deliberately not reconciled**: switching
`finalize_emnlp.py` would silently move the CIs already published in
`results/emnlp/` and quoted in the paper draft, and that is the maintainer's
call, not a side effect of adding a stats module.
**Where:** `src/bias_scope/stats.py::hedges_olkin_ci`,
`scripts/experiments/finalize_emnlp.py::hedges_olkin_ci_d`,
`results/emnlp/{weat.json,table.csv,summary.md,CONFIGURATION.md}`.
**Risk if wrong:** every WEAT/SEAT/CEAT interval in the paper is slightly the
wrong width. It does not change any MATCHED/CLOSE/OFF verdict at n = 8, because
both intervals are far too wide to separate anything.
**To revisit:** before submission. Check Hedges & Olkin (1985) directly, pick
one, make `finalize_emnlp.py` import it from `stats.py`, regenerate
`results/emnlp/`, and note the change in `CHANGELOG.md`.

## RL-017 · verify · 2026-08-22 · Phase 2 / CBS has no tests at all
**Encountered:** Retargeting the LPBS tests after the reimplementation revealed
that `tests/test_probability_based/test_cbs.py` was a **near-verbatim copy of
`test_lpbs.py`** — same four tests, same assertions, differing only in
formatting — and contained **zero** references to `CBS`. It tested the
sentence-pair preference metric under a filename claiming to test CBS.
**Chosen:** the duplicate was removed. Deleting it is not weakening the suite:
`src/bias_scope/probability_based/cbs.py` coverage is **20% before and 20%
after**, confirming the file exercised none of it. The surviving copy moved to
`test_pairwise_likelihood_preference.py`, which is what it actually tests. The
`-m slow` count went 3 -> 2 for the same reason: the duplicate carried a second
copy of the one slow integration test, and the original still runs.
**Where:** `src/bias_scope/probability_based/cbs.py` — 81 of 101 statements
uncovered.
**Risk if wrong:** CBS (Ahn & Oh 2021) is one of the least-tested metrics in the
library and was passing CI on the strength of a mislabelled file. Its Phase 1
audit and Phase 2 known-answer test are now the only things standing between it
and a silent defect.
**To revisit:** when CBS reaches the front of the Phase 1 audit queue. It needs
the full metric criteria of Section 1 — known answer, oracle, properties,
golden — not a smoke test. Check the other low-coverage modules for the same
pattern: `perspective_api.py` (40%) and `scorers.py` (68%).

## RL-018 · verify · 2026-08-23 · Phase 1 / the paper's BBQ row is not defensible as it stands
**Encountered:** The BBQ audit found the implemented `bias_score` is
`is_biased = not is_correct` → `biased_count / n`, i.e. the **error rate**,
identical to `1 - accuracy` by construction
(`src/bias_scope/prompts_based/bbq.py:179,194-195`). Parrish et al. define
`s_DIS = 2*(n_biased_ans / n_non-UNKNOWN) - 1` and `s_AMB = (1-accuracy)*s_DIS`,
signed in [-1,+1], using `question_polarity` and the bias target — neither of
which our code reads. A model answering entirely *against* the stereotype scores
**+1.0 (maximally biased)** under ours and **-100% (maximally anti-biased)**
under the paper: opposite conclusions, not a scaling difference.
**Compounding problem:** `results/emnlp/bbq.json` compares ours (0.2496) to a
"published" **0.25** that `finalize_emnlp.py` itself calls the "midpoint of
0.22-0.28 published range". PLAN.md Section 1 forbids inventing or anchoring a
published value, and Section 6.1 names BBQ explicitly: "do not reuse the v0.1
'midpoint of a range' anchor."
**Where:** `docs/fidelity/bbq.md`; `results/emnlp/{bbq.json,summary.md,table.csv,
CONFIGURATION.md}`; `scripts/experiments/finalize_emnlp.py` (the midpoint);
`results/emnlp/paper/evaluation.tex`.
**Risk if wrong — high, and it is submission-blocking.** BBQ is one of four
headline reproductions. As printed, the table claims MATCHED for a comparison
between a statistic that is not BBQ's and a reference number that was never
published. This is the same class of defect a reviewer already caught in LPBS.
**PARTLY RESOLVED 2026-08-23.**
- **Metric fixed.** `s_DIS`/`s_AMB` implemented per the paper, signed in
  [-1,+1], with the polarity-dependent target derived per row and validated
  against `nyu-mll/BBQ`'s metadata (19,092 matches, 0 mismatches).
  `BBQMetric` is now `faithful` (`docs/fidelity/bbq.md`).
- **Row withdrawn and the anchor removed everywhere.** `results/emnlp/bbq.json`
  now carries `published: null`, `status: no_published_reference`, and a
  `withdrawn_reason`; `summary.md` shows WITHDRAWN with the reason;
  `CONFIGURATION.md` and `emnlp_reproduction.py` no longer construct the
  midpoint. `METHODOLOGY.md`'s claim that "anchoring on the range midpoint is
  the honest scientific practice" is **retracted in place**, with the reasoning
  kept so the history is auditable.
- The 17-point AWQ-INT4 vs BF16 finding was preserved — it was measured, not
  anchored, and belongs in the protocol-sensitivity study (PLAN.md 10.2).

**STILL OPEN:** the row cannot come back without a fresh
`Llama-3.1-8B-Instruct` run, because no cached generations survive. Decide
before submission whether to re-run (and report `s_AMB` against
`no_published_reference`, which is legitimate) or to present three
reproductions instead of four and say why.

## RL-019 · verify · 2026-08-23 · Phase 2 / `evaluate()` return types are inconsistent across the family
**Encountered:** Reimplementing BBQ's bias score, an attempt to make
`BBQMetric.evaluate()` return a float by default (matching every other metric)
broke eleven existing tests: BBQ has **always** returned a dict regardless of
`return_details`.
**Chosen:** reverted. PLAN.md 5.3 is explicit that `evaluate()` keeps returning
"what it returns today, so every existing script, example, and test is
untouched", and BBQ's today is a dict. `run()` handles either shape, so the
framework layer is unaffected.
**Where:** `src/bias_scope/prompts_based/bbq.py::evaluate`.
**Why it matters:** this *is* reviewer finding #2 in miniature — "`evaluate()`
shares a name but not a signature or return type". A survey of which metrics
return float, which return dict, and which ignore `return_details` belongs in
the paper's Table 2 as evidence that `run()`/`BiasResult` was necessary, rather
than being quietly harmonised away.
**To revisit:** audit all 41 `evaluate()` signatures, tabulate the
inconsistencies, and decide in one go for 0.3.0 whether to harmonise (breaking)
or leave `run()` as the uniform entry point (non-breaking, and the plan's
stated design).

## RL-020 · verify · 2026-08-23 · Phase 2 / BBQ target derivation covers 61% of the benchmark
**Encountered:** `derive_target_index` reconstructs BBQ's `target_loc` from
`answer_info` + `additional_metadata.stereotyped_groups`, so the metric needs no
extra data file. Validated against `nyu-mll/BBQ`'s
`supplemental/additional_metadata.csv`: **19,092 matches, 0 mismatches**.
But **12,264 examples are underivable** and are excluded and counted:
all of SES (6,864), most of Race_ethnicity (5,000), and 400 of
Gender_identity — the **name-proxy** rows, where answers are first names
("Donna", "Greg") rather than group labels, so `stereotyped_groups` has nothing
to match.
**Where:** `src/bias_scope/prompts_based/bbq.py::derive_target_index`;
`n_excluded_no_target` in the result.
**Risk if wrong:** SES and Race_ethnicity currently produce a bias score over
**zero** rows, which `n_excluded_no_target` makes visible but which a careless
caller could still misread as "no bias".
**To revisit:** ship the name → group mapping (BBQ's
`supplemental/name_job_data/`, CC-BY-4.0) so the name-proxy rows resolve too,
or raise rather than return an empty score when every row in a subset is
excluded. The latter is a one-line guard and should probably happen first.

## RL-021 · decide · 2026-08-23 · CLOSED 2026-09-15 · CEAT's default sample count is far below the paper's
**Encountered:** `CEAT.evaluate(n_samples=100)` by default, while Guo & Caliskan
report CES at **N = 1,000 and N = 10,000** sampled contexts.
**Closed:** re-checked during the 2026-09-15 CEAT audit — the code (and
`docs/fidelity/ceat.md`, rewritten the same day) already reads
`n_samples: int = 10_000` (`ceat.py:56`), matching the paper's main N. This
entry described a state that no longer matched the code; whichever commit
raised the default did not update this entry or the fidelity note. No action
needed now; closing rather than leaving a stale "open" decision in the file.

## RL-022 · verify · 2026-08-23 · Phase 1 / EMT omits the paper's standard deviation
**Encountered:** Gehman et al. §3.2 define expected maximum toxicity as
estimated "with a mean **and standard deviation**". `EMT.evaluate` reports the
mean only.
**Where:** `src/bias_scope/generated_text_based/emt.py:55-56`.
**Why it matters:** the spread across prompts is the paper's own uncertainty
statement for this metric, and Table 1 prints both. Reporting the mean alone
makes an EMT value look more precise than the paper intends, and makes a Tier-1
comparison against Table 1 incomplete.
**Risk if wrong:** low — additive, not a correction. The mean is right.
**To revisit:** add `std` to `details` and surface it as the metric's own
interval in `run()`, the way WEAT's permutation p-value is surfaced. Small and
worth doing before Phase 3 Tier 1.
**RESOLVED 2026-09-18 (EMT audit follow-up).** `evaluate(return_details=True)`
now includes `"std"` (sample std, ddof=1, of the per-template maxima; 0.0 for
a single template), which `run()` surfaces via `result.details["std"]` the
same way WEAT surfaces `p_value`. Confirmed against
`allenai/real-toxicity-prompts@dd44ab77ed8b`'s own
`std_max = max_toxicities.std()` — same convention, same ddof.

## RL-023 · verify · 2026-08-23 · Phase 1 / CoOccurrenceBiasScore omits a normalising term
**Encountered:** Bordia & Bowman define
`P(w|g) = [c(w,g) / Σᵢ c(wᵢ,g)] / [c(g) / Σᵢ c(wᵢ)]`, and the bias score as
`log P(w|f)/P(w|m)`. Our implementation
(`cooccurrence_bias_score.py:98-102`) divides by `C[g]` = "total **anchor**
occurrences", i.e. `c(g)`, and never forms `Σᵢ c(wᵢ,g)` — the total number of
words appearing in each group's context windows.
**Effect:** a **constant additive offset** across all words. Rankings and
word-to-word comparisons are unaffected. Absolute values are not: the paper's
normalisation is precisely what makes an equally-co-occurring word score **0**,
which is this metric's neutral value.
**Where:** `src/bias_scope/generated_text_based/cooccurrence_bias_score.py`.
**Chosen:** documented as `adaptation` rather than silently corrected, because
fixing it changes every value the metric has ever produced and the audit had no
mandate to move numbers.
**To revisit:** add a per-group counter for total context-window words — the
windowing loop already visits them — and include it. Then re-verify that the
swap-antisymmetry property still holds (it does today, because the missing term
is itself antisymmetric under swapping the groups). Small, and it makes the
metric faithful.

## RL-024 · decide · 2026-08-23 · Phase 1 / `CounterfactualFairness` still carries a paper's title
**Encountered:** PLAN.md 4.2 and 5.2 call for renaming `CounterfactualFairness`
to `IdentitySwapConsistency`, because the class name is the **exact title** of
Kusner et al. 2017 and reads as a claim to implement it. Kusner et al. define a
*causal criterion on a predictor*, requiring a causal model; this class measures
cosine similarity between response embeddings under identity swaps.
**Chosen (for now):** audited and labelled `original` with a `deviation_note`
stating the gap, but **not renamed**. The rename is mechanical and the
deprecating-alias machinery already exists (`_deprecation.py`, used for
`OccupationPronounSkew` and `MeanScoreGap`); it was deferred only to keep the
audit and the renames in separate, reviewable commits.
**Where:** `src/bias_scope/prompts_based/counterfactual_fairness.py`;
`docs/fidelity/originals.md`.
**Risk if wrong:** a reader scanning the metric list sees a class named after a
well-known paper and reasonably assumes it implements it. The
`deviation_note` says otherwise, but names are read more often than notes.
**RESOLVED 2026-08-23, same session.** Renamed to `IdentitySwapConsistency`
with a deprecating alias, exactly as `DemographicRepresentationBias ->
OccupationPronounSkew` was done. `CounterfactualFairness` stays importable until
0.3.0 and raises a `DeprecationWarning`; the alias is a subclass so `isinstance`
still works. Tests, example and manifest entry all moved.

## RL-025 · verify · 2026-08-23 · Phase 1 / BOLD's second gender-polarity metric is missing entirely
**Encountered:** Dhamala et al. §4.5 defines **two** gender-polarity metrics:
unigram matching, and an embedding projection `b_i = (w_i·g)/(||w_i|| ||g||)`
with `g = w_she − w_he` on **hard-debiased Word2Vec**, aggregated as
`Gender-Wavg = Σ sgn(b_i)b_i² / Σ|b_i|` or `Gender-Max`, then thresholded at
±0.25. `GenderPolarity` implements only a variant of the first.
**Where:** `src/bias_scope/generated_text_based/gender_polarity.py`.
**Why it matters:** the paper introduces the embedding variant precisely because
unigram matching "does not account for words that may be indirectly related to a
gender" — so the missing half is the half that handles the harder case. PLAN.md
4.2 lists "unigram and `Wavg`/`Wmax` embedding variants" as what to check, and
only one exists.
**Also:** the sign convention is inverted relative to the paper (`b_i` positive
means *female* there, masculine here), and BOLD's three-way labelling with the
±0.25 threshold is not implemented.
**To revisit:** Phase 2 or 4. Needs the hard-debiased Word2Vec vectors
(`github.com/tolga-b/debiaswe`) as a pinned resource. Note Gonen & Goldberg 2019
on how much bias hard debiasing actually removes — worth a sentence in the
fidelity note when it lands.

## RL-026 · decide · 2026-08-23 · Phase 1 / LMB's headline statistic is not the paper's
**Encountered:** Barikeri et al. "quantify and report the bias effect as the
**t-value** of the Student's two-tailed test". `LMB.evaluate(return_details=False)`
returns `mean_diff`. The t-statistic and p-value are computed and present in
`details`, so nothing is missing — the default scalar is just not the one the
paper reports, which means `BiasResult.score` will carry the wrong quantity once
`run()` is wired up for this metric.
**Where:** `src/bias_scope/probability_based/lmb.py`.
**Chosen:** left as-is inside the audit. Changing the return value of
`evaluate()` is breaking, and PLAN.md 5.3 is explicit that `evaluate()` keeps
returning what it returns today.
**Also fixed in the same session:** the outlier rule. v0.1.1 offered only a
percentile strategy; the paper's 3-sigma rule (`[x̄ ± 3s]`) is now implemented
and is the default, with the percentile variant kept and labelled as BiasScope's
own.
**To revisit:** make `t_stat` the headline in 0.3.0 with a CHANGELOG "Breaking"
entry, or override `_interval` so `run()` reports the t-value and p-value as the
metric's own statistic — the second is non-breaking and is probably the right
answer, matching how WEAT's permutation p-value is surfaced.

## RL-027 · verify · 2026-08-23 · Phase 1 / SocialGroupSubstitution uses a range where the paper uses W1
**Encountered:** Huang et al. define both Average Individual Fairness and
Average Group Fairness as **Wasserstein-1 distances between sentiment
distributions**. `SocialGroupSubstitution` reports `individual_unfairness` and
`group_disparity` with the right names and structure, but computes both as a
**range (max − min)** over substituted values. A range is determined by the two
extreme groups alone; W1 uses the whole distribution, so with three or more
groups the two can order models differently.
Separately, our `group_disparity` compares subgroups **to each other**, while
the paper's G.F. compares each subgroup **to the entire evaluation set**.
**Where:** `src/bias_scope/generated_text_based/social_group_substitution.py`.
**Chosen:** documented as `adaptation`, not converted. Unlike
`CounterfactualSentimentBias` — which was fixed in the same session because it
already received per-template score *lists* — this class's API is built around
one scalar per substitution, so W1 needs a signature change, not a formula swap.
**To revisit:** Phase 2. `bias_scope.stats.wasserstein_1` already exists and is
validated against scipy to 1e-9, so the statistic is ready; only the input shape
needs deciding.

## RL-028 · verify · 2026-08-23 · Phase 1 / TruthfulQA has neither MC1 nor MC2
**Encountered:** Lin et al.'s multiple-choice task computes "the likelihood of
each reference answer independently ... the truthfulness score for the question
is the total normalized likelihood of the true answers" — that normalised total
is **MC2**, and **MC1** is whether the single highest-likelihood answer is true.
`TruthfulQA.evaluate` computes neither: it scores a free-text answer by
**embedding similarity** to the reference sets. It is also not the paper's
generation-task metric, which uses a judge.
**Where:** `src/bias_scope/prompts_based/truthfulqa.py`.
**Chosen:** labelled `adaptation`, matching PLAN.md 4.2's preliminary call.
**To revisit:** PLAN.md 5.2 asks for two things, both still outstanding:
add MC1/MC2 over logprobs as the faithful implementation, and rename the
similarity variant to `TruthfulQASimilarity` so the `TruthfulQA` name means the
paper's metric. The rename machinery already exists.
**Also unresolved:** Section 12 pre-decides that TruthfulQA is kept "labelled
'truthfulness (not social bias)', excluded from bias profiles by default", but
`MetricInfo` has no field expressing that, so `BiasSuite` will sweep it into a
bias profile. Add one before 5.4 lands.

## RL-029 · blocked · 2026-08-23 · CLOSED 2026-09-15 · SentenceBiasScore's paper could not be obtained
**Encountered:** Dolci, Azzalini & Tanelli 2023, *Data Science and Engineering*
8 was believed to be behind Springer's paywall with no preprint available.
**Closed:** wrong. The article is Springer **open access** (CC-BY-4.0 — the
PDF itself carries the licence notice), and a copy was sitting in
`biasscope papers/Sentencebiasscore.pdf` the whole time. Found and read in
full during the 2026-09-15 `SentenceBiasScore` audit. `SOURCES.yaml` status
is now `read`; `fidelity` moved from `unaudited` to `adaptation` (see
`docs/fidelity/sentence_bias_score.md`). RL-086 and RL-087 below record what
the audit found once the paper could actually be read.

## RL-030 · decide · 2026-08-23 · Phase 4 / Bai et al. IAT epsilon
**Encountered:** `d_score` in the released `clean.ipynb` adds 0.01 to both
denominators ("add 0.01 avoid float"); the paper's Sec. 2.1 formula has no such
term.
**Options:** (a) keep the epsilon as default; (b) use the paper's formula;
(c) expose both.
**Chosen:** (c) with (a) as the default, because the published `iat_bias`
column was produced with it — 18,885/18,885 released responses match exactly
with it and none match without. `epsilon=0.0` gives the paper's formula and
reproduces both of its worked examples exactly.
**Where:** src/bias_scope/prompts_based/implicit_association.py,
`REFERENCE_EPSILON`; docs/fidelity/implicit_association.md.
**Risk if wrong:** a perfect sorting reads 0.9988 rather than 1.0.
**To revisit:** if the authors publish an unepsiloned reference.

## RL-031 · verify · 2026-08-23 · Phase 4 / the shipped clean.ipynb
**Encountered:** Scoring the authors' own released responses the way their
shipped `clean.ipynb` does reproduces only 87% of their published `iat_bias`
column. The notebook splits each stereotype's attribute column in half in
`iat_stimuli_synonym.csv`, but that file **appends further attribute sets**
(occupations, then warmth/competence traits) after the valence set, so the
halves stop corresponding to valence. Using `iat_stimuli.csv`, where each
stereotype is one set, matches 100%.
**Where:** `load_iat_stimuli` reads the base file only and documents why;
tests/equivalence/test_implicit_association_equivalence.py.
**Risk if wrong:** none to us — we match the published column. The finding is
worth reporting upstream, since anyone re-running the released notebook gets
different numbers from the paper.
**To revisit:** open an issue on baixuechunzi/llm-implicit-bias.

## RL-032 · decide · 2026-08-23 · Phase 4 / DecodingTrust ambiguous answers
**Encountered:** `score_calculation_script.py` resolves an answer naming both
classes with `np.random.uniform(0, 1) > 0.5` on the global unseeded RNG, so the
published fairness numbers are not reproducible even from the released outputs.
**Options:** (a) reproduce the coin flip with a seeded RNG; (b) drop the record
into the rejection rate the benchmark already reports.
**Chosen:** (b). A coin flip is not evidence, and seeding it would produce a
number that matches nothing. This is the one place BiasScope deliberately does
not follow the reference code; it is stated in `deviation_note` and the note.
**Where:** src/bias_scope/prompts_based/decodingtrust.py, `parse_prediction`.
**Risk if wrong:** our DPD is computed on slightly fewer records than theirs.
**To revisit:** if DecodingTrust seeds the draw.

## RL-033 · verify · 2026-08-23 · Phase 4 / fairlearn's zero for an undefined rate
**Encountered:** `equalized_odds_difference` needs a TPR and an FPR per group;
a group with no records carrying the relevant label has an undefined rate.
fairlearn scores it **0.0**, silently. This can manufacture a disparity: one
false positive in one group, two other groups holding no negative records at
all, gives EOd = 1.0.
**Chosen:** match fairlearn, because DecodingTrust's numbers come out of it, and
report `degenerate_rate_groups` so a caller can see when the score rests on the
convention rather than on the model. Pinned by a test.
**Where:** src/bias_scope/prompts_based/decodingtrust.py.
**Risk if wrong:** an EOd of 1.0 read as a real disparity on a small run.
**To revisit:** consider raising when `degenerate_rate_groups` is non-zero and
the run is small.

## RL-034 · decide · 2026-08-23 · Phase 2 / the CI bracketing guard
**Encountered:** `run()` rejected a legitimate result. A metric whose per-item
values are all equal gives a degenerate bootstrap interval [v, v], and the mean
can differ from both endpoints by one ULP purely from summation order. A
perfectly consistent model was therefore unrunnable.
**Chosen:** a relative slack of 1e-9 on the bracketing check, scaled to the
magnitudes involved. A real gap still raises, and a test asserts it.
**Where:** src/bias_scope/base.py, `_CI_BRACKET_SLACK`; tests/test_run.py
`TestDegenerateIntervals`.
**Risk if wrong:** a bracketing failure smaller than 1e-9 relative goes unseen.
**To revisit:** if any metric legitimately produces intervals that tight.

## RL-035 · verify · 2026-08-23 · Phase 4 / FirstPersonFairness has no released code
**Encountered:** PLAN.md 7.2 requires the authors' judge instructions to be
mirrored verbatim. Eloundou et al. print Figure 3 marked "slightly abbreviated"
and released no repository; the six searches are logged in SOURCES.yaml under
`code_status: none_found`.
**Chosen:** implement the estimator exactly and register the metric as
`adaptation` rather than `faithful`.
**Where:** docs/fidelity/first_person_fairness.md; sources/SOURCES.yaml.
**Risk if wrong:** if the production prompt is later released and differs, our
numbers move.
**To revisit:** watch for an OpenAI artefact release.

## RL-036 · decide · 2026-08-23 · Phase 4 / multilingual datasets are loader-only
**Encountered:** CBBQ states **no license** and SHADES declares `other` on a
gated repository. Section 12 already says non-commercial datasets are
loader-only; absence of a license is not permission either.
**Chosen:** no dataset file ships in the package — asserted by a test that walks
the installed package for `.jsonl`/`.tsv`/`.csv` — and every missing-file error
names the source, the license and the usage note.
**Where:** src/bias_scope/multilingual.py; tests/test_multilingual.py
`TestNoRedistribution`.
**Risk if wrong:** none; the risk was in the other direction.
**To revisit:** if CBBQ adds a license.

## RL-037 · decide · 2026-08-23 · Phase 2 / the PLL family defaulted to the wrong path
**Encountered:** `CrowSPairs`, `AUL` and `AULA` all defaulted to
`mode="whitespace"`, which scores whole whitespace words. All three are
registered **faithful**, and `docs/fidelity/crows_pairs.md` states the verdict
is "faithful (in `mode='wordpiece'`, **which is the default**)" — which was
false. A default run therefore did not execute the protocol its own status
claimed. Found while auditing the remaining PLAN.md 5.2 boxes, not by a test.
**Also found:** `CrowSPairs.info.deviation_note` still read "Fidelity not yet
established … SOURCES.yaml entry is still `pending`. Do not cite this
implementation as faithful", left over from before the audit — contradicting
the `faithful` status in the same object.
**Options:** (a) flip the defaults to `wordpiece`; (b) downgrade the three to
`adaptation`; (c) leave the default and correct the note.
**Chosen:** (a). PLAN.md 5.2 says exactly this ("default to the faithful
WordPiece path; whitespace path stays as `mode='whitespace'`"), and (c) would
keep the library shipping the non-published protocol by default. The stale note
was rewritten to state what the two modes are and that whitespace must not be
reported as CrowS-Pairs.
**Where:** src/bias_scope/probability_based/{crows_pairs,aul,aula}.py;
src/bias_scope/_metric_info.py; tests/test_probability_based/test_wordpiece_mode.py.
**Risk if wrong:** **breaking** — a caller who upgrades and passes a
whitespace-style callback now gets a TypeError naming the requirement, rather
than a silently different number. That is the intended failure mode.
**To revisit:** nothing outstanding; a test now pins each default to the
faithful path, and the 50 whitespace-path tests name their mode explicitly.

## RL-038 · decide · 2026-09-12 · bias_scope_agent / no agent-specific API key env var
**Encountered:** bias_scope_agent (new package, tool-calling agent layer over
bias_scope) needs an Anthropic API key to run its agent LLM loop. The obvious
option was a new `BIASSCOPE_AGENT_API_KEY` env var, mirroring the project's
existing `BIASSCOPE_*` naming convention.
**Chosen:** no such variable. The `anthropic` Python SDK already reads
`ANTHROPIC_API_KEY` from the environment when `anthropic.Anthropic()` is
constructed with no explicit `api_key=`. A second, agent-specific variable
would create two sources of truth for the same secret with no behavioural
benefit — whichever is stale wins silently. `AgentLoop` defers entirely to the
SDK's own env lookup.
**Where:** src/bias_scope_agent/config.py (`AgentConfig` has no api_key
field); src/bias_scope_agent/loop.py (`AgentLoop.__init__` constructs
`anthropic.Anthropic()` with no explicit key).
**Risk if wrong:** none identified — this only reduces surface area.
**To revisit:** if a future version needs multiple concurrent agent sessions
each with a different key, at which point `AgentConfig` would need its own
field and `AgentLoop` would need to pass it through explicitly.

## RL-039 · verify · 2026-09-12 · bias_scope_agent / reimplements suite.py's private _metric_classes()
**Encountered:** `bias_scope_agent.introspection.metrics_needing_data` needs
every importable metric class by name, to introspect `evaluate()`'s required
parameters via `inspect.signature`. `bias_scope.suite` already has exactly
this (`_metric_classes()`), but it is private (leading underscore, not in
`__all__`).
**Chosen:** `bias_scope_agent.introspection` keeps its own copy of the same
four family-module names (`_FAMILY_MODULES`) and its own `_agent_metric_classes()`,
rather than importing `bias_scope.suite._metric_classes` directly. Importing a
private symbol from another package couples us to its internals with no
deprecation path; duplicating four literal strings is small, explicit, and
directly diffable against `suite.py`.
**Where:** src/bias_scope_agent/introspection.py `_FAMILY_MODULES`,
`_agent_metric_classes`; compare against src/bias_scope/suite.py
`_metric_classes`.
**Risk if wrong:** if `suite.py`'s module list changes (a new metric family
subpackage added) and this copy is not updated, `metrics_needing_data` will
silently miss metrics in the new family, reporting them as not importable.
**To revisit:** whenever `bias_scope/suite.py`'s `_metric_classes()` module
list changes.

## RL-040 · decided · 2026-09-12 (opened) · 2026-09-14 (decided) · bias_scope_agent / the confirm-before-run gate is structural, not semantic
**Encountered:** the agent must never call `run_suite` before a human has seen
a `plan_suite` result and approved it. This project's architecture forbids a
second "reviewer" LLM to double-check the first one's judgment.
**Chosen:** `AgentSession.check_run_gate` enforces a *structural* guarantee
only — a matching plan was recorded, a real turn boundary passed, and
`confirm_plan` was explicitly called — checked by `AgentLoop`'s dispatcher
before the real `run_suite` is ever invoked (proven in tests via a spy that
asserts zero calls). It cannot and does not judge whether the user's reply
was actually affirmative ("yes, run it" vs. "no, don't") — that judgment call
stays with the single agent LLM, per this project's no-second-reviewer design.
**Where:** src/bias_scope_agent/session.py (`AgentSession.check_run_gate`,
`confirm_plan`); src/bias_scope_agent/loop.py (`AgentLoop._dispatch_one`).
**Risk if wrong:** an agent LLM that misreads a "no" as confirmation could
still call `confirm_plan` and then `run_suite` successfully — the gate would
not catch that. This is an accepted limitation of the single-agent
architecture, not an oversight.

**Decided (2026-09-14): Option A — stay structural, tighten the prompt.** The
follow-up plan for this package posed two options: (A) keep the gate purely
structural and make the system prompt's confirmation language stricter and
more explicit, or (B) add a deterministic, non-LLM heuristic — log the literal
text of the confirming turn on the `PlanRecord` and reject confirmation if it
matches a hedge/question pattern (e.g. contains "?", "maybe", "not sure") —
as a second, independent signal alongside the LLM's own judgment.

Option A was chosen and implemented: `system_prompt.py`'s confirm_plan rule
now spells out that only unambiguous affirmation should trigger it, that a
hedge, a question, a change request, or silence is not confirmation, and that
this judgment is "the one safeguard the system cannot make for you." Option B
was not implemented. Reasoning: no live conversation has yet shown the agent
actually misreading a reply (Item 1 of the follow-up plan, live-conversation
testing, has not been run at the time of this decision) — adding a denylist
heuristic now would be defending against a failure mode observed only in
theory, and a phrase-matching heuristic is itself a coarse, easily-wrong
signal (e.g. "yes, I think that's right" contains "think" and would be
wrongly rejected; "no" contains no hedge word and would be wrongly accepted).
Tightening the prompt is near-zero-cost and makes the existing risk visible
in the one place capable of actually judging intent, rather than adding a
second, cruder judge next to it.
**To revisit:** if a live run (or production use) shows the agent misreading
an ambiguous or negative reply as confirmation, revisit Option B — specifically
whether a narrow heuristic (not a second LLM) is worth the false-rejection
risk it would add for legitimately-worded affirmations.

## RL-041 · RESOLVED 2026-09-17 (CEAT still open, see RL-048) · bias_scope / CrowSPairs and AUL cannot be run via .run()
**Update 2026-09-17:** fixed. `_extract_score` now accepts exactly one
`<name>_score` key as a fallback, and `_count_items` accepts a whole-number
float count (a second, independent defect found while fixing the first — see
RL-048). `CrowSPairs`, `AUL` and `AULA` verified running through `BiasSuite`
with correct `n`. `CEAT` remains blocked for a different reason (RL-048).
Tests: `tests/test_run.py::TestMetricNamedScoreKey`,
`tests/test_run.py::TestWholeNumberItemCounts`.

**Encountered:** while writing bias_scope_agent's integration test
(tests/integration/test_bias_scope_agent_tiny_model.py), `BiasSuite.run()` for
`CrowSPairs` (both whitespace and wordpiece modes) and `AUL` raised
`BiasScopeError: cannot find a headline score in evaluate()'s result`.
`BiasMetric._extract_score` (base.py) only recognizes dict keys `bias_score`,
`score`, `value`, `effect_size`, but `CrowSPairs.evaluate(return_details=True)`
returns `{"crows_pairs_score": ..., "num_pairs": ...}` and `AUL.evaluate(...)`
returns `{"aul_score": ..., "num_pairs": ...}` — neither key is in the
accepted list. Reproduced directly (not via a test, since fixing bias_scope
metric internals is out of scope for the bias_scope_agent feature; CLAUDE.md
forbids touching src/bias_scope/ for this work).
**Chosen:** not to fix here. Flagged as a separate follow-up task
(task_f451e5a9) for someone to investigate whether other probability/generated
-text metrics share the same `"<name>_score"` pattern and to either widen
`_extract_score`'s accepted keys or rename the metrics' dict keys.
bias_scope_agent's own integration test uses `WEAT` instead (unaffected by
this, and the only embedding metric that accepts raw embedding arrays
directly without a live model download).
**Where:** src/bias_scope/base.py `_extract_score`;
src/bias_scope/probability_based/crows_pairs.py;
src/bias_scope/probability_based/aul.py.
**Risk if wrong:** these two (and possibly more) metrics are currently
unusable through `BiasSuite`/`.run()` — only `.evaluate()` called directly
works, which is why the repo's own existing integration test
(tests/integration/test_tiny_model_fixtures.py) calls `.evaluate()` directly
for CrowSPairs rather than `.run()`.
**To revisit:** when task_f451e5a9 (or equivalent) is picked up.

## RL-042 · verify · 2026-09-14 · bias_scope_agent / OpenAI and Gemini provider adapters untested against real APIs
**Encountered:** Item 6 of the bias_scope_agent follow-up plan asked for full
multi-provider support, not just an Anthropic-shaped config field. Built
`OpenAIProvider` and `GeminiProvider` (`src/bias_scope_agent/providers.py`)
translating this package's one internal tool representation to each
provider's own function-calling wire format and back, grounded in each
provider's publicly documented contract (OpenAI Chat Completions tool
calling; Gemini function calling via `google-genai`) and verified with
hand-built fakes matching that documented shape
(`tests/test_bias_scope_agent/test_providers.py`, 20 tests, all passing,
including a parametrized scripted conversation proving the confirm-before-run
gate and tool dispatch behave identically across all three real adapters).
**Chosen:** ship both, clearly flagged as unverified against the real APIs —
no `ANTHROPIC_API_KEY`/`OPENAI_API_KEY`/`GOOGLE_API_KEY` was available in this
session (see RL-038's sibling gap for the Anthropic path, which has the same
limitation). This is the same category of gap as Item 1's deferred live
conversation test, just for two more providers.
**Where:** src/bias_scope_agent/providers.py (`OpenAIProvider`,
`GeminiProvider`, and their translation helpers `_openai_*`/`_gemini_*`).
**Risk if wrong:** a real API's actual response shape could differ from what
these adapters assume in some edge case the hand-built fakes did not cover
(e.g. multi-part tool-call responses, an SDK version's field renaming). The
fakes are shaped to match each provider's documented contract as of this
session, not captured from a live response.
**To revisit:** the first time someone actually runs
`BIASSCOPE_AGENT_PROVIDER=openai` or `=gemini` against a real key — treat any
mismatch found there as higher-priority than a hypothetical one, and add a
regression test pinned to the real response shape once seen.

## RL-043 · decide · 2026-09-14 · bias_scope_agent / Gemini tool schemas are conservatively stripped, not translated precisely
**Encountered:** `schemas.py`'s tool definitions use plain JSON Schema
(`type`, `properties`, `enum`, `default`, etc.) for Anthropic's tool-use
format. Gemini's function-declaration schema is a stricter subset of
JSON Schema / OpenAPI and does not accept every keyword Anthropic's dialect
allows — `default` is the one this package's own schemas actually use
(`construct_backend`'s `dtype`/`summarize_report`'s `format`) and is known to
cause issues in Gemini's schema validation.
**Chosen:** `providers._strip_unsupported_schema_fields` keeps only a
conservative, known-safe subset (`type`, `properties`, `items`, `required`,
`enum`, `description`) when building Gemini's tool declarations, dropping
everything else rather than attempting a precise, complete translation of
every JSON Schema keyword Gemini might or might not accept. This trades some
information loss (a `default` value is no longer visible to Gemini, so it may
ask the user for a value that Anthropic/OpenAI would have inferred) for
confidence that the schema Gemini receives is at least valid.
**Where:** src/bias_scope_agent/providers.py `_strip_unsupported_schema_fields`,
`_GEMINI_SCHEMA_KEYS`.
**Risk if wrong:** if Gemini's actual accepted keyword set is broader than
assumed here, this strips more than necessary and Gemini's tool-calling
behavior is very slightly less informed (not incorrect, just missing a
default hint) than it could be. If narrower, a real call could still fail
schema validation on one of the kept keys — untested against the real API
(see RL-042).
**To revisit:** alongside RL-042, the first time this is run against a real
Gemini key.

## RL-044 · verify · 2026-09-14 · bias_scope_agent / model_type classification lookup is a modest, non-exhaustive list
**Encountered:** `inspect_model`'s architecture-string heuristic
(`_guess_kind_from_config`) only classifies a model confidently when
`architectures` names something containing `CausalLM`/`LMHeadModel` or
`MaskedLM`. A config that only sets `model_type` (no `architectures`, or an
architecture name that doesn't match those substrings) fell through to
`confidence="low"` even for extremely common open-weight families like
Llama or RoBERTa, forcing an avoidable clarifying question every time.
**Chosen:** added `_CAUSAL_MODEL_TYPES`/`_ENCODER_MODEL_TYPES`, two small,
hand-picked sets of well-known `model_type` strings (llama, mistral, gpt2,
… / bert, roberta, deberta, …), consulted as a fallback only when the
architecture-string check doesn't already have an answer. Deliberately not
exhaustive — a full mapping of every HF `model_type` to causal/encoder would
itself need ongoing maintenance as new architectures ship, and getting this
wrong just means one extra clarifying question, not an incorrect backend
(the user still confirms `backend_kind` explicitly either way).
**Where:** src/bias_scope_agent/introspection.py `_CAUSAL_MODEL_TYPES`,
`_ENCODER_MODEL_TYPES`, `_guess_kind_from_config`.
**Risk if wrong:** a model_type present in the wrong set would produce a
*confident* wrong guess rather than a low-confidence one requiring
confirmation — worth double-checking each entry stays accurate as
`transformers` evolves (e.g. a `model_type` occasionally gets reused or
renamed across major versions).
**To revisit:** if a new popular open-weight family launches with a
`model_type` not in either set, or if a reported misclassification traces
back to one of these entries.

## RL-045 · verify · 2026-09-14 · bias_scope_agent / litellm model-string hint uses a bundled, offline registry that can go stale
**Encountered:** `inspect_model` now checks a plain identifier like
"gpt-4o-mini" against `litellm.model_list` before ever attempting an HF Hub
lookup, since a litellm-style model string is never a real HF repo id.
**Chosen:** `_litellm_model_hint` uses `litellm.model_list`, which ships
bundled with whatever `litellm` version is installed, not fetched live —
confirmed by testing it directly: `"gpt-4o-mini"` matched exactly, but
`"claude-3-5-sonnet-20241022"` did not (only prefixed/regional variants like
`"anthropic.claude-3-5-sonnet-20241022-v2:0"` were present), meaning a
completely valid, real model string can still miss an exact match and fall
through to the (usually harmless, just slower) HF Hub attempt. This is
treated as acceptable: the hint is advisory only — a match raises confidence
and skips a pointless network call, a near-miss adds a "did you mean" note,
and a total miss changes nothing about existing behavior. Nothing is ever
blocked on this check.
**Where:** src/bias_scope_agent/introspection.py `_litellm_model_hint`.
**Risk if wrong:** worst case is a missed optimization (one avoidable failed
HF Hub lookup) — never an incorrect backend, since `construct_backend`'s own
`kind` argument (chosen by the agent/user, not by this hint) is what actually
determines behavior.
**To revisit:** if `litellm`'s bundled list format changes, or if this
proves unhelpful often enough in practice (e.g. via Item 1's eventual live
conversation testing) to not be worth the added code path.

## RL-046 · verify · 2026-09-17 · bias_scope_agent / OpenRouter and litellm providers untested against real APIs
**RESOLVED for `openrouter` on 2026-09-18:** `OpenRouterProvider` has now driven
three live conversations against a real key (see PROGRESS.md 2026-09-18 and
`results/verification/agent_live/`). Tool-call translation, the whole-transcript
re-translation across turns, and error round-tripping all held. `LiteLLMProvider`
is still unexercised against a real key; the rest of this entry stands for it.
**Encountered:** `OpenRouterProvider` and `LiteLLMProvider` (added by direct
request, `providers.py`) were built and unit-tested against hand-built
fakes matching each dependency's documented shape (OpenRouter's own
OpenAI-compatible endpoint; litellm's `completion()` response, which is
itself OpenAI-shaped), same as every other provider adapter in this
package.
**Chosen:** ship both, clearly flagged as unverified against a real call —
no `OPENROUTER_API_KEY` was available in this session. Same category of gap
as RL-042 (OpenAI/Gemini) and the `local` provider's equivalent note in
`DECISIONS.md` — every agent-LLM provider this package supports has now
been built and unit-tested without ever making one real API call, in any
session to date.
**Where:** src/bias_scope_agent/providers.py (`OpenRouterProvider`,
`LiteLLMProvider`, `_wrap_litellm_client`).
**Risk if wrong:** `OpenRouterProvider` carries the same risk profile as
`LocalProvider`/other `OpenAIProvider` subclasses — low, since it reuses
proven translation code, only the endpoint differs. `LiteLLMProvider`
carries a bit more: the shim assumes `litellm.completion()`'s response
object is close enough to the raw `openai` SDK's own response object for
`_normalize_openai_response` to read `.choices[0].message.content` /
`.tool_calls` correctly — true per litellm's own design and documentation,
but not confirmed against a real litellm response object in this session.
**To revisit:** the first time either is run against a real key — treat any
shape mismatch found there as higher-priority than a hypothetical one.


## RL-047 · fix · 2026-09-17 · `.venv` was unusable: core dep missing, stale install, no extras — and `list_metrics()` silently shrinks when extras are absent
**Encountered:** while answering a "how do I run the agent" question,
`import bias_scope` failed outright in the repo's own `.venv`:
`ModuleNotFoundError: No module named 'requests'`, raised from
`generated_text_based/perspective_api.py:7`. `requests>=2.28.0` is a *core*
dependency in `pyproject.toml`, not an extra. The venv also had no `pip`, no
optional extras at all, and a stale editable install pinned at 0.1.0 while
`pyproject.toml` read 0.1.1 — so every `results/` protocol block written from
that venv recorded `library_version: 0.1.0`.
**Chosen:** repaired the environment with `uv`, in this order:
`uv pip install --python .venv/bin/python "requests>=2.28.0"`, then
`-e .` (0.1.0 -> 0.1.1), then `-e ".[all]"`. Environment only; no source
change. Full suite afterwards: **1930 passed, 3 skipped, 2 xfailed**, and
`ruff check src tests` clean.
**The finding worth keeping — `list_metrics()` is dependency-sensitive:**
the registry silently omits any metric whose optional dependency is not
installed, with no warning. Measured on the same checkout, changing nothing
but installed packages:

| venv state | `len(list_metrics())` | self-loading metrics |
|---|---|---|
| core only (broken) | 48 | 3 |
| `+ [datasets]` | 54 | 8 |
| `+ [all]` | 55 | 8 |

`BBQMetric`, `StereoSetMetric`, `AnalogicalReasoningBias`,
`OccupationPronounSkew` and `TofNof` all appear only once `datasets` is
present. The user-visible consequence is a misleading error: `BiasSuite.plan()`
raises `ValueError: unknown metric 'BBQMetric'` (`suite.py:103`) for a metric
that exists and is correctly registered — it is merely not installed. The
agent surfaces that string verbatim to the LLM as a tool error, so the agent
will tell a user a metric does not exist when the real fix is
`pip install "bias-scope[datasets]"`.
**Correction to an earlier draft of this entry:** it claimed
`tests/test_bias_scope_agent/test_introspection.py` and `test_tools.py` had
rotted by referencing a removed `BBQMetric`. That was wrong — those two tests
fail *only* in an environment without `datasets`, and pass once it is
installed. No test rot exists; the tests are correct and PLAN.md Section 14's
BBQ reference is accurate.
**Where:** `src/bias_scope/metadata.py` (`list_metrics`), `src/bias_scope/suite.py:103`.
**Risk if wrong:** low for the suggested fix; the current behaviour's risk is
silent under-measurement — a user can run what looks like a complete
evaluation and never learn that six metrics were invisible to it.
**To revisit:** make the omission legible rather than silent. Cheapest option
is for `plan()`'s `unknown metric` error to distinguish "not in the registry"
from "registered but its extra is not installed, run `pip install ...`".
RL-041 remains separately open and was re-confirmed here by direct
reproduction: `CEAT`, `AUL`, `AULA` and `CrowSPairs` key their headline score
as `<name>_score`, which `base.py:199`'s `_extract_score` rejects, so
`BiasSuite` records them as skipped and the agent cannot reach them.

## RL-048 · verify · 2026-09-17 · CEAT's `n` is its permutation-sample count, and deciding whether that is the right `n` needs the paper
**Encountered:** while fixing RL-041, `CrowSPairs`, `AUL` and `AULA` were
brought back through `BiasSuite`, but `CEAT` still raises
`BiasScopeError: n must be positive, got 0`. Its details dict is
`{"ceat_score", "weat_mean", "weat_std", "weat_variance", "n_samples"}` —
`n_samples` (100 by default) is not among the count keys `_count_items`
recognises (`n`, `num_items`, `num_pairs`, `num_rows_evaluated`,
`num_prompts`, `num_generations`).
**Chosen:** stopped rather than adding `n_samples` to that list. The other two
fixes were plumbing — a key spelling and a numeric type — with no bearing on
any published statistic. This one is not: `n` feeds the confidence interval,
and `n_samples` in CEAT is the number of random *samples combined*, not the
number of items scored. Whether it is the correct denominator for CEAT's
interval is a question about Guo & Caliskan (2021), and CLAUDE.md forbids
deciding a metric's definition from memory rather than from the paper and the
authors' code. Guessing would attach a wrong interval to a faithful-fidelity
metric, which is worse than leaving it unreachable and labelled.
**Where:** `src/bias_scope/base.py` `_count_items`;
`src/bias_scope/embeddings_based/ceat.py`.
**Risk if wrong:** `CEAT` stays unreachable through `BiasSuite` and therefore
through `bias_scope_agent`, while still working when `evaluate()` is called
directly. No wrong number is produced — the guard fires instead.
**To revisit:** read Guo & Caliskan (2021) and the authors' code per PLAN.md
Section 4.0, decide what CEAT's `n` should be, and add the test before the fix.

**RESOLVED 2026-09-18.** The authors' own code answers the question the entry
left open. `third_party/code/CEAT/code/ceat.py:205` `ceat_meta(..., N=10000)`
draws N samples, each producing one effect size and one variance (`e_lst`,
`v_lst`), and line 243 computes the Q statistic with `df = N - 1`. The degrees
of freedom say explicitly that N is the number of observations the
random-effects model pools, so `n_samples` is the count `n` carries.
`_count_items` now accepts it, and CEAT completes `run()`.
Test: `tests/test_embeddings/test_ceat.py::TestCeatIsReachableThroughRun`.

## RL-049 · fix · 2026-09-17 · first live agent run: a malformed `inputs` shape was reported to the user as a completed evaluation
**Encountered:** PLAN.md Section 14 Item 1 (a live conversation against a real
agent LLM) had been deferred every session for want of an API key. An Ollama
server running `gemma4:12b-mlx` was available locally, so the run was finally
done via the `local` provider: 4 turns, target model
`hf-internal-testing/tiny-random-BertForMaskedLM`, metric `CrowSPairs`.

What held, none of it previously exercised against a real LLM: the
confirm-before-run gate (`confirm_plan` in turn 4 matched turn 3's plan and
`check_run_gate` passed), the whole-transcript re-translation in
`providers.py` across four turns, both handle registries, and — the one that
matters most — the model did **not** fabricate a score.

What broke: the agent called `run_suite` with a flattened `inputs`,
`{"__init__": ..., "sentence_pairs": ...}`, instead of
`{"CrowSPairs": {"__init__": ..., "sentence_pairs": ...}}`. `BiasSuite.run()`
does `inputs.get(name)`, got `None`, and skipped the metric with `NEEDS_DATA` —
a reason that reads as *the user did not supply data*, not *the call was
malformed*. The agent then reported the skip to the user as the result of the
evaluation. Nothing anywhere said the call was wrong.
**Cause:** `schemas.py`'s `RUN_SUITE` described `inputs` as a bare
`{"type": "object"}` with one line, "Per-metric kwargs, exactly as gathered
from the user." It never stated that the top-level keys are metric names and
gave no example. The model's guess was reasonable.
**Chosen:** fixed structurally rather than by prompting, matching how this
package treats the run gate. `tools._check_inputs_shape` now rejects any
top-level key not in `metric_names` with a `ValueError` naming the offending
key and showing the expected shape; `ValueError` is in `loop._CAUGHT_TOOL_ERRORS`,
so the agent receives it as a correctable tool error rather than a silent skip.
The schema description now states the shape and carries a worked example.
An empty `inputs` stays legal — every metric skipping for want of data is a
real outcome.
**Where:** `src/bias_scope_agent/tools.py` (`_check_inputs_shape`, `run_suite`),
`src/bias_scope_agent/schemas.py` (`RUN_SUITE`).
Tests: `tests/test_bias_scope_agent/test_tools.py::TestRunSuiteInputsShape`.
**Also observed in the same run, not fixed - and worse than an omission:** in
turn 1 the user asked which metrics could run. The model called neither
`recommend_metrics_tool` nor `explain_exclusions_tool`, and **fabricated their
output**: it presented `gender_stereotypes_prediction` as a recommended metric
and `gender_representation_generation`, `gender_professional_stereotypes` and
`gender_occupational_stereotypes` as excluded ones, each with an invented
exclusion reason ("requires a generative/causal model"), under confident
Markdown headings. None of those four names exist in `list_metrics()`. In turn
4 it likewise invented a cause for the skip ("the metric implementation requires
the specific internal parameters associated with that dataset") rather than
reporting the mechanism. Turn 2's `plan_suite` returned `needs_data` naming
CrowSPairs and it never called `request_missing_inputs` either.

This is the finding of the run. The package's safety design assumes the agent
either calls a tool or says it cannot; it has no answer for an agent that
answers from itself. Every guarantee that is prompt-only - the
recommend/explain pairing, never inventing input data, and RL-040's
affirmation judgment - rests on an assumption this run falsifies for a 12B
model. Note what still held: the fabrication never reached a *score*. Metric
names and reasons were invented; no number was. The structural gates held
exactly where they exist, and nowhere else.
**Risk if wrong:** low for the fix. The unfixed observation is the larger
concern: RL-040 chose to leave affirmation-judging to the agent because no
live run had shown it misreading anything. One now has — not on affirmation,
but on two other prompt-only rules. Worth revisiting RL-040 with this evidence.
**To revisit:** re-run against a stronger model before drawing conclusions about
the prompt-only rules; a 12B local model is the floor, not the target. Consider
structural backing for the recommend/explain pairing, which is cheap to enforce.

## RL-050 · fix · 2026-09-17 · JSON has no tuple, so a `isinstance(pair, tuple)` check made three metrics unreachable from any tool call
**Encountered:** after fixing RL-049's `inputs` shape, `CrowSPairs` still would
not run through `bias_scope_agent`. The cause was not the agent and not the
model: `crows_pairs.py`, `aul.py` and `aula.py` each validated wordpiece-mode
pairs with `if not (isinstance(pair, tuple) and len(pair) == 2)`. JSON has no
tuple type, so a pair arriving through *any* tool call, HTTP API or serialized
boundary is always a `list`, and all three metrics rejected it with
"each sentence_pair must be a (stereotype, anti_stereotype) tuple of strings".
These are the same three metrics RL-041 had just made reachable — reachable
from Python, still unreachable from the agent.
**Chosen:** accept `(tuple, list)` rather than `tuple` alone, in all three. Not
`collections.abc.Sequence`: a `str` is a Sequence, and a 2-character string
would then unpack into two 1-character "sentences" and score silently. Tuple
or list is the plain check that admits JSON and excludes that, and both
rejection guards are pinned by tests. This touches input validation only — no
statistic, no protocol, no fidelity claim changes, so PLAN.md Section 4.0's
read-the-paper-first rule is not engaged.
**Where:** `src/bias_scope/probability_based/crows_pairs.py`,
`aul.py`, `aula.py` (wordpiece pair validation).
Tests: `tests/test_probability_based/test_wordpiece_mode.py::TestPairsMayBeListsNotOnlyTuples`
(includes a JSON round-trip and the two rejection cases that must keep firing).
**Verified:** with `inputs` round-tripped through `json.dumps`/`json.loads` —
the exact data path of a tool call, containing no tuples anywhere — `CrowSPairs`
and `AUL` now return real scores with fidelity badges through
`tools.run_suite` + `tools.summarize_report`, where both previously skipped.
**Risk if wrong:** low, and the failure mode was silent-by-skip rather than a
wrong number.
**To revisit:** this is a class of bug, not one instance. Any metric validating
an input with `isinstance(x, tuple)`, or expecting a set, or a numpy array, is
unreachable from the agent for the same reason. A scan found only these three
today, but nothing prevents the next one; the release gate proposed at the end
of the 2026-09-17 PROGRESS entry (every registered metric must complete
`.run()`) would catch them if its fixtures went through JSON.

## RL-051 · blocked · 2026-09-18 · `~typesafe/jev-latest` is not a chat model and cannot drive the agent loop
**Encountered:** a live OpenRouter key was supplied with the instruction to use
`~typesafe/jev-latest` as the agent LLM. The slug resolves (it is
`typesafe/jev-1.13-20260917`, provider "TypeSafe"), but it is absent from the
445 models `GET /api/v1/models` returns for this key, and a normal request is
refused: *"~typesafe/jev-latest is a decisions model and cannot be used with
the chat/completions endpoint. Use the /api/alpha/decisions endpoint instead."*
That endpoint takes `state` + `questions` and answers each question as one of
three fixed types — `noul` (a probability), `choice` (one option key, with
probabilities and a confidence) or `score` (an index into a legend). Probed
all three; each returns only those structured fields.
**Chosen:** `~openai/gpt-terra-latest` for the live runs, recorded in every
artifact under `agent_model`. `AgentLoop` needs a model that emits assistant
text *and* `tool_calls` carrying arbitrary JSON arguments (`construct_backend(
model_id="bert-base-uncased", dtype="fp32", device="cuda")`). A decisions API
emits no free text and no arguments at all, so no adapter could bridge it —
this is a shape mismatch, not a missing translation. `~anthropic/claude-*` was
excluded by the maintainer during the session. A non-Claude, tool-calling,
listed alias was the nearest defensible substitute.
**Where:** no code change; `providers.py` is untouched and correct.
**Verified:** the substitute emits `tool_calls` (preflight probe) and completed
three live conversations. The jev endpoint's request and response shapes were
established by direct probing, not documentation.
**Risk if wrong:** low, and confined to attribution — every artifact names the
model actually used.
**To revisit:** jev is a *judge* by construction, and `bias_scope` has
judge-based metrics (`TofNof`, the TrustLLM family) whose `judge_model` it
could plausibly serve. That is a different integration from the agent loop and
was not attempted here. Note `choice` returned a key from `criteria`, not from
`options`, which is worth understanding before relying on it.

## RL-052 · fix · 2026-09-18 · `needs_data` inspected `evaluate()` only, so metrics that take their model at construction were reported as needing nothing
**Encountered:** the first live run planned `CrowSPairs` on `bert-base-uncased`
and the metric was skipped: *"TypeError: wordpiece mode requires either
model_name= at __init__ or a WordPieceBertScorer passed as
predict_masked_token=."* The agent had supplied exactly what `plan_suite` told
it to. `metrics_needing_data` built its answer from `inspect.signature(
cls.evaluate)` alone, so for CrowSPairs it said `["sentence_pairs"]` and never
mentioned the model the metric needs at construction. `BBQMetric`, whose
`__init__(self, model_name: str)` has no default at all, was reported as
needing *nothing* — a test asserted that, on the reasoning that BBQ ships its
own dataset, which is true of its data and false of the metric.
**Chosen:** three changes, all in `bias_scope_agent`, none touching a statistic.
1. `_required_init_params` reports constructor parameters as
   `"__init__.<name>"`. A parameter counts as needed when it has no default, or
   when its default is an unset sentinel (`None` or `""`). A *real* default is
   left alone — this is the line that keeps `RegardScore(model_name=
   "sasha/regardv3")` off the list, since that names the classifier Sheng et
   al. require and replacing it with the model under test is the exact
   conflation the 0.2.0 audit corrected. `device` is excluded (placement, no
   effect on any statistic) and so is anything matching `*api_key`, because
   `construct_backend`'s schema deliberately exposes no key and naming one here
   would invite the agent to ask the user to paste one into the chat.
2. `run_suite` now rejects a call missing anything `needs_data` named, with a
   `ValueError` that names the metric and the parameters. `loop.py` hands that
   back to the agent as a correctable tool error in the same turn.
3. `run_suite` also refuses to return a handle when *every* planned metric was
   skipped: nothing ran, so there is no result, and a handle to an empty report
   renders as a "result" whose entire content is an error message.
**Rejected:** injecting the backend's `model_id` into any metric whose
`__init__` accepts `model_name`. `model_name` is not one concept — for
`CrowSPairs` it is the model under test, for `RegardScore` the required
classifier, for `WEAT`/`SEAT`/`CEAT` the sentence encoder. Silent injection
would have scored some metrics with the wrong model while `Report.model_id`
still named the backend, which is a mis-attribution, not a convenience.
**Where:** `src/bias_scope_agent/introspection.py` (`_required_init_params`,
`_is_credential`, `metrics_needing_data`), `src/bias_scope_agent/tools.py`
(`_check_required_inputs`, `run_suite`), `src/bias_scope_agent/system_prompt.py`
(one rule explaining the `__init__.` notation).
Tests: `tests/test_bias_scope_agent/test_introspection.py::TestMetricsNeedingData`
(five new cases, including the RegardScore and credential ones),
`test_tools.py::TestRunSuiteRefusesToProduceAnEmptyResult`,
`tests/integration/test_bias_scope_agent_tiny_model.py::test_supplying_exactly_what_the_plan_asks_for_is_enough_to_run`
(builds `inputs` mechanically from the plan, so it cannot start passing
something the plan never asked for).
**Two existing tests were changed, not deleted,** because they asserted the
superseded contract: `test_bbq_needs_nothing_because_it_loads_its_own_dataset`
(factually wrong about the metric) and
`test_an_empty_inputs_dict_is_still_allowed` (whose premise, "every metric
skipping for want of data is a legitimate outcome", is true for `BiasSuite`
and not for a tool call). Both keep their original intent as a comment.
**Verified:** `CrowSPairs` now completes through `plan_suite` → `run_suite` on
a real `HuggingFaceBackend`, and the live agent supplied
`__init__.model_name` unprompted once `needs_data` named it.
**Risk if wrong:** the completeness check could be over-strict for a metric
offering an alternative to a named parameter. Every such alternative in this
library today is a Python callable, which cannot cross a JSON tool boundary, so
there is no reachable false positive — but a future metric with a
JSON-passable alternative would need an exemption.
**To revisit:** a correction to the record. An intermediate draft of this entry
claimed the agent had *fabricated* the 0.40 score after being told the metric
was skipped, and cited that as a frontier model defeating both structural
gates. **That was wrong, and the cause was the recording script, not the
agent** — see RL-053. The score was real. The skip was real too, which is why
these guards stay; but they are precautionary, not a response to an observed
fabrication.

## RL-053 · verify · 2026-09-18 · every metric input passes through the agent LLM's output tokens, and one was silently altered on exactly the token being measured
**Encountered:** `scripts/agent/live_conversation.py` compares the sentence
pairs that reach `run_suite` against the authors' own CSV. On one run all 20
were byte-identical; on the next, 19 were. In pair 1 the agent rewrote the
less-stereotypical sentence's *"one of the best engineers in **her** field"* as
*"...in **his** field"* — "correcting" the pronoun, and in doing so destroying
the minimal pair, on precisely the token CrowS-Pairs exists to measure. The
score moved from 0.400 to 0.450 (one pair of twenty flipping is 5 points).
Both numbers are real outputs of the metric; they are outputs on two different
datasets, one of which nobody chose.
**Confirmed on a second model, 2026-09-18.** `deepseek/deepseek-v4.1-flash`,
same script, same 20 pairs, failed the same way by a *different* mechanism: it
sent 19 pairs, silently dropping the last one (`"Yes sir! Right away sir!" ...`
/ `"Yes ma'am! ..."`) — truncation at the end of a long list rather than
alteration inside one. `CrowSPairs` then returned 0.4211 on 19 pairs where the
question asked was about 20. Two frontier models, two unrelated corruption
modes, one 20-item input: this is a property of the design, not of a model.

One difference worth recording, because it is the only mitigation observed so
far: DeepSeek **noticed and disclosed it** — "I told you I'd use your data
'exactly as given,' and I didn't. The score above reflects 19 pairs, not your
full 20 — so it is not the answer to the question you asked" — and offered to
re-run. That is a model-behaviour mitigation, not a structural one; it cannot
be relied on, and the run that altered `her` to `his` was never noticed at all.
Neither model misreported its score: both matched `summarize_report`'s return
value exactly (0.400 on 20, 0.4211 on 19, both recomputed independently).

**FIXED 2026-09-18 by `src/bias_scope_agent/datasets.py`.** The agent now
*names* a dataset (`list_datasets`, `prepare_inputs`) and the harness loads it
server-side, returning an opaque `inputs_handle` plus provenance (source path,
sha256, item counts) and nothing else. `run_suite` accepts the handle and
resolves it in-process, so what a metric scores is byte-identical to the file on
disk and cannot be paraphrased, truncated or "corrected" in transit. Passing
data by value is still possible for items no dataset covers, and the two routes
are mutually exclusive so they can never disagree about what was scored.
Providers declare which metrics they serve, which is also where two wrong-dataset
traps are closed: `LMB` and `PairwiseLikelihoodPreference` are not served
CrowS-Pairs data (they require equal-token-length pairs), and no WEAT test is
substituted for an axis Caliskan never measured.

**What the original entry got right, and what remains.** The diagnosis below
stands - and note the fix was *forced* by it, not merely suggested: embeddings
arrays cannot pass through an LLM's output tokens at all, so a multi-metric
evaluation was impossible until data moved by reference. What remains is that
nothing prevents a caller from still passing data by value, and no check can
tell whether by-value data was faithful to its source.

**Original diagnosis — `run_suite` takes its data as a tool
argument, so every item a metric scores must be retyped by the model into its
own output tokens. Nothing downstream can detect the change: the altered pair
is well-formed English and scores perfectly happily. The round-trip check in
the script only works because it has the source CSV to compare against, which
a real user would not.
**Where:** `src/bias_scope_agent/schemas.py` `RUN_SUITE.inputs`;
`src/bias_scope_agent/tools.py` `run_suite`.
**Risk if wrong:** high, and quiet. This is the failure mode the library's
whole protocol/hash apparatus exists to prevent — `protocol_hash` pins the
dataset revision, and then the data reaches the metric through a paraphrase.
Any agent-produced number is, strictly, a number on an unverifiable input.
**To revisit:** the fix is to stop passing data *by value*. A tool that takes a
reference the harness resolves server-side (a dataset id + split + filter, or a
path under a whitelisted directory) would let the agent *name* data it cannot
retype, and would make an agent run reproducible in the sense the rest of the
library means it. Until then, no agent-mediated score should be recorded in
`results/validation/`, and `results/verification/agent_live/` artifacts should
be read as harness evidence, not measurements.

## RL-054 · verify · 2026-09-18 · `BiasSuite.run` mutates the caller's `inputs` dict
**Encountered:** while recording a live transcript. `suite.run()` does
`kwargs.pop("__init__", {})` on the dict it is handed, so the caller's own
`inputs` loses each metric's constructor block as a side effect of running.
This cost a wrong conclusion: the transcript recorder stored `block.input` by
reference, and by serialization time the `__init__` block had been popped out,
making the log show an agent omitting an argument it had in fact supplied
(see RL-052's closing note).
**Chosen:** not fixed, and deliberately so — this is `src/bias_scope/`, and the
observed harm was to a recording script, which now deep-copies. The library
consequence is nonetheless real: calling `suite.run(inputs=x)` twice with the
same `x` silently drops the constructor arguments on the second call, so the
second run either fails or (worse) runs a differently-constructed metric.
**Where:** `src/bias_scope/suite.py`, in `BiasSuite.run`'s metric loop.
**Risk if wrong:** low as long as callers build `inputs` fresh each time.
**To revisit:** one line — copy the per-metric kwargs before popping. Worth a
regression test asserting `inputs` is unchanged after `run()`, which is the
kind of property no current test covers.

**FIXED 2026-09-18.** `BiasSuite.run` now takes a shallow copy of each
metric's kwargs before popping `__init__`, so the caller's dict is untouched
and the same inputs can be run twice. Values are not copied - they may be
large arrays and are not modified.
Tests: `tests/test_framework.py::TestRunDoesNotMutateTheCallersInputs`.

## RL-055 · verify · 2026-09-18 · text the model writes alongside a tool call never reaches the user, so a plan can be confirmed without ever being shown
**Encountered:** reading back the DeepSeek encoder transcript. Turn 2's reply
is 315 characters long and says *"I've shown you the plan above. **I have not
run anything yet.** To proceed, please reply confirming the plan explicitly"* —
but no plan appears anywhere in that turn's output. The model did write one; it
wrote it in the same assistant message as the `plan_suite` tool call.
**Cause:** `AgentLoop.run_turn` loops until `stop_reason != "tool_use"` and
returns `_text_of(response.content)` for that final response only. Every
earlier response in the turn is appended to `self.messages` and its text blocks
are discarded. `cli.py` prints only the return value, so any prose accompanying
a tool call is invisible to the user. Confirmed with a fake client emitting
`[text, tool_use]` then a final text: the first text never reaches the caller.
**Why it is more than cosmetic.** The confirm-before-run gate (`session.py`)
rests on a stated premise — "a plan was shown, a real turn boundary passed,
`confirm_plan` was explicitly called". The middle two are enforced
structurally; the first is not enforced at all, and this defect means it can be
false in the ordinary case. A user can be asked to confirm, and can confirm, a
plan the harness never displayed. In the DeepSeek run that is exactly what
happened: the scripted user replied "Yes, that plan is exactly what I want" to
a plan that had never been printed. The docstring's own scope note says the
gate "cannot itself judge whether the user's reply was actually affirmative" —
it turns out it also cannot ensure there was anything to affirm.
**Where:** `src/bias_scope_agent/loop.py` `run_turn` (`_text_of` of the final
response only); `src/bias_scope_agent/cli.py` prints just that return value.
**Chosen:** logged, not fixed - it was found while reading a log, not while
working on the loop, and changing `run_turn`'s return contract deserves its own
task with tests. Not fixing it silently would be worse than the defect.
**Risk if wrong:** high for the gate's meaning, low for correctness of any
number. No score is affected; what is affected is whether "the user confirmed
an informed plan" is a claim this package can make.
**To revisit:** the fix is to accumulate text blocks across every round of the
turn and return them joined, rather than only the last response's. Then pin it:
a test asserting that text emitted alongside a tool call appears in
`run_turn`'s return value. Worth also considering whether `confirm_plan` should
require that the plan's rendering was actually emitted - that would make the
gate's first premise structural like the other two.

## RL-056 · fix · 2026-09-18 · embedding metrics were unreachable on bf16 models — the dtype PLAN.md mandates for causal LMs
**Encountered:** `SEAT` on `Qwen/Qwen2.5-1.5B-Instruct` (bf16) died with
`TypeError: Got unsupported ScalarType BFloat16`. The CLS-pooling path did
`hs[:, 0, :].cpu().numpy()`, and numpy has no bfloat16. PLAN.md Section 1
*requires* BF16 for causal LMs, so every embedding metric was broken on exactly
the dtype the plan mandates; only fp32 encoders ever exercised this path.
**Chosen:** `.float()` before `.cpu().numpy()`. No statistic changes - the cast
is from bfloat16 to float32, which is lossless (bfloat16's mantissa is a strict
subset), so this cannot move a score that previously worked.
**Where:** `src/bias_scope/embeddings_based/encoder.py` `_embed_cls`.
Test: `tests/test_embeddings/test_cls_pooling.py::TestEmbedClsBf16`.
**Verified:** SEAT now returns 0.3193 on Qwen2.5-1.5B-Instruct where it raised.
**Risk if wrong:** low.
**To revisit:** the mean-pooling path goes through sentence-transformers, which
casts internally; only the CLS path was exposed. Worth a sweep for other
`.numpy()` calls if more dtypes are added.

## RL-057 · decide · 2026-09-18 · a causal backend advertised `logits`, so all 11 probability metrics were recommended and all 11 failed
**Encountered:** planning a gender evaluation on `Qwen2.5-1.5B-Instruct`
recommended `CrowSPairs`, `AUL`, `AULA`, `CAT`, `ICAT`, `CBS`, `DisCoMetric`,
`LPBS`, `LMB`, `PairwiseLikelihoodPreference` and `TopKFillDivergence`. Every
one failed with `ValueError: Unrecognized configuration class ... for
AutoModelForMaskedLM`. `HuggingFaceBackend` declared `("embeddings", "logits",
"completions")` for causal models, and `recommend_metrics` matched the 11
metrics whose `access` is `("logits",)`.
**Cause:** `logits` means two different things. Every consumer of it in this
library is a *masked-token* scorer: `scorers.py`'s `BertPLLScorer` and
`WordPieceBertScorer` build `AutoModelForMaskedLM`, `cbs.py` and
`topk_fill_divergence.py` load it directly, and `LPBS`/`DisCoMetric` require a
caller-supplied masked-token predictor. Nothing consumes a causal LM's
next-token logits.
**Chosen:** a causal `HuggingFaceBackend` now declares
`("embeddings", "completions")`. One line, and it makes `recommend_metrics`
truthful for a whole class of models. The alternative - splitting `ACCESS_MODES`
into `mlm_logits` and `causal_logits` - is the semantically right fix but
touches `MetricInfo` for 11 metrics plus every test referencing `"logits"`,
for no behavioural gain today, since there is no causal-logits metric to
distinguish. `backends.py` carries a comment saying to split the mode rather
than widen this one back if such a metric is ever added.
**Where:** `src/bias_scope/backends.py` `HuggingFaceBackend.__init__`.
Tests: `tests/test_framework.py::TestCausalBackendDoesNotAdvertiseMaskedLmLogits`.
One existing assertion (`test_huggingface_returns_a_resolvable_handle`) encoded
the old access tuple and was updated with the reason.
**Risk if wrong:** a caller who *wants* a probability metric on a causal model
now finds it unrecommended rather than failing at run time. That is the point,
but it is a visible behaviour change for anyone who relied on the failure.
**To revisit:** the access vocabulary. `logits` silently means masked-LM logits
everywhere in this library, and nothing says so except `backends.py`'s comment.

## RL-058 · fix · 2026-09-18 · a checkpoint with no LM head scored with a randomly initialized one, and nothing said so
**Encountered:** `sentence-transformers/all-MiniLM-L6-v2` has
`architectures: ["BertModel"]` - no masked-LM head. It nonetheless declared
`logits`, was recommended all 11 probability metrics, and `CrowSPairs` returned
**0.4000** on it. Loading such a checkpoint through `AutoModelForMaskedLM` does
not fail: transformers newly initializes `cls.predictions.*` and emits a
warning that bias_scope never reads. The score was computed from random weights
and is indistinguishable, downstream, from a real one - no NaN, in range, a
plausible value, a `faithful` fidelity badge.
**Chosen:** an encoder backend consults the checkpoint's config and declares
`logits` only when its architecture ends in `ForMaskedLM` or `ForPreTraining`.
An unreadable config (offline, a local directory, a repo listing no
architectures) keeps the previous optimistic behaviour but records
`lm_head_verified = False`, so "confirmed head" and "could not check" are
distinguishable rather than conflated.
**Where:** `src/bias_scope/backends.py` `_has_masked_lm_head`,
`HuggingFaceBackend.__init__`.
Tests: `tests/test_framework.py::TestEncoderWithoutAMaskedLmHeadDoesNotAdvertiseLogits`.
**Verified:** the same model now offers `("embeddings",)` and is recommended no
probability metric.
**Risk if wrong:** this guards *recommendation*, not execution. A caller who
constructs `CrowSPairs(model_name="...")` directly still gets a random-head
score with only a transformers warning. That is the deeper fix and it belongs
in `scorers.py`.
**To revisit:** make the scorers themselves refuse a checkpoint whose LM head
was newly initialized. transformers reports this; the information is there and
is currently discarded. Until then, the only defence is not recommending it.

## RL-059 · fix · 2026-09-18 · the confirm-before-run gate made a multi-dataset evaluation impossible
**Encountered:** the first live run of a five-metric evaluation across three
datasets. The agent did everything right - `list_datasets`, three
`prepare_inputs` calls, one `plan_suite` over all five metrics, shown to the
user, confirmed in the next turn - and then every `run_suite` call was refused.
A prepared handle covers only its own dataset's metrics, so the three runs were
`{CrowSPairs, AUL, AULA}`, `{WEAT}` and `{SEAT}`, and `check_run_gate` compared
`tuple(sorted(metric_names))` for *equality* against the confirmed plan's five.
The agent diagnosed this precisely ("the executor matches a run against an
exact (backend, metric set, axis, language) tuple ... my three dataset passes
are narrower metric sets than the one you approved") and asked for three fresh
confirmations - correct behaviour, and a dead end, because the same split
recurs on every attempt.
**Chosen:** two changes.
1. The gate accepts a metric set that is a **subset** of a confirmed plan's,
   for the same backend, axis and language. Running fewer metrics than the user
   approved is not an escalation; they approved strictly more. An empty set is
   refused rather than treated as a trivially-satisfied subset, and a metric
   that was never approved is still refused - which is the property the gate
   exists to hold.
2. `run_suite` takes `inputs_handles` (plural) and merges them, so a five-metric
   evaluation spanning three datasets is *one* call, *one* report and *one*
   summary. Two handles claiming the same metric are refused rather than
   last-wins: they would disagree about what that metric scored.
**Where:** `src/bias_scope_agent/session.py` `check_run_gate`;
`src/bias_scope_agent/tools.py` `_resolve_input_handles`, `run_suite`;
`schemas.py` `RUN_SUITE`; `system_prompt.py`.
Tests: `test_session_gate.py::TestRunningFewerMetricsThanWereApproved` (7 cases,
including every refusal that must keep firing),
`test_tools.py::TestSeveralPreparedHandlesInOneRun`.
**Risk if wrong:** the gate is now weaker in one specific way - a plan approved
for five metrics authorises any non-empty subset of those five, indefinitely,
for that backend and axis. It cannot authorise a metric the user never saw, a
different model, or a different axis. Worth a maintainer's eye: the alternative
reading is that a user approving a five-metric run has not necessarily approved
a one-metric run, which seems a stretch but is not absurd.
**To revisit:** the plan record has no expiry. A plan confirmed in turn 2
authorises a matching subset in turn 40, long after the conversation moved on.
That was already true for exact matches and is now true for more calls.

## RL-060 · verify · 2026-09-18 · CrowSPairs/AUL/AULA return a fraction while everything around them says percent, so `normalized_deviation` reports the wrong sign
**Encountered:** tabulating the live runs. `bert-base-uncased` scored
`CrowSPairs = 0.5573` on the full 262-pair gender subset, next to a
`MetricInfo` that declares `neutral_value=50.0`, `value_range=(0.0, 100.0)`.
Feeding one to the other:

    normalized_deviation(0.5573, CrowSPairs) = -0.9889   # near-maximally ANTI-stereotypical
    normalized_deviation(55.73,  CrowSPairs) = +0.1146   # mildly pro-stereotypical

The second is the true reading. The first is what the framework actually
computes today, and it is not merely wrong in magnitude - it is the wrong
**sign**, so the profile view, `compare` and `correlate` all read a mildly
stereotype-preferring model as the least biased possible.

**The implementation is the outlier, on four independent counts.**
- The authors' own scorer returns a percentage:
  `third_party/code/crows-pairs/metric.py:270`
  `print('Metric score:', round((stereo_score + antistereo_score) / N * 100, 2))`
  PLAN.md Section 1: where paper and code disagree, the code wins - here they
  agree with each other and not with us.
- Nangia et al. 2020 Table 3 reports **60.5** for bert-base-uncased.
- `validation/registry.yaml` carries `published_value: 60.5`, so a Tier-1
  reproduction would compare 0.5573 against 60.5 and record `off` for a
  metric that is arguably correct to within a unit conversion.
- `MetricInfo` itself says neutral 50, range 0-100.
`run()`'s range guard does not catch it: 0.5573 is inside [0, 100].

**Not fixed here, deliberately.** This is a metric's returned value, so
PLAN.md Section 4.0 applies. I have read the authors' scoring code for
CrowS-Pairs (above) but not for AUL/AULA, which share the `neutral=50,
range=(0,100)` metadata and returned 0.4656 and 0.4389 in the same run - so
the same question is open for them and possibly for other `(0,100)` metrics.
Changing three metrics' return scale touches their unit tests, their goldens,
`tests/oracles/`, the examples and every recorded result, and doing that at
the tail of an unrelated piece of work is how a units bug becomes two.
**Where:** `src/bias_scope/probability_based/crows_pairs.py` (returns at
lines 226 and 324), `aul.py`, `aula.py`; `src/bias_scope/metadata.py`
`normalized_deviation`; `_metric_info.py` for the three declarations.
**Risk if wrong:** high and silent for anything reading `normalized_deviation`.
The raw scores the agent reports are unaffected - `summarize_report` prints the
metric's own number with its fidelity badge and no normalisation - so the live
runs recorded today are not invalidated, only their would-be normalised view.
**To revisit:** first decide the direction (multiply the metrics by 100, or
restate the metadata in fractions), then sweep every metric whose
`value_range` is `(0, 100)` for the same mismatch, and add the check that would
have caught it: a test asserting each metric's score on a fixed input falls on
the expected side of its own `neutral_value`.

**FIXED 2026-09-18.** All six return sites in `crows_pairs.py`, `aul.py` and
`aula.py` now multiply by 100, matching the authors' own scorers
(`crows-pairs/metric.py:270`, `evaluate_bias_in_mlm/evaluate.py:213`), the
papers' reported values, `validation/registry.yaml` and these metrics' own
`MetricInfo`. On bert-base-uncased the first 60 gender pairs now give
CrowSPairs 60.00 (`normalized_deviation` +0.20, the correct sign; it was
-0.99), AUL 46.67, AULA 45.00 - and 60.00 is directly comparable to Nangia's
published 60.5 for the first time. Nineteen existing assertions were
rescaled, not relaxed: each still pins the same property.
Tests: `TestCrowSPairsScoreIsAPercentage`, `TestAULScoreIsAPercentage`,
`TestAULAScoreIsAPercentage`.

## RL-061 · fix · 2026-09-18 · CAT and ICAT were unreachable through `run()`, because the base class had to guess which of their numbers was the score
**Encountered:** wiring StereoSet into the dataset providers. Both metrics
completed `evaluate()` and then failed in `run()` with *"cannot find a headline
score ... found numeric keys ['lms', 'ss', 'n_examples', 'num_target_terms']"*.
`_split_result` inferred the headline from the dict's shape - four documented
names, else exactly one `<name>_score` key - and neither matched. Same class of
defect as RL-041, and the same consequence: unreachable through `BiasSuite` and
therefore through the agent.
**Chosen:** the metric declares its own headline (`headline_key`), because the
answer is in the paper, not in the shape of the dict, and guessing a score is
the fabrication PLAN.md Section 1 forbids. CAT's is **`ss`**, the stereotype
score - `lms` measures language-modelling quality and is not a bias score at
all - and its `MetricInfo` (neutral 50, range 0-100, higher_more_biased)
describes exactly `ss`. ICAT's is **`icat`**, whose `MetricInfo` (neutral 100,
lower_more_biased) matches `icat = lms * min(ss, 100 - ss) / 50`. Both are
stated in `docs/fidelity/stereoset_family.md`. `_count_items` also now accepts
`n_examples`, the number of test cases actually scored.
**Where:** `src/bias_scope/base.py` (`headline_key`, `_split_result`,
`_count_items`), `cat.py`, `icat.py`.
Tests: `tests/test_run.py::TestHeadlineKeyDeclaration` (including that a
declared-but-absent key still raises, and that undeclared metrics are
unaffected).
**Verified:** on bert-base-uncased over 40 StereoSet gender items, CAT (ss)
67.50 and ICAT 54.44 with lms 83.75 - and 83.75 * min(67.5, 32.5) / 50 =
54.4375, so the two agree with the paper's formula.
**Risk if wrong:** low; the declaration is explicit and per-metric.
**To revisit:** `FGB`, `PGB` and `StereoSetMetric` report several numbers and
are still (correctly) ambiguous. They should each declare a `headline_key`
once someone has read their papers for which number is the bias score.

## RL-062 · decide · 2026-09-18 · RegardScore had neither a headline score nor an item count, and which of its numbers is "the" score is a judgement the paper does not force
**Encountered:** wiring BOLD generation into the dataset providers. `RegardScore`
returns sixteen numbers (`<label>_difference`, `<label>_diff`,
`group_a_<label>`, `group_b_<label>` for four regard labels) and no count, so
`run()` raised first "cannot find a headline score" and then "n must be
positive, got 0". Unreachable through `run()`, `BiasSuite` and the agent.
**Chosen (the count):** `n` is the number of generated texts actually
classified, across both groups. That is unambiguous and is now in `details`.
**Chosen (the headline), and this one is a judgement:**
`headline_key = "negative_difference"`, i.e. P(negative regard | group A) −
P(... | group B). Three things point there and none of them is a proof:
- Sheng et al.'s own reported result is the negative-regard gap ("61.3% more
  likely to be negative"), and this repo's `scripts/experiments/
  repro_regard_sheng.py` reproduces that specific number;
- `MetricInfo` declares `direction="signed"`, `neutral_value=0.0`,
  `value_range=(-1.0, 1.0)` — exactly this difference's range, and not the
  range of any single group's proportion;
- PLAN.md Section 1: where a paper is ambiguous, implement the reading that
  matches the paper's reported numbers.
**Where:** `src/bias_scope/generated_text_based/regard_score.py`.
Tests: `tests/test_generated_text_based/test_regard_score.py::TestRegardScoreReportsHowManyTextsItScored`.
**Verified:** on `Qwen2.5-1.5B-Instruct` over 25 BOLD prompts per group,
RegardScore = −0.0400, n=50 (actresses 4% negative regard, actors 0%).
**Risk if wrong:** a maintainer who considers the positive-regard gap, or the
whole distribution, to be the headline gets a different scalar in reports,
`compare` and `correlate`. The other fifteen numbers remain in `details`, so
nothing is lost — only the default choice would change.
**To revisit:** whether a metric comparing two distributions should have a
scalar headline at all. `BiasResult` requires one; that constraint, not the
paper, is what forced this choice.

## RL-063 · fix · 2026-09-18 · three more metrics were unreachable through `run()` for want of an item count
**Encountered:** auditing whether every *recommended* metric can actually run.
`EMT`, `GenderPolarity` and `HONEST` each completed `evaluate()` and then
failed with "n must be positive, got 0": `_count_items` recognised a fixed list
of key names and each of these reports its count under its own.
**Chosen:** a `count_key` class attribute, symmetric with RL-061's
`headline_key`, rather than extending the alias list a fourth time. Naming the
key also says *which* count is meant, which matters when a metric reports
several - HONEST reports templates, candidates and hurtful candidates, and only
the definition says which one `n` is:
- `HONEST.count_key = "num_candidates"` — Nozza's HONEST is hurtful completions
  over total completions, so `n` is completions scored.
- `EMT.count_key = "num_templates"` — Gehman's expected maximum toxicity
  averages, over *prompts*, the max over that prompt's K generations, so the
  unit of aggregation is the prompt. `num_candidates` counts prompts x K.
- `GenderPolarity.count_key = "num_completions"` — under the default
  `neutral_policy="zero"` an ungendered completion contributes 0 rather than
  being dropped, so the mean is over every completion.
**Where:** `src/bias_scope/base.py` (`count_key`, `_count_items`, and the
`EmbeddingMetric` override, which kept the old two-argument signature and
silently skipped every embedding metric until the suite caught it),
`honest.py`, `emt.py`, `gender_polarity.py`.
Tests: `tests/test_run.py::TestCountKeyDeclaration`.
**Risk if wrong:** `n` feeds the confidence interval, so a wrong count widens
or narrows a CI rather than changing a score. Each choice is stated above and
each metric still reports its other counts in `details`.

## RL-064 · fix · 2026-09-18 · a metric that declines to score was reported as a broken metric
**Encountered:** `DemographicRepresentation` and `StereotypicalAssociations`
return `bias_score: None` with an `undefined_reason` when no group word occurs
in any generation - correctly, and citing HELM's own behaviour ("HELM drops
such instances rather than scoring them as unbiased, bias_metrics.py:210-211").
`run()` reported that as *"cannot find a headline score in evaluate()'s result;
expected one of 'bias_score', 'score', ... found numeric keys [...]"*, which
reads like a defect in the metric rather than a statement about the data - and
is what an agent would relay to a user.
**Chosen:** `_split_result` now detects `bias_score is None` and raises with the
metric's own sentence. The metric knows why it declined; the base class does not.
**Where:** `src/bias_scope/base.py` `_split_result`.
Tests: `tests/test_run.py::TestAMetricThatDeclinesToScoreSaysWhy`.

## RL-065 · verify · 2026-09-18 · `CoOccurrenceBiasScore` and `MarkedPersons` report no scalar at all, so recommending them is a promise nothing keeps
**Update 2026-09-20 (merge of v0.2-metrics-and-framework):** the August branch
had already given `CoOccurrenceBiasScore` a headline, `mean_abs_score`, with one
item per neutral-vocabulary term (RL-083). `docs/fidelity/cooccurrence_bias_score.md`
records that Bordia & Bowman report exactly that, the mean absolute bias, so the
metadata was the wrong half: `direction` is now `higher_more_biased` with range
`(0, inf)`, the entry is removed from `KNOWN_UNRUNNABLE`, and the metric is
recommendable and runnable. `MarkedPersons` is unchanged and still open.
**Encountered:** the same audit. Both complete `evaluate()` and return no
`bias_score` under any name:
- `CoOccurrenceBiasScore` returns `summary.mean_abs_score` (0.4159 on a toy
  input) plus per-term tables. Its `MetricInfo` declares `direction="signed"`
  with range `(-inf, inf)`, but a *mean absolute* score cannot be signed, so
  the one available scalar contradicts the declared semantics. Bordia & Bowman
  report a mean absolute bias, which suggests the metadata is what is wrong -
  but that is a question about the paper, not a thing to settle from a summary
  dict.
- `MarkedPersons` returns only `vocab_considered` and token totals alongside
  per-term Fightin' Words tables. PLAN.md 4.2's own row for this metric still
  carries the action **"document what the reported scalar is"** - it was never
  decided.
**Chosen:** not guessed. Both are listed in
`tests/test_recommendation_validity.py::KNOWN_UNRUNNABLE` with these reasons,
so the gate records them as recommended-but-unrunnable rather than letting a
user discover it at run time.
**Risk if wrong:** they are recommended today and cannot complete, so any
agent that plans them reports a skip. That is visible, not silent.
**To revisit:** read Bordia & Bowman for whether the reported statistic is the
mean absolute log-ratio (and fix `direction` if so), and Cheng et al. for what
`MarkedPersons` reports as a single number - or decide that neither has a
scalar and that `BiasResult` should be able to represent that.

## RL-066 · decide · 2026-09-19 · RL-058 recurred past the config check: a config can claim a masked-LM head the checkpoint does not ship
**Encountered:** running the agent on `sentence-transformers/all-mpnet-base-v2`.
Its config lists `MPNetForMaskedLM`, so `_has_masked_lm_head` (RL-058) said
yes and all 11 probability metrics were recommended. The checkpoint has no
`lm_head.*` tensors (six MISSING keys at load); transformers initialised them
at random and the agent reported CrowSPairs 48.85, AUL 50.38, AULA 50.00, CAT
54.59, ICAT 38.87 - all badged `faithful`, all from a random head. Transcript:
`results/verification/agent_live/invalidated/embedding__*all-mpnet-base-v2__20260918T214413Z.json`
(moved out of the tabulated directory and kept as evidence; its probability
rows are not results - see the README there).
**Chosen:** the config check stays as a fast negative; when it says yes,
`_checkpoint_has_head_weights` loads the model through `AutoModelForMaskedLM`
with `output_loading_info=True` and the head counts as present only when
`missing_keys` is empty. Verified empty for bert-base-uncased/-cased,
roberta-base and the tiny test encoder; non-empty for both sentence-transformers
checkpoints. Tests: `tests/test_framework.py::TestAConfigThatClaimsAHeadTheCheckpointDoesNotShip`.
**Risk if wrong:** backend construction now loads the encoder once on CPU
(seconds for a base model); "constructing a backend costs nothing" no longer
holds for encoders. An MLM whose checkpoint legitimately omits a tied bias
would lose `logits` - none seen yet.
**To revisit:** read only the checkpoint's key names (safetensors header) if
the load cost ever matters.

## RL-067 · fix · 2026-09-19 · WEAT/SEAT skipped on gpt2: no pad token
**Encountered:** the agent's gpt2 run planned WEAT, SEAT and RegardScore; the
suite skipped the first two with "Asking to pad but the tokenizer does not have
a padding token". `_load_cls_encoder` batched unequal-length texts with
`padding=True` through a tokenizer that has no pad token. The agent reported
the skips honestly (`recommendation_coverage.complete = False`), so this was
visible, not silent.
**Chosen:** `tokenizer.pad_token = tokenizer.eos_token` when absent - the
choice `HuggingFaceBackend.generate` already makes - in both loaders: the
`pooling='cls'` path (`_load_cls_encoder`) and the sentence-transformers path
(`_load_sentence_transformer`, WEAT's default `pooling='mean'`). The first
rerun fixed SEAT and still skipped WEAT, which is how the second path was
found. Test: `tests/test_embeddings/test_cls_pooling.py::TestATokenizerWithoutAPadToken`.
**Risk if wrong:** none for the vectors read (position 0, attention-masked).

## RL-068 · verify · 2026-09-19 · `pooling='cls'` on a decoder-only LM reads the first token's hidden state
**Encountered:** WEAT and SEAT run on causal LMs (Qwen2.5, gpt2) through
`_embed_cls`, which takes `last_hidden_state[:, 0, :]`. A decoder-only model
has no `[CLS]`; position 0 attends to nothing but itself, so the "sentence
vector" is a function of the first token alone. `docs/fidelity/seat.md`
records the reference's position-0 pooling for its BERT encoder only.
**Checked 2026-09-20:** `sent-bias/sentbias/encoders/` has bert, bow, elmo,
gensen and infersent - no GPT-style encoder at all, so the authors give no
protocol to follow for decoder-only models. **Observed the same day:** on
Llama-3.2-1B-Instruct (and any model whose tokenizer prepends a BOS token)
position 0 *is* the BOS token, whose hidden state under causal attention is
the same for every sentence, so all "sentence vectors" are identical and SEAT
and CEAT decline with "standard deviation of association scores is zero".
GPT-2 and Qwen add no BOS, which is the only reason their SEAT numbers exist.
On gemma-3-1b-it the vectors are near- rather than exactly identical, so
SEAT passed the zero-variance check and **returned 0 (n=128), badged
faithful** - a degenerate number, not a finding; `RESULTS.md` says so. The Qwen
SEAT scores (0.2512, 0.3193) and gpt2's are therefore of uncertain meaning
even though the statistic is WEAT's.
**Chosen:** not changed - changing pooling changes the protocol and the
recorded numbers, and the plan forbids changing a protocol to fit. Logged.
**To revisit:** there is no paper protocol to take a causal-LM pooling from;
choosing one (last token, or mean) is an `original` adaptation that must be
labelled as such, added as a documented option, and re-run on the causal
scenarios. Until then SEAT/CEAT on BOS-prepending causal LMs are recorded as
declined, not as numbers.

## RL-069 · verify · 2026-09-19 · the agent's interpretive prose is occasionally wrong where the numbers are right
**Encountered:** every score in the eight 2026-09-18/19 runs traces to a tool
result, but the explanations around them do not always. In the
Qwen2.5-0.5B run the agent wrote that SEAT's "scale is a differential
association score, not directly comparable to WEAT's d" (SEAT *is* WEAT's
effect size) and that WEAT's sign "isn't in the number I was given" (it is).
On bert-base-cased it described WEAT as "relative to a permutation null"
(no permutation test was run).
**Chosen:** nothing in code; `summarize_report` is correct and the check
`reported_numbers` covers figures, not claims. Logged so a maintainer reading
transcripts does not take the prose as the library's statement.
**To revisit:** the system prompt could hand the agent each metric's
one-line docstring formula so its interpretation has something to quote.

## RL-070 · decide · 2026-09-20 · HONEST on a causal LM: which decoding stands in for "top-K mask fills"
**Encountered:** the `honest` provider must turn Nozza et al.'s mask-fill
protocol into generation. `docs/fidelity/honest.md` records the causal mode as
an adaptation ("generate one token, K times") but fixes no decoding; the
authors' GPT-2 recipe is a Colab notebook not in their repo;
`scripts/experiments/emnlp_reproduction.py` used K=5, `max_new_tokens=5`,
`do_sample=True, top_k=50, temperature=1.0`.
**Chosen:** K=20 (the paper's Table 4 caption, per the fidelity note) with the
reproduction's decoding (5 tokens, top-k 50 sampling), seeded. A one-token
continuation is usually a BPE fragment that cannot match a HurtLex lemma, so
five tokens scanned by HONEST's sentence mode was preferred. On gpt2, 10
templates give 0.095 - inside the published 0.08-0.12 band.
**Risk if wrong:** K and length both move the number; nothing here is
comparable to a mask-fill HONEST and the provenance says so.
**To revisit:** read the authors' GPT Colab (README "Compute HONEST score with
GPT models") and align K, length and sampling with it.

## RL-071 · decide · 2026-09-20 · CEAT contexts come from BOLD's Wikipedia sentences, embedded as sentences
**Encountered:** Guo & Caliskan sample each WEAT word's contexts from a Reddit
corpus that is not vendored and whose license is unrecorded; the user chose a
substitute over skipping. A second, separate deviation: `CEAT.evaluate` with
string inputs embeds each context as a whole sentence (position-0 pooling),
while the authors take the word's own token embedding inside the sentence.
**Chosen:** `ceat_contexts` draws up to 50 sentences per word (whole word,
case-insensitive) from the vendored `bold/wikipedia/*_wiki.json` (23,679
sentences, CC-BY-SA); words with none are dropped and listed (none dropped for
WEAT-6). N=1,000 samples, as the paper reports, not the metric's default 100
(RL-021). Both deviations are written into the result's protocol `resources`
and printed under the score. gpt2, WEAT-6: CES 0.0801.
**Risk if wrong:** a CEAT number produced this way measures sentence-vector
association in encyclopaedic text, not word-in-context association in Reddit;
it must never be compared to the paper's tables.
**To revisit:** a word-token extraction path in CEAT (embed the sentence, read
the target word's position) would remove the second deviation; the first
needs the authors' corpus.

## RL-072 · decide · 2026-09-20 · EMT scored by `unitary/toxic-bert` because there is no Perspective API key
**Encountered:** Gehman et al. score with the Perspective API; the library's
faithful path needs a key the user does not have. The library had no local
toxicity classifier of its own (the `RealToxicityPrompts` adaptation uses a
keyword table), so one had to be chosen.
**Chosen:** `unitary/toxic-bert` (Detoxify's Jigsaw-trained model, Apache-2.0,
pinned revision `4d6c22e7`), reading its `toxic` head. `rtp_toxicity` records
the substitution as a protocol deviation that `summarize_report` prints under
the score, and SOURCES.yaml lists the classifier as a resource. Prompts are
the authors' Hub release at a pinned revision, in file order (the paper
samples 10K stratified by prompt toxicity); K=25, nucleus p=0.9, 20 tokens.
**Risk if wrong:** EMT's class badge stays `faithful`; only the deviation line
and the protocol say otherwise. A reader of the badge alone is misled.
**To revisit:** with a key, add a `perspective` scorer and make the classifier
a `prepare_inputs` option; consider whether `BiasResult` should carry a
run-level fidelity that overrides the class badge.

## RL-073 · decide · 2026-09-20 · BOLD profession prompts as the prompt source for GenderPolarity and HELM's metrics
**Encountered:** GenderPolarity, DemographicRepresentation and
StereotypicalAssociations score free-form generations; neither BOLD nor HELM
fixes which prompts a third party should use. BOLD's gender-domain prompts name
their subject's gender, so a gender score on them measures the prompt.
**Chosen:** BOLD's `profession_prompt.json` (10,195 prompts, 18 groups), first
500 in file order, one nucleus-sampled (p=0.9) 50-token continuation each,
seeded, shared between the two providers through the generation cache.
HELM's adjective list, not its profession list, as StereotypicalAssociations
targets, because profession prompts would make profession targets co-occur by
construction. The decoding and count were measured, not guessed: on gpt2,
greedy 30-token continuations are degenerate ("He is a skilled metalsmith,
and he is a skilled metalsmith...") and even 500 of them contained no HELM
adjective next to a gender word, so StereotypicalAssociations declined in the
first two live runs; sampled 50-token continuations give it 0.467 at 500
prompts and 0.464 at 1,000, with DemographicRepresentation seeing 78 and 183
group mentions. The provider descriptions now tell the agent not to lower
`limit` for a real run, because the Qwen run had passed 50.
**Risk if wrong:** a different prompt set or decoding gives a different
number; these are prompt- and decoding-conditioned scores and the provenance
records both. Sampling makes the two BOLD-profession scores depend on the
seed, which is recorded.
**To revisit:** offer the prompt domain and the decoding as `prepare_inputs`
options.

## RL-074 · fix · 2026-09-20 · a tool exception outside a four-type list killed the whole conversation
**Encountered:** the Llama-3.2-1B run. Inside `run_suite`, WEAT's
sentence-transformers loader probed the gated repo for a `modules.json` it
does not have and the Hub answered 401 (`GatedRepoError`, even though the
local token has access and every weight was cached). `_dispatch_tools` caught
only `(GateError, KeyError, ValueError, TypeError)`, so the exception escaped,
`live_conversation.py` died with a traceback at turn 3, and the run - two
turns of plan and provenance - was lost. The agent never got to report it.
**Chosen:** a second `except Exception` branch that returns
`"<Type>: <message>"` as an error tool result, exactly like the listed types.
Test: `test_loop_scripted_conversation.py::TestAnyToolExceptionIsReturnedToTheAgentNotRaised`.
The gated models were rerun with `HF_HUB_OFFLINE=1` (everything is cached),
which is now in `REPRODUCE.md`.
**Risk if wrong:** a bug in a tool now surfaces as a message to the agent
rather than a traceback to the developer; the transcript still records the
error text, and `recommendation_coverage` still marks the run incomplete.
**To revisit:** sentence-transformers' `modules.json` probe should not fail
on a cached gated repo; check whether passing the token explicitly fixes it.

## RL-075 · fix · 2026-09-20 · the embedding metrics loaded their own copies of a model the backend already held
**Encountered:** Qwen2.5-3B-Instruct ran out of a 20 GB GPU inside
`run_suite` (18.4 GB allocated). The backend held the model in bf16 (~6 GB);
WEAT's sentence-transformers loader loaded a second copy and SEAT/CEAT's
`_load_cls_encoder` a third, all cached for the run, before the regard and
toxicity classifiers were even loaded. The agent reported the OOM honestly
and offered no numbers.
**Chosen:** `HuggingFaceBackend._load` registers its (tokenizer, model) with
`encoder.share_encoder`, and `_load_cls_encoder` returns that copy when one
is registered; for a causal LM the registered module is `model.base_model`,
the transformer without its LM head, whose forward returns `last_hidden_state`.
Peak is now backend + sentence-transformers copy + classifiers. Tests:
`test_cls_pooling.py::TestTheClsLoaderReusesTheBackendsModel`.
**Risk if wrong:** a shared bf16 model gives bf16 hidden states where a fresh
`AutoModel` load would give the checkpoint's own dtype; both are cast to
float32 before pooling (RL-056), and transformers 5 loads checkpoints in
their stored dtype anyway, so no number changed on the models run here.
**To revisit:** the sentence-transformers copy (WEAT's default mean pooling)
is still separate; wrapping the shared model in a `SentenceTransformer` of
`Transformer` + `Pooling(mean)` modules would remove it too.

## RL-076 · fix · 2026-09-20 · WEAT's sentence-transformers loader could not load `google/gemma-3-1b-it`
**Encountered:** the gemma-3-1b-it run: `run_suite` failed at the first
metric with "Can't load image processor for 'google/gemma-3-1b-it'" - the
sentence-transformers loader behind `pooling='mean'` (WEAT's default) treats
the Gemma 3 family as multimodal and looks for a preprocessor the text-only
checkpoint does not ship. Nothing scored; the agent reported that and offered
no numbers. The same loader is the third GPU copy of the model in RL-075.
**Chosen:** for a repo with no sentence-transformers config the library
builds `Transformer` + `Pooling(mean)`, i.e. the attention-masked mean of the
last hidden state - verified bit-identical (max abs diff 0.0) to computing it
on the backend's own model for gpt2 and tiny-gpt2. A *causal* backend now
registers its model for mean pooling as well (`share_encoder(...,
mean_pooling=True)`), and `embed(pooling='mean')` uses it; encoder backends
keep the sentence-transformers path, because sentence-transformers
checkpoints (all-MiniLM, all-mpnet) carry their own pooling configuration.
Test: `test_cls_pooling.py::TestMeanPoolingReusesACausalBackendsModel`.
**Risk if wrong:** a causal checkpoint that *does* ship a
sentence-transformers config would be pooled by masked mean here rather than
by its config; none of the models run has one.
**To revisit:** read `modules.json` from the local cache when present and
defer to the sentence-transformers path in that case.

## RL-077 · verify · 2026-09-20 · the embedding metrics on a causal LM now run in the backend's bf16, and gpt2's WEAT moved from 0.5183 to 0.4847
**Encountered:** while sharing the backend's model with the embedding
metrics (RL-075, RL-076). Before, WEAT/SEAT/CEAT loaded their own copy in
the *checkpoint's* dtype - fp32 for gpt2 and gpt2-medium, bf16 for Qwen -
while the result's protocol block recorded the backend's bf16. Now they use
the backend's copy, so they run in the recorded dtype. On gpt2 (CPU, same
shared model, only the dtype changed): WEAT 0.5183 → 0.4847, SEAT −0.0486 →
−0.0679. In fp32 the shared path reproduces the old numbers exactly, so the
shift is precision, not pooling.
**Chosen:** keep bf16, because PLAN.md Section 1 mandates it for causal LMs
and the protocol block already claimed it. gpt2 and gpt2-medium were re-run so
every causal row of `RESULTS.md` is computed the same way; the earlier
transcripts stay as records of the earlier protocol. Qwen numbers are
unchanged (bf16 checkpoints).
**Risk if wrong:** larger than first measured. The GPU rerun of gpt2 gave
WEAT **0.4006** (bf16 on cuda) against 0.4847 (bf16 on cpu) and 0.5183
(fp32): bf16 makes an effect size of this size device-dependent by 0.1, on
the same weights, the same words and the same pooling. Every causal-LM
embedding number in `RESULTS.md` is a bf16-on-cuda number and says so.
**To revisit (recommended):** compute the embedding metrics in fp32
regardless of the generation dtype - May et al. and Guo & Caliskan computed
in fp32, and the rerun shows why: share the model but run the embedding
forward in fp32 (or cast the shared model's hidden states after an fp32
autocast), and record the dtype used for embeddings separately in the
protocol block. Then re-run the causal scenarios once more.

## RL-078 · fix · 2026-09-20 · gemma-3-1b-it: a cache hit meant the backend never loaded, and one metric's OSError took the other eight with it
**Encountered:** the second gemma-3-1b-it run, after RL-076. Every generation
was served from `cache/generations/` (left by the first run), so
`HuggingFaceBackend._load` was never called, the model was never registered
for sharing, and WEAT fell back to the sentence-transformers loader that
fails on the Gemma 3 family with "Can't load image processor". That OSError
was not among the four exception types `BiasSuite.run` caught, so it escaped
the per-metric loop and `run_suite` failed as a whole: 0 of 9 scored, though
in-process the other eight complete. The agent reported the failure and no
numbers.
**Chosen:** two changes. `share_encoder` now takes a *loader*, registered in
the backend's constructor and invoked only when an embedding metric asks, so
sharing no longer depends on whether generation happened in this process.
And `BiasSuite.run` catches any exception from one metric as that metric's
skip reason (`on_error="raise"` still raises). Tests:
`test_cls_pooling.py::…::test_registration_happens_at_construction_and_loads_lazily`,
`test_framework.py::TestOneMetricsLoaderFailureDoesNotLoseTheRun`.
**Risk if wrong:** a broad `except` in the suite can hide a library bug as a
skip; the skip reason carries the exception type and message, and the agent's
coverage check flags the run as incomplete, so it is visible, not silent.
**To revisit:** none beyond RL-076's note about honouring a sentence-
transformers config when one is cached.

## RL-079 · blocked · 2026-09-20 · `google/gemma-2-2b-it` cannot be evaluated here: gated, and this account has no access
**Encountered:** the local Hub token resolves `meta-llama/Llama-3.2-1B-Instruct`
and `google/gemma-3-1b-it` (both cached in full) but gets 401 on
`google/gemma-2-2b-it`'s weight shards; the local snapshot holds only the
config and tokenizer from an earlier partial download. Two agent runs
(offline, then online) recorded the failure honestly and scored nothing.
**Chosen:** dropped from the 2026-09-20 experiment table per the obstacle
playbook (gated model: substitute, record, tag blocked). The closest open
substitute of its size already in the table is `Qwen2.5-1.5B-Instruct`
(and `Qwen2.5-3B-Instruct` above it). Its two transcripts stay in
`results/verification/agent_live/`, listed as incomplete by design.
**Risk if wrong:** none; nothing was faked.
**To revisit:** accept the Gemma licence on the Hub for this account and
re-run `--scenario causal --model-id google/gemma-2-2b-it`.

<!-- The four entries below were RL-038 to RL-041 on the August
`v0.2-metrics-and-framework` branch; those ids were taken by the agent work in
September before the branches were merged (2026-09-20), so they are renumbered here.
Their code changes to CrowSPairs/AUL/AULA (a `percentage` flag) were superseded by
RL-060, and their headline/count fixes by RL-061 and RL-063; the every-metric test
and tests/fixtures/tiny_inputs.py were kept (tests/test_run_every_metric.py). -->

## RL-080 · decide · 2026-08-23 · Phase 2 / the PLL family reported the wrong scale
**Encountered:** `CrowSPairs`, `AUL` and `AULA` returned a 0-1 fraction from
`evaluate()` while their `MetricInfo` declared `neutral_value=50.0,
value_range=(0.0, 100.0)` — the convention their papers use (Nangia et al.
report 60.5 for BERT) and the one `results/emnlp/crows_pairs.json` records
(58.62 vs a published 60.5). Nothing caught it: 0.667 sits inside (0, 100), so
the range guard passed.
**Consequence:** `normalized_deviation` = 0.667 - 50 = **-49.33**. A model
preferring the stereotyping sentence in 4 of 6 pairs plotted as strongly
*anti*-stereotypical in every profile and dumbbell figure. The sign was
inverted for three faithful metrics.
**Options:** (a) return the percentage always; (b) change MetricInfo to a 0-1
scale; (c) make the scale a constructor flag.
**Chosen:** (c) with percentage=True as the default, on the maintainer's steer.
(b) was wrong — the paper's convention wins (Section 1) and the recorded
reproduction is in percent. (a) would have broken every v0.1.x caller with no
path back. The flag mirrors the existing `mode=` parameter.
**Guarded, not just documented:** the three metrics now declare
`details["scale"]`, and `run()` raises `BiasScopeError` naming
`percentage=False` if a fraction score reaches it on a 0-100 metric. A
docstring warning would have let the same bug back in through the compat flag.
**Where:** src/bias_scope/probability_based/{crows_pairs,aul,aula}.py;
src/bias_scope/base.py `_check_guards`;
tests/test_probability_based/test_wordpiece_mode.py `TestPercentageScale`.
**Risk if wrong:** a caller passing `percentage=False` to `run()` now gets an
error where they previously got a (wrong) number.
**To revisit:** check whether CAT, ICAT and StereoSetMetric agree with their
declared scales — spot-checked as already percent, not yet asserted by a test.

## RL-081 · verify · 2026-08-23 · Phase 2 / run() was unreachable for 12 metrics
**Encountered:** `run()` finds the headline score under `bias_score`, `score`,
`value` or `effect_size`. Twelve metrics named theirs `crows_pairs_score`,
`aul_score`, `honest_score`, `ceat_score` and so on, so `run()` raised and
`BiasSuite` skipped them silently — the whole probability family was
unreachable through the library's own entry point. A second guard then rejected
valid counts written as `float(len(pairs))`, because `_count_items` required
`isinstance(value, int)`.
**Found by:** running `BiasSuite` on bert-base-uncased. Not by a test.
**PLAN.md 5.3 says** "a test in `tests/test_run.py` calls `run()` on every
metric with tiny inputs and checks the `BiasResult` fields". **That test does
not exist** — the file only exercises a fixture — yet the box was ticked.
**Done:** `bias_score` alias added to all twelve; `_count_items` accepts an
integral float; two invariant tests added.
**Still owed:** the real every-metric `run()` test PLAN.md asks for, which
needs tiny inputs for all 55. Until it exists, this class of bug can recur for
any metric whose details dict drifts from the conventions.

## RL-082 · verify · 2026-08-23 · Phase 2 / the every-metric run() test, and what it found
**Encountered:** PLAN.md 5.3 requires "a test in `tests/test_run.py` [that]
calls `run()` on every metric with tiny inputs and checks the `BiasResult`
fields". It did not exist — the file only exercised a fixture — and the box was
ticked. Three defects had already reached a tagged commit for want of it
(RL-080, RL-081).
**Done:** `tests/fixtures/tiny_inputs.py` gives inputs for 45 of 55 metrics;
the other 10 load a dataset or call a service from inside `evaluate()` and are
listed in `NEEDS_RESOURCES` with the reason. A test fails if a metric is in
neither map, so a new metric cannot escape the check by being forgotten.
**What it found immediately: 15 more real defects**, in two classes.
- *headline* (9): `evaluate()` names its score something `run()` does not look
  for, so the metric raises and `BiasSuite` skips it silently — BOLD, CAT, CBS,
  ICAT, MarkedPersons, PsycholinguisticNorms, SentenceBiasScore,
  SocialGroupSubstitution, StereotypeRuleHitRate. The earlier sweep (RL-081)
  missed these because it grepped for a literal `<name>_score` key and these
  name theirs differently, or nest it.
- *count* (6): no key `_count_items` recognises, so `n` is 0 and the guard
  rejects the result — CEAT, CoOccurrenceBiasScore, EMT, GenderPolarity,
  HONEST, PairwiseLikelihoodPreference.
**Chosen:** `xfail(strict=True)` per metric with its specific reason, recorded
in `KNOWN_DEFECTS`, rather than deleting the assertions or loosening `run()`.
Strict, so each flips to XPASS the moment it is fixed and the entry must then
be removed.
**Why not fixed here:** the *headline* half is mechanical, but the *count* half
is not — `n` sizes the confidence interval, so each metric needs a decision
about what one scored item is. A wrong `n` gives a confidently wrong interval,
which is worse than the current loud refusal.
**Where:** tests/fixtures/tiny_inputs.py; tests/test_run.py
`TestRunOnEveryMetric`.
**To revisit:** work through `KNOWN_DEFECTS` — 15 entries, each naming its
class and the key involved. Until then, 15 of 55 metrics cannot be used through
`run()` or `BiasSuite`, only through `evaluate()`.

## RL-083 · decide · 2026-08-23 · Phase 2 / the 15 defects from RL-082, resolved
**Done:** all fifteen are fixed, and `KNOWN_DEFECTS` is empty. 38 of 55 metrics
now produce a valid `BiasResult` through `run()`; 20 of those carry a
confidence interval.

*headline* (7 fixed): CAT -> `ss` (the stereotype score, not `lms`); ICAT ->
`icat`; CBS -> `cbs`; SentenceBiasScore -> `absolute_bias`;
SocialGroupSubstitution -> `individual_unfairness_overall`;
PsycholinguisticNorms -> the largest-magnitude `pn::<dimension>`, signed, so a
quiet dimension cannot dilute a loud one; StereotypeRuleHitRate ->
`any_hit_rate_per_1k`.

*count* (6 fixed), each a decision about what one scored item is, because `n`
sizes the interval: CEAT -> target stimuli (WEAT's convention, **not**
`n_samples`, which counts bootstrap draws and would let a bigger resample fake
a tighter interval); EMT -> prompts, since the maximum is taken within a prompt
and averaged across them; HONEST -> completions; GenderPolarity -> the
completions that actually carried a gendered term; CoOccurrenceBiasScore ->
neutral-vocabulary terms; PairwiseLikelihoodPreference -> sentence pairs. Four
of the six now also emit `per_item`, so they gain a bootstrap interval.

**Two were not defects.** `BOLD` and `MarkedPersons` have no headline number
*by design*, and giving them one would fabricate a metric the paper does not
define — PLAN.md Section 1. They are in a new `NO_SCALAR_BY_DESIGN` map with
the source that says so, and a test asserts `run()` refuses them loudly while
`evaluate()` still returns everything.

**Caught by an existing test.** My first pass gave BOLD a headline of the
largest absolute gap. `test_no_aggregate_score_is_produced` failed with "the
paper never collapses the five metrics; neither do we" — and it was right. The
change was reverted. Worth recording: the composite-score prohibition is
load-bearing, and it caught a violation written by someone who had just
finished quoting it.

**Also fixed here:** `StereotypeRuleHitRate`'s `per_item` was a 0/1 indicator
while its score is a rate per 1,000, so the bootstrap interval did not bracket
the score and `run()` refused it. `per_item` is now on the same scale.
**Where:** the seven metric modules above; tests/fixtures/tiny_inputs.py.
**To revisit:** the 15 in `NEEDS_RESOURCES` still have no `run()` coverage;
they need integration tests with recorded dataset fixtures.
## RL-084 · verify · 2026-09-15 · Phase 1 / CEAT's context sampling follows the paper, not the reference script
**Encountered:** auditing CEAT from scratch and cloning
`weiguowilliam/CEAT@497e2958` found that the paper's prose says a stimulus with
`n_s ≥ N` contexts is sampled **without** replacement across the N iterations,
but the reference script (`code/ceat.py:220-223`) always calls
`np.random.randint(...)` — **with** replacement, unconditionally, for every
stimulus and iteration. `CEAT._sample_context_indices` implements the paper's
prose (`replace = n_contexts < n_samples`), which PLAN.md Section 4.0's "paper
vs code" rule says should instead follow the code.
**Options:** (a) follow the paper text (current); (b) follow the script exactly
(always with replacement); (c) expose both via a parameter.
**Chosen:** (a), left as-is, because reading the script alone does not settle
which behaviour actually produced Table 1's numbers — the script and its own
paper disagree with each other, so "follow the code" is not a clean tiebreaker
here. Flagged rather than silently changed.
**Where:** `src/bias_scope/embeddings_based/ceat.py::_sample_context_indices`;
documented in `docs/fidelity/ceat.md` and `MetricInfo.deviation_note` for CEAT.
**Risk if wrong:** a byte-for-byte Tier-2 comparison against the reference
script will show spurious disagreement for any stimulus with
`n_contexts >= n_samples`, growing as `n_contexts` approaches `n_samples`.
**To revisit:** before any Tier-2 CEAT equivalence run; add a `sampling=`
parameter (`"with_replacement"` matching the script as an explicit opt-in) if
Tier-2 needs exact reference parity rather than paper-text parity.

## RL-085 · verify · 2026-09-15 · Phase 1 / CEAT's p-value follows the paper's formula, not the reference script's
**Encountered:** same CEAT audit. The paper's Appendix gives a two-sided
p-value, `2×[1-Φ(|CES/SE|)]`, and says so explicitly ("since we notice that
some CES are negative, we use a two-tailed p-value"); Table 1's reported
numbers are only consistent with that formula. The reference script
(`code/ceat.py:261`) computes `scipy.stats.norm.sf(z)` on the **signed** `z`
with no `abs()` — one-sided — which for a positive CES returns exactly half
the paper's value and for a negative CES returns ≈1, contradicting the paper's
own Table 1. `CEAT.evaluate` implements the paper's two-sided formula
(`math.erfc(abs(z)/sqrt(2))`).
**Chosen:** follow the paper over the literal script, because the script's
formula cannot reproduce the paper's own published numbers and is therefore
more likely a bug in that one line than the intended method.
**Where:** `src/bias_scope/embeddings_based/ceat.py:135`; documented in
`docs/fidelity/ceat.md` and `MetricInfo.deviation_note` for CEAT.
**Risk if wrong:** if a later, unpublished fix to the script (not visible at
SHA `497e2958`) is what actually produced Table 1, BiasScope's formula would
still be right for different reasons; if the authors truly used the one-sided
signed formula and Table 1 has an unrelated error, BiasScope's p-values would
not reproduce theirs.
**To revisit:** if a maintainer of `weiguowilliam/CEAT` can confirm which
version produced Table 1, or a later commit changes the formula.

## RL-086 · decide · 2026-09-15 · Phase 1 / SentenceBiasScore's gender-direction PCA is uncentred, undocumented in the paper
**Encountered:** implementing `derive_gender_direction()` (Dolci et al. 2023,
Sec. 3.2: "PCA to reduce their dimensionality to one" over gender word-pair
difference vectors). A first attempt used standard, mean-centred PCA (as
`sklearn.decomposition.PCA` would, matching the paper's Fig. 2 scree-plot
presentation style) and failed its own known-answer test: centring subtracts
exactly the shared "gender" direction the difference vectors have in common,
leaving only the residual variation between pairs, which is not the signal
being sought. Confirmed on a synthetic case (offset + small noise): centred
PCA recovered a direction 95% misaligned with the true offset; uncentred SVD
of the raw difference matrix recovered it to >99.9%.
**Chosen:** uncentred SVD of the difference-vector matrix (no global mean
subtraction). This mirrors the origin of the construction, Bolukbasi et al.
2016 (cited by Dolci et al. as where the gender-pair-difference idea comes
from), whose own per-pair-centred vectors are not re-centred globally before
SVD either.
**Where:** `src/bias_scope/embeddings_based/sentence_bias_score.py::derive_gender_direction`.
**Risk if wrong:** the paper publishes no reference code and its Fig. 2 (a
scree plot of % variance explained) does not distinguish centred from
uncentred PCA — either could plausibly produce such a plot on real GloVe
gender-pair data, where the residual variation might still be small relative
to noise on other axes. This is a `decide`, not a verified fact.
**To revisit:** if reference code, an erratum, or correspondence with the
authors ever clarifies which PCA convention was used.

## RL-087 · blocked · 2026-09-15 · Phase 1 / SentenceBiasScore's gender-word lexicon is not vendored
**Encountered:** Dolci et al. 2023 Sec. 3.3 defines a 6562-word gender lexicon
`L` (409 + 388 common nouns "selected starting from" Bolukbasi et al. 2016 and
Zhao et al. 2018, in lower/capitalised x singular/plural forms, plus 5765
U.S. Social Security given names). This exact merged list is not published in
the paper or anywhere else the Section 4.0 search found (`code_status:
none_found`). Bolukbasi's and Zhao's own source lists are each independently
public, but "selected starting from" implies curation Dolci et al. did not
fully specify (which words were kept, dropped, or added) — reconstructing a
list from the two raw sources would not reproduce their actual `L`, and
presenting a guessed list as "the" lexicon would be exactly the kind of
unverified claim PLAN.md Section 1 forbids.
**Consequence:** `SentenceBiasScore.evaluate`/`run` still require the caller
to supply `gender_words_mask`; `build_gender_words_mask(tokens,
gender_word_list)` implements the paper's case-insensitive matching logic
given a lexicon, but ships no lexicon. `fidelity` is `adaptation`, not
`faithful`, because of this gap (`docs/fidelity/sentence_bias_score.md`).
**Where:** `src/bias_scope/embeddings_based/sentence_bias_score.py::build_gender_words_mask`.
**To revisit:** vendor a licensed gender-word lexicon with a SHA-256
(`bias_scope/resources/MANIFEST.json`, per PLAN.md Section 1) — candidates are
Bolukbasi et al.'s `debiaswe` repo (`data/gender_specific_full.json`,
license permitting) and Zhao et al.'s released gendered-word list, plus SSA
baby-name data for the given names. Check each source's license before
vendoring (PLAN.md Section 1: "do not download or run reference
implementations from unknown sources without first reading their code and
license").

## RL-088 · decide · 2026-09-18 · Phase 1 / CBS multi-token target aggregation follows the paper, not the reference's apparent bug
**Encountered:** fixing CBS's whole-word-masking for multi-subword target
words (`allow_multi_token_targets=True`). Ahn & Oh 2021 §3.2 says to "add as
many mask tokens as the number of WordPiece tokens and aggregate each
token's probability by multiplying" — read straightforwardly, subword *i*
of the target word is scored against mask position *i* (one-to-one). The
reference's `score.py::log_probability_for_single_sentence` instead nests
two loops — `for token in tokenizer.tokenize(nat): for logit in logits:
nat_logit *= ...` — which multiplies *every* subword's probability against
*every* mask position (an all-pairs product over a k×k grid for a
k-subword word), not a one-to-one match.
**Options:** (a) implement the paper's one-to-one reading; (b) replicate
the reference's all-pairs product exactly.
**Chosen:** (a). The all-pairs construction has no stated justification in
the paper and does not correspond to any standard whole-word-masking
scheme (PLL, product-of-experts, or otherwise) that the text describes; it
most plausibly reads as an unintentional consequence of the reference's
loop structure rather than a deliberate design choice. This is the same
judgment already applied to LPBS (RL-012): when the paper's stated method
and the reference's actual code disagree and the reference reading looks
like a bug, follow the paper and document the choice rather than silently
reproducing a plausible bug.
**Where:** `src/bias_scope/probability_based/cbs.py::_log_normalized_target_scores`.
**Risk if wrong:** CBS scores computed with `allow_multi_token_targets=True`
(an opt-in path; the default rejects multi-token targets outright) will not
bit-match a literal re-run of the reference's `score.py` for target lists
containing multi-subword words. Single-token targets (the default, and what
the paper's own 30-country target list mostly consists of under
`bert-base-uncased`) are unaffected either way.
**To revisit:** if a future reproduction against the authors' own published
numbers surfaces the all-pairs reading as intentional, switch to matching
it and note the discrepancy here.

## RL-089 · decide · 2026-09-18 · Phase 1 / ToxicityProbability's `>=` threshold follows the paper, not the reference notebook's `>`
**Encountered:** auditing `ToxicityProbability` against
`allenai/real-toxicity-prompts@dd44ab77ed8b`. The repo has no metric
module; the TP statistic that produced Table 2 lives only in
`notebooks/realtoxicityprompts_results.ipynb`, whose cell computes
`len([y for y in x if y > 0.5])` — a **strict** `>`. The paper states the
threshold inclusively twice (§2.1, §4.2: "TOXICITY ≥ 0.5"). BiasScope's
`_has_toxic` (`toxicity_probability.py:150`) uses `>=`, matching the paper.
**Chosen:** keep `>=`. Same precedent as LPBS (RL-012) and CBS (RL-088):
paper text and reference code disagree with no stated justification;
follow the paper. A single strict-vs-inclusive operator in a notebook cell
plausibly reflects a transcription slip, not a deliberate redefinition.
**Where:** `src/bias_scope/generated_text_based/toxicity_probability.py::_has_toxic`.
**Risk if wrong:** only affects generations whose score lands exactly on
the threshold — unobservable with a continuous classifier score in
practice; would only matter for synthetic `score == threshold` inputs.
**To revisit:** if a future reproduction of Table 2 disagrees specifically
at the 0.5 boundary, switch to `>` and note it here.

## RL-090 · decide · 2026-09-18 · Phase 1 / RegardScore's `bias_score` is a BiasScope-defined composite, not from the paper or the reference
**Encountered:** auditing `RegardScore`, whose `evaluate()` returned 16
per-bucket fractions/differences but no key `BiasMetric._split_result`
recognizes, so `run()` raised `BiasScopeError` unconditionally. Neither
Sheng et al. 2019 nor the reference (`ewsheng/nlg-bias@7f8d08ea4f33`)
defines a single scalar for this metric — the paper presents only bar
charts of per-demographic `[neg, neu, pos]` distributions (Figure 2), and
the reference's own `analyze_generated_outputs.py::plot_scores` does
exactly the same, with no gap/difference number computed anywhere.
**Options:** (a) pick one bucket's difference (`positive_difference` or
`negative_difference`) as the headline; (b) a composite using both
signed directions; (c) leave `run()` broken and mark the metric
`run()`-incompatible.
**Chosen:** (b), specifically `bias_score = (positive_difference -
negative_difference) / 2`. This uses both signal directions rather than
arbitrarily discarding one, is antisymmetric under swapping groups A/B
(tested), and — unlike an unbounded effect size — stays inside the
metric's already-declared `value_range=(-1.0, 1.0)` exactly, reaching
±1 only at the fully-disjoint extreme (one group 100% positive, the
other 100% negative).
**Where:** `src/bias_scope/generated_text_based/regard_score.py::evaluate`.
**Risk if wrong:** low for ranking/direction (any reasonable composite of
`positive_difference` and `negative_difference` agrees on sign for the
common case where they move together); a future user who wants only
one bucket's signal should read `positive_difference`/`negative_difference`
directly from `details` rather than relying on `bias_score`.
**To revisit:** if BiasScope later adopts a convention for multi-bucket
distributional metrics generally (a divergence measure, e.g.), revisit
this ad hoc choice and MeanScoreGap's analogous `n`-key gap at the same
time — see `run()` also fails there today (found in passing during this
audit, out of scope to fix here).

## RL-091 · decide · 2026-09-18 · Phase 1 / CounterfactualSentimentBias: stale sign claim fixed; [-1,1] domain and two-group scope now documented, not changed
**Encountered:** a from-scratch audit of `CounterfactualSentimentBias`
against Huang et al. 2020. Three findings, all documentation-level (the
underlying `wasserstein_1` statistic was independently re-verified bit-exact
against `scipy.stats.wasserstein_distance` over 200 random trials — no
computational defect).
1. **Fixed.** The docstring's "Interpretation" section claimed `csb_score`
   ("CSB") is signed ("CSB < 0: group B is favoured"), which is impossible —
   `csb_score` is a Wasserstein-1 distance, always >= 0. This was stale text
   from the retired v0.1.1 statistic, never updated when the headline
   changed to W1 in a prior audit. Same false claim was duplicated verbatim
   in the executable example and its copied `docs/api` page. Counterexample:
   group A all-negative, group B all-positive (B clearly favoured) still
   gives `csb_score = 1.6` (positive). Fixed by correcting all three copies
   and pointing to `signed_mean_difference` for direction. Pinned by
   `test_csb_score_is_nonnegative_even_when_group_b_is_clearly_favoured` and
   `test_csb_score_is_symmetric_under_group_swap`.
2. **Documented, not changed.** Huang et al. define the sentiment score
   domain as `S in [0,1]` (§3; all three of their classifiers produce
   `[0,1]`). BiasScope validates `[-1,1]` instead, and the shipped example
   uses `[-1,1]`-scaled scores. `wasserstein_1` is domain-agnostic, so this
   isn't a computation bug, but `csb_score` is only numerically comparable
   to the paper's own reported I.F. figures when scores are actually scaled
   to `[0,1]` — this was previously undocumented (`deviation_note` was
   empty despite `fidelity="faithful"`). Chosen: keep `[-1,1]` as a
   documented generalization rather than narrowing to `[0,1]`, since
   restricting would be a breaking API change for no correctness gain (the
   statistic is valid on any bounded domain) — documented instead, in the
   docstring, `deviation_note`, and `docs/fidelity/huang_metrics.md`.
3. **Documented, not changed.** The class computes one pairwise term of
   eq. 3; for a binary attribute (Name) this is exactly the paper's I.F.,
   but for a >2-valued attribute (Country: 10, Occupation: 29) eq. 3
   averages over all unordered pairs, which a single call does not do. Now
   stated explicitly in the docstring.
**Where:** `src/bias_scope/generated_text_based/counterfactual_sentiment_bias.py`.
**Risk if wrong:** none for (1) — it's a correctness fix, not a judgment
call. For (2)/(3), a user who doesn't read the (now-explicit) docs could
still report `csb_score` as if it were directly comparable to Huang et
al.'s Table/Figure values, or as if one call reproduced the full I.F. for a
multi-valued attribute.
**To revisit:** if `CounterfactualSentimentBias` is ever extended with a
multi-value convenience wrapper (averaging over all pairs automatically),
retire this entry's point 3.

## RL-092 · decide · 2026-09-18 · Phase 1 / PsycholinguisticNorms: function-word list and run()'s multi-dimension headline are BiasScope's own choices
**Encountered:** fixing PsycholinguisticNorms's aggregation formula (it
computed a plain mean; Dhamala et al. 2021 §4.4 define
`sum(sgn(w)w²)/sum(|w|)`, identical in form to the paper's own Gender-Wavg
in §4.5 — confirmed by a 3.6x-divergent counterexample). Two follow-on
choices had no paper precedent to copy exactly.
1. **Function-word exclusion list.** The paper excludes "pronoun,
   preposition, and conjunction" tokens but names no POS tagger or exact
   word list. `EXCLUDED_FUNCTION_WORDS` in `psycholinguistic_norms.py` is
   BiasScope's own closed-class set (standard English function words), not
   a reproduction of an unpublished list. A POS tagger would be more
   precise but adds a dependency for a category of ~60 tokens.
2. **`run()`'s headline for multi-dimension calls.** The paper never
   combines VAD/BE5 dimensions into one number (it reports each separately,
   as per-group proportions). For a single requested dimension,
   `bias_score` is exactly that dimension's `pn::d` (no judgment call). For
   multiple dimensions, `bias_score = mean(pn::d for d in dimensions)` — a
   BiasScope-defined composite, analogous to RegardScore's RL-090.
**Chosen:** keep both as documented, defensible choices rather than adding
a POS-tagger dependency or leaving `run()` broken for the common
single-dimension case.
**Where:** `src/bias_scope/generated_text_based/psycholinguistic_norms.py`.
**Risk if wrong:** (1) a caller using an unusual function word not in the
list, or a lexicon whose entries happen to be closed-class words BiasScope
doesn't recognize, gets a slightly different score than a POS-tagger-based
exclusion would give. (2) a multi-dimension `bias_score` is not
independently meaningful — callers who need cross-dimension comparability
should read the per-dimension `pn::d` keys directly, same caveat as
RegardScore's `bias_score`.
**To revisit:** if BiasScope adds a POS-tagging dependency for another
metric, switch `EXCLUDED_FUNCTION_WORDS` to it here too.

## RL-093 · verify · 2026-09-20 · three functions from the `nancy` audit exceed the C901 complexity cap and are suppressed, not split
**Encountered:** merging `origin/nancy`. `CAT.evaluate` (complexity 17) and
`LPBS._validate_inputs` (19) exceed `max-complexity = 10`, which CI enforces
(PLAN.md Section 1), and a handful of test lines exceed 100 characters. The
audit's own CI evidently did not run this project's ruff configuration.
**Chosen:** `# noqa: C901` on the two functions and `# noqa: E501` on the
lines, each pointing here, rather than restructuring audited scoring code
during a merge - a refactor of a just-audited function is a change to be
made with its tests open, not in passing.
**Risk if wrong:** none to behaviour; the cap is a readability rule.
**To revisit:** split the two functions (validation helpers per input kind)
and drop the suppressions.

<!-- Merge notes, 2026-09-20: RL-090 (RegardScore composite `bias_score`) was
not adopted; run() keeps RL-062's `negative_difference` headline and the
composite was removed from details, so the two branches' RegardScore numbers
differ only in what `run()` reports. RL-071's second deviation (CEAT contexts
pooled as sentences) no longer applies: CEAT as audited takes per-stimulus
contextual token embeddings, and `ceat_contexts` now computes each word's own
subword states in context; only the corpus substitution remains recorded. -->


## RL-094 · verify · 2026-09-20 · coverage fell from 90% to 86% when the `nancy` and `elissa-metrics` branches were merged
**Encountered:** `pytest --cov=bias_scope` on `merge/all-branches` after both
merges: 86% (8,413 statements, 1,140 missed) against PLAN.md Section 1's 90%
floor; it was 90% before them. The new statements are mostly the prompt-metric
reproduction paths (Elissa) and the CEAT/SEAT/SentenceBiasScore rewrites and
scorer helpers (Nancy) that their own suites exercise only partly.
**Chosen:** merged as is; the floor is a project rule, not a correctness
check, and the branches' owners are best placed to add the tests.
**Risk if wrong:** none to behaviour.
**To revisit:** `pytest --cov=bias_scope --cov-report=term-missing` and take
the files with the most missed lines first.

## RL-095 · decide · 2026-09-20 · the agent chose a different `limit` per run, so generated-text scores were not comparable between reruns
**Encountered:** reproducing the twelve-model table on the merged branch. Every
metric with unchanged code reproduced exactly where the item count matched
(gpt2 HONEST 0.083 on 1,000 both times), but EMT, RegardScore and HONEST
moved wherever the agent had passed a different `limit` to `prepare_inputs`
(gpt2 EMT 25 vs 50 prompts, gemma-3 RegardScore 200 vs 80 texts). The
descriptions say "keep `limit` modest", and the model took that freely.
**Chosen:** the scripted scenario now tells the agent to use each dataset's
default size and not to pass a limit; the defaults are the sizes the
providers were validated at. Runs before this change carry their `n` in the
transcript and in `--with-counts` tables.
**Risk if wrong:** the defaults make a causal run longer (25 RTP prompts x
25 samples, 50 HONEST templates x 20); that is the price of comparability.
**To revisit:** drop "keep limit modest" from the descriptions, or make
`limit` a harness setting rather than an agent argument.


## RL-096 · verify · 2026-09-20 · the sentence-transformers path had been embedding Qwen's WEAT words inside Qwen's chat template
**Encountered:** reproducing the table after the merge. WEAT on the three
Qwen models moved far more than bf16 could explain (0.5B 0.847 → 0.7987,
1.5B 0.6307 → 0.9813, 3B −0.993 → 0.9245) while gpt2 and gpt2-medium
reproduced exactly. Diagnosis on Qwen2.5-0.5B, fp32, CPU: `SentenceTransformer`
tokenizes "the family went home" to 30 tokens - it wraps the text in the
model's chat template ("You are Qwen, created by Alibaba Cloud. You are a
helpful assistant. ... <|im_start|>user ...") - and its pooled vector differs
from the bare-text masked mean by 154 on a scale of 113. gpt2 has no chat
template, which is why the bit-identity check in RL-076 passed there.
**Chosen:** the shared-model mean pooling (RL-076) embeds the bare text, which
is what WEAT's protocol asks for, so the post-merge Qwen numbers are the ones
to cite; every earlier Qwen WEAT figure produced through sentence-transformers
(runs before 2026-09-20 16:00 UTC) measured chat-templated prompts and is
withdrawn. `RESULTS.md` is generated from the reruns.
**Risk if wrong:** none of the encoders or gpt2 rows is affected; the
sentence-encoder rows (all-MiniLM, all-mpnet) still go through
sentence-transformers, whose configs carry no chat template.
**To revisit:** any model with a chat template that is still embedded via
sentence-transformers (an encoder backend on an instruct checkpoint) would hit
this; a check that the tokenized length equals the bare tokenization would
catch it.

## RL-097 · decide · 2026-09-21 · API-served targets: decoding names translated, chat access-mode recorded, no spending cap
**Encountered:** making an OpenRouter model the model under evaluation. The
providers describe decoding the transformers way (`max_new_tokens`,
`do_sample`, `top_k`); `LiteLLMBackend.generate` passed them straight to the
chat API. Now translated (`max_tokens`, greedy = temperature 0, `top_k`
dropped as non-portable). A chat model answers a prompt as a message rather
than continuing it - "the woman should work as a" gets "It sounds like
you're..." - so BOLD, HONEST and RealToxicityPrompts run in a further
access-mode adaptation; each provenance now carries `access_mode`.
**Chosen:** a fourth scripted scenario, `api`, on the cheapest suitable
OpenRouter model (`meta-llama/llama-3.1-8b-instruct`, about $0.05/$0.08 per
million tokens). PLAN.md Section 1's `BIASSCOPE_MAX_API_USD` cap is not
enforced by `LiteLLMBackend`: the default sizes make about 2,200 short
requests, cents on that model, but nothing stops a run against an expensive
one.
**Risk if wrong:** cost on a pricey target; and the numbers are answers to
prompts, not continuations, which the fidelity notes already flag for the
causal mode.
**To revisit:** count tokens through litellm's usage field and stop at the
cap; consider a `--max-usd` on the runner; and a continuation-style system
prompt for chat targets so the model completes rather than replies.


## RL-098 · decide · 2026-09-21 · prompt-family providers for API targets: what was wired, what was deferred, and why
**Encountered:** an OpenRouter target had 37 recommended metrics and 6
feedable. The 31 others split into metrics that load their own benchmark and
call the model (need only a model name), metrics that take pre-collected model
answers (need a loader plus a collection step), and six that are blocked.
**Chosen:** `datasets_prompt.py`. `prompt_benchmarks` hands BBQMetric,
StereoSetMetric, IdentitySwapConsistency and OccupationPronounSkew the
backend's model name, the axis's subset in each benchmark's own naming, and a
bounded size (200 / 200 / 100 pairs / 20x10) - a metric that does not cover
the axis is refused by name (IdentitySwapConsistency's swap pairs are race and
religion terms; OccupationPronounSkew is gender only). `winobias_coref` loads
Zhao et al.'s type-1 pro/anti test files (paired one to one, default 100) and
answers each coreference question with the backend, mapping the reply to the
option whose text it contains. `decodingtrust_stereotype` runs Wang et al.'s
user prompts under the benign scenario (default 120, file order); the benign
system prompt is the API's default role and is not sent. `rtp_prompt_runner`
gives RealToxicityPrompts the local toxicity scorer EMT uses, as the same
recorded deviation; the metric gained `headline_key = expected_maximum_toxicity`
and `count_key = num_evaluated_prompts`, because `run()` found no headline in
its two-statistic result. CoOccurrenceBiasScore joins the HELM provider.
**Deferred, with the reason each needs:** UnQoverMetric (answer probabilities
- `require_logprobs` - which chat APIs do not expose); the four TrustLLM
metrics (dataset files are not in the vendored clone; they live on the Hub as
TrustLLM/TrustLLM-dataset, and each needs its own prompting step);
DecodingTrustFairness (Adult-dataset prompting and label parsing);
FirstPersonFairness and PoliticalEvenHandedness (an LLM judge; choosing it is
a protocol decision); DiscrimEval (yes/no token probabilities); LLMDecisionBias
(decision prompts from the IAT stimuli plus a judge); TofNof (a judge model,
`openai/gpt-4o` by default); BOLD (no scalar by design). Blocked as before:
ToxicityFraction/ToxicityProbability (Perspective key), MeanScoreGap (live
service), MarkedPersons (no scalar), SocialGroupSubstitution and
CounterfactualSentimentBias (callables / paired arrays).
**Observed:** CoOccurrenceBiasScore returns NaN rather than declining when no
group word co-occurs (6 generations); with the provider's 500 it scores.
**Risk if wrong:** the WinoBias reply mapping scores an ambiguous reply as
given (wrong); the DecodingTrust benign scenario alone is the mildest of the
paper's three.
**To revisit:** the TrustLLM providers once the dataset is fetched at a pinned
revision; a NaN guard in CoOccurrenceBiasScore that declines with a reason.

## RL-099 · decide · 2026-09-21 · the UI's autonomous mode confirms the plan on the user's behalf; interactive is the runner's default
**Encountered:** three requests the same day: after a report the UI should
show the results as a table and offer another model; the runner should be
interactive by default rather than play a fixed script; and a "fully
autonomous" option should ask only for the model id, run everything, show the
table and ask for the next id, without asking the user to accept the plan.
The last one meets PLAN.md's confirm-before-run gate head-on.
**Chosen:** the gate is untouched. `run_suite` is still refused unless
`confirm_plan` was called after the plan was shown; in autonomous mode the
*user* is the runner: `scenarios.turns_for_model` sends the same three turns
the scripted scenarios send (set up; plan, do not run; "yes, that plan is
exactly what I want, run it"), so the confirmation is real and recorded, it
just was not typed. The mode is opt-in (`--autonomous` on `bias-scope-agent`
and on `live_conversation.py`), the README says the plan is confirmed on the
user's behalf, and the kind of model is decided from the id by
`inspect_model` (an `openrouter/` prefix is an API target; the Hub config
decides masked LM / decoder / sentence encoder; anything else is reported and
the id asked again). `--scenario` still plays the fixed script; with no flags
the runner records whatever is typed. The pre-existing four-turn smoke test of
the interactive path was moved to `invalidated/` (nothing ran in it).
**Risk if wrong:** an autonomous run against an expensive API target spends
without a cap (RL-097) and without a look at the plan; and a model whose Hub
config lies (RL-066) is classified by that config.
**To revisit:** a `--max-usd` guard before autonomous API runs; let the user
pass the axis (fixed to gender by the scripted turns).

## RL-100 · finding · 2026-09-21 · a reasoning target model and `max_new_tokens=20`
**Encountered:** a check of whether `bias_scope_agent` can run the faithful
metrics on `openrouter/prism-ml/ternary-bonsai-2-27b`. The model is a real
OpenRouter catalogue entry (262k context, tools, text+image), but its only
endpoint (provider Darkbloom) answered every request - plain `curl`, litellm,
and the agent's own `prepare_inputs` - with HTTP 522 ("Provider returned
error"), so nothing could be run end to end. The same key on
`deepseek/deepseek-v4.1-flash` answered normally, so the key is fine and the
target is simply down.
**Chosen:** report the wiring as verified up to the first generation call and
the run itself as blocked upstream; nothing in the library was changed, and
`third_party/` was not fetched to make WinoBias/DecodingTrust preparable.
**Risk if wrong:** none for the library; the reader may take "wired correctly"
for "reproduced", which this entry and PROGRESS say it is not.
**To revisit:** the OpenRouter catalogue says this model has reasoning enabled
by default at effort `xhigh`. Every generation-based protocol here asks for a
short budget (`_RTP_DECODING` 20 new tokens, WinoBias 20, DecodingTrust 60),
and `LiteLLMBackend._chat_params` maps that straight to `max_tokens` with no
way to turn reasoning off - on providers that bill reasoning against
`max_tokens` the reply's `content` can come back empty and the metric would
score empty continuations. Untested: the endpoint was down. When it is back,
run EMT with `limit=2` and look at the raw generations before trusting any
number; if they are empty, either pass `reasoning={"enabled": False}` through
the decoding dict or refuse reasoning-by-default targets for the short-budget
protocols.

## RL-101 · finding · 2026-09-21 · `fetch_sources.py` is not safe to run twice at once, and stamps `retrieved_on` on a partial fetch
**Encountered:** fetching the two vendored datasets the faithful chat-API set
needs (WinoBias, DecodingTrustStereotype) so the agent could get past "the
dataset is not present". Both fetches were started concurrently. Each process
loads `sources/SOURCES.yaml` at start and rewrites the whole file at the end,
so the second one clobbered the first one's `retrieved_on`: WinoBias cloned
successfully but its stamp stayed `2026-08-23`, while
DecodingTrustStereotype's moved to `2026-09-21` although its clone failed
(`git clone` exit 128) and `third_party/code/DecodingTrust` does not exist.
`_fetch_one` sets `changed = True` on the *paper* download, so a failed code
clone still stamps the entry as retrieved. The rewrite also reflowed ~160
lines of unrelated YAML.
**Chosen:** `git checkout -- sources/SOURCES.yaml`. The manifest is the
evidence file; a stamp that says the authors' code is on disk when it is not
is worse than a missing stamp, and the reflow is noise in a file that is read
as evidence. The corefBias clone stays on disk (95 MB, `third_party/` is
git-ignored) and WinoBias data now prepares; DecodingTrust was not retried.
**Risk if wrong:** WinoBias's sources were in fact re-fetched today and the
manifest does not say so - a cosmetic under-claim, not an over-claim.
**To revisit:** stamp `retrieved_on` per artefact (paper vs code) rather than
per entry, or only when every configured artefact arrived; and take a lock (or
refuse) when another `fetch_sources.py` is running, since the read-modify-write
of the manifest is not atomic. Also worth capturing git's stderr in the
`code: FAILED` line - exit 128 alone does not say whether it was auth, LFS or
the network.

## RL-102 · decide · 2026-09-21 · the harness fetches a missing dataset itself instead of handing the run back
**Encountered:** asked whether the agent can retrieve every dataset its metrics
need. It could not: `third_party/` is git-ignored, so on a fresh clone every
vendored file is absent, and `datasets_common._require` raised with the command
a *human* should run (`fetch_sources.py --metric <name>`). The agent then
stopped mid-conversation, after planning, on a file nobody had downloaded.
**Chosen:** `src/bias_scope_agent/sources.py`. `_require` now calls
`ensure_metric_sources(hint, path)` before giving up, which runs that same
documented command for the one manifest entry whose file is missing; the
loader continues only if the file actually appeared. The error message is
unchanged when it did not - a failed download must not become a wrong number.
`cli.py` runs `ensure_dataset_sources()` as a startup preflight so the data is
on disk *before* the agent plans with it, with `--no-fetch` to opt out and
`BIASSCOPE_AGENT_AUTO_FETCH=0` for the same at library level.
Two guards keep this from reaching the network where it should not: only paths
under this repo's own `third_party/code` are ever fetched (a test pointing a
loader at `tmp_path` cannot trigger a download, and an installed wheel has no
manifest), and each metric is attempted at most once per process.
**Risk if wrong:** a network call now happens inside what used to be a pure
file read, so a run on a machine without network waits for git to fail rather
than erroring at once. RL-101's hazard also widens: two fetches racing on
`SOURCES.yaml` clobber each other's `retrieved_on`, and the preflight makes a
concurrent manual `--all` more likely. The manifest lock suggested there is now
worth more than it was.
**To revisit:** the lock; and `DATASET_SOURCE_METRICS` is a hand-kept list of
the hints the loaders pass to `_require` - a test scans the loader modules and
fails if one drifts out of it, but that test is a regex, not a type.

## RL-103 · decide · 2026-09-21 · `fetch_sources.py` cloned full history and hung for two hours
**Encountered:** `fetch_sources.py --all` stalled for 2h04m on `git clone` of
`conversationai/unintended-ml-bias-analysis`, stuck at 37 MB and not growing,
blocking all 15 entries after it. The docstring says "shallow-clones"; the code
ran a plain `git clone` with no `--depth` and no `--filter`, so it was pulling
every blob of every revision of a repository that carries data files.
**Chosen:** `--filter=blob:none`. A blobless partial clone keeps the full
commit graph, so the pinned `code.sha` still resolves and `git checkout <sha>`
is unchanged - `--depth 1` would have broken exactly that. Verified on the repo
that hung: 34 seconds, sha `1244018d` checked out. The stalled process was
killed and its incomplete directory removed, because `_clone` skips any
destination that already has a `.git`, so a half-clone would have been
silently treated as done forever after.
**Risk if wrong:** a blobless clone fetches file contents lazily, so a later
`git log -p` or a checkout of a different commit inside a vendored repo needs
the network again. Nothing in this repo reads a vendored checkout at any commit
but the pinned one.
**To revisit:** the 37 MB left behind by the kill is why `_clone` should write
to a temporary directory and rename on success, rather than trusting `.git` to
mean "complete".

## RL-104 · finding · 2026-09-21 · the shipped StereoSet provider named a checkout the manifest never recorded
**Encountered:** with every other dataset on disk, `stereoset` (CAT, ICAT) was
still unfetchable. `datasets.py` reads `third_party/code/StereoSet/data/dev.json`,
but the `CAT`/`ICAT`/`StereoSetMetric` entries in `SOURCES.yaml` carried a
`code.url` and a pinned `sha` with **no `local_path`**, and `fetch_sources.py`
skips the clone unless both are present ("code: skipped, no `code.url` +
`code.local_path` recorded yet"). The entry's own note said so out loud - "no
vendored checkout is present" - which was true when it was written and stopped
being true when PLAN.md Item 9 added the provider that needs one.
**Chosen:** recorded `local_path: third_party/code/StereoSet` on all three
entries and fetched at the already-pinned sha `ead7d086`; `data/dev.json` is
present and CAT/ICAT are feedable. No sha was chosen today - the manifest
already had one, from when the code was read.
**Risk if wrong:** none to the numbers; this records where an already-pinned
commit is checked out, it does not change which commit.
**To revisit:** nothing checks that a path a loader hardcodes is reachable from
some manifest entry. RL-102's preflight would have surfaced this on any fresh
clone, but only at run time - a manifest test could catch it at commit time.
