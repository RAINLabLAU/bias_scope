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

## RL-021 · decide · 2026-08-23 · Phase 1 / CEAT's default sample count is far below the paper's
**Encountered:** `CEAT.evaluate(n_samples=100)` by default, while Guo & Caliskan
report CES at **N = 1,000 and N = 10,000** sampled contexts. CES is a weighted
mean over those samples, so N controls its variance directly: at N = 100 the
estimate is materially noisier than anything the paper reports, and two runs at
different seeds can differ enough to change a qualitative reading.
**Where:** `src/bias_scope/embeddings_based/ceat.py:100`.
**Chosen (provisional):** left at 100 for now and documented in
`docs/fidelity/ceat.md`, because raising it changes every existing CEAT number
and the audit had no mandate to move results. Flagged rather than acted on.
**Risk if wrong:** any CEAT value produced at the default is not comparable to
the paper's, and nothing in the output currently says so.
**To revisit:** before Phase 3 Tier 1. Either raise the default to 1,000, or
make `n_samples` required with no default so the caller must choose. The second
is more honest and is a one-line change; it is breaking, so it needs a CHANGELOG
entry.

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

## RL-029 · blocked · 2026-08-23 · Phase 1 / SentenceBiasScore's paper could not be obtained
**Encountered:** Dolci, Azzalini & Tanelli 2023, *Data Science and Engineering*
8(2) is behind Springer's paywall and no preprint exists — PLAN.md Appendix E
already flagged "DOI via Springer; check for preprint", and the Section 4.0
search found none. This is the **only** metric in the library whose source could
not be read.
**Consequence:** `SentenceBiasScore` keeps `fidelity: unaudited` and its
`SOURCES.yaml` status stays `pending`. Section 4.0 forbids assigning a status
without reading the paper, and guessing would be exactly the unverified claim
v0.2 exists to eliminate.
**Where:** `docs/fidelity/sentence_bias_score.md`.
**To revisit:** obtain the paper — institutional access, or email the
corresponding author for the accepted manuscript, which usually works. If
neither succeeds, the honest resolution is to **relabel the class `original` and
drop the citation**: an unverifiable citation is worse than none. That is the
maintainer's call.
**Meanwhile:** the Tier-3 metamorphic properties can be run without the paper
and should be, since they are the only objective evidence available for this
metric.

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

## RL-040 · decide · 2026-09-12 · bias_scope_agent / the confirm-before-run gate is structural, not semantic
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
**To revisit:** if false-positive confirmations are observed in practice, a
review would have to weigh a second check against the project's explicit
no-second-reviewer-LLM constraint.

## RL-041 · verify · 2026-09-12 · bias_scope / CrowSPairs and AUL cannot be run via .run()
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

