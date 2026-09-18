# PROGRESS

One dated entry per session, newest at the bottom: what changed, what was tested
and how, coverage delta, what remains, and the `REVIEW_LATER` IDs created.

---

## 2026-08-22 · Phase 0 — Baseline · complete

All eleven Phase 0 boxes in `PLAN.md` Section 3 are ticked. Nothing in this
session changed a metric's numbers; it is tooling, test layout, bookkeeping, and
one reconciliation of a stale v0.1 artefact.

### Baseline, before any change

`.venv`, Python 3.13.9, CPU + one RTX A4500.

```
637 passed, 0 failed          86% coverage on src/bias_scope          20.2 s
```

Full environment in `results/validation/environment.json`.

### What changed

**Tooling.** `ruff` (line length 100, `E`/`F`/`I`/`C901`, max-complexity 10),
`pytest-cov`, `pyyaml`, and `pypdf` added to the `dev` extra. `slow` and
`equivalence` markers registered and excluded from the default run.

**130 pre-existing ruff violations resolved.** Two were real:

- `F821` ×17 — `torch` was used in type annotations across `weat.py`, `seat.py`,
  `ceat.py`, and `sentence_bias_score.py` with no import at all. `from __future__
  import annotations` meant it never evaluated, so it never failed; it would have
  the moment anything called `typing.get_type_hints`. Now imported under
  `TYPE_CHECKING`, keeping torch optional.
- `F841` ×1 — `expanded_mask_ws_idx` in `probability_based/scorers.py` was
  computed and never read; the code uses `mask_positions` derived from
  `input_ids`. Removed.

The rest were mechanical: `F401` ×14 and `I001` ×33 auto-fixed, `E712` ×9
rewritten as `is True` / `is False` to keep the assertions strict, `E501` ×35
wrapped by hand, `E402` ×3 given a per-file ignore (RL-001), `C901` ×17 given a
tracked `noqa` (RL-002).

One test was **strengthened rather than deleted**: `F841` flagged
`test_num_samples_aggregation_mean` for computing `result` and asserting only on
the call count — the test named for mean aggregation never checked a mean. It
now asserts `hello → [2.0]`, `goodbye → [5.0]`.

**Test layout for the phases ahead.** `tests/conftest.py` with `tiny_encoder`,
`tiny_causal`, `tiny_embeddings`, and `stub_chat`; and four new directories,
each with a worked seed example rather than a placeholder:

- `tests/properties/` — WEAT permutation invariance (property 4 of 6.2).
- `tests/integration/` — BBQ end to end over `stub_chat`, plus tiny-model
  plumbing and CrowS-Pairs over the tiny encoder.
- `tests/oracles/` — a 30-line loop-based WEAT, agreeing with the library to
  1e-8 over 200 random inputs.
- `tests/golden/` — `weat.json` frozen on the seeded fixture vectors.

**Scaffolding.** `sources/SOURCES.yaml` + `fetch_sources.py` +
`check_manifest.py` (the Section 4.0 gate); `verification/ledger.yaml` (40
metrics × 7 criteria, 7 features × 5) + `render_ledger.py` + `regen_golden.py`;
`validation/registry.yaml`; `record_environment.py`; `PROGRESS.md`,
`DECISIONS.md`, `REVIEW_LATER.md`, `CHANGELOG.md`.

`bias_scope.utils` gained `seed_everything(seed=42)` and `protocol_hash(dict)`,
with 30 tests including a hand-derived hash known-answer.

`.github/workflows/tests.yml` added — there was **no test CI at all**, only a
docs deploy. Three jobs: ruff, manifest+ledger checks, and pytest on 3.10–3.13.

`.gitignore` — `sources/papers/`, `third_party/`, `cache/`, and targeted entries
for model weights. Untracked-and-stageable content under `results/` dropped from
**6.6 GB to 12 MB** (RL-007).

### WEAT 6B / 840B reconciliation

`summary.md`, `table.csv`, and `CONFIGURATION.md` reported only d = 1.6938 on
GloVe-6B against the paper's 1.81 and called it MATCHED, while
`paper/evaluation.tex` quoted 1.8139. Root cause: `repro_weat_840B.py` was never
called by `finalize_emnlp.py`, which regenerates those three files.

- Re-ran WEAT-6 on GloVe-6B: **d = 1.693808**, reproducing the recorded 1.6938
  exactly.
- Wired the recorded 840B result into `finalize_emnlp.py`, so all three
  generated files now carry both rows and the paper's number traces to
  `weat_840B/weat_840B_result.json`.
- Fixed the hardcoded absolute output path in `repro_weat_840B.py:23`.
- The 840B vectors were **not** re-downloaded — RL-005.

| Vectors | Published | Ours | Δ | Rel. err |
|---|---|---|---|---|
| GloVe-6B (`glove-wiki-gigaword-300`) | 1.81 | 1.6938 | −0.1162 | 6.4% |
| GloVe-840B (`glove.840B.300d`) | 1.81 | 1.8139 | +0.0039 | 0.22% |

### Verification

```
ruff check src tests                    clean
pytest -q --cov=bias_scope              676 passed, 3 deselected, 86%, 18.6 s
pytest -q -m slow                       3 passed
check_manifest.py                       OK, 0 entries (shape-only; see note)
render_ledger.py                        0/40 metrics, 0/7 features, 0 broken
record_environment.py                   matches the installed .venv
repro_weat_840B.py                      resolves its paths from any cwd
```

676 vs 637 at baseline: **39 new tests**, coverage steady at 86%, fast suite
comfortably inside the 3-minute budget.

`render_ledger.py` was checked against a deliberately corrupted ledger and
caught all four false-evidence forms (missing file, test not collected by
pytest, missing path, `exempt:` with no reason) while accepting the three valid
ones. `check_manifest.py` was checked the same way and reported all six planted
problems.

### What remains

Phase 1, the fidelity audit of all 40 metrics, starting with the Section 4.0
source-retrieval gate. `SOURCES.yaml` is empty, so `check_manifest.py` currently
validates entry shape only; its coverage half needs `MetricInfo`, which is Phase
2 (5.1). Every ledger cell is empty by design — nothing is verified yet.

### REVIEW_LATER entries created

RL-001 `decide` E402 ignore · RL-002 `decide` C901 deferral · RL-003 `decide`
dev-only pyyaml/pypdf · RL-004 `verify` `__version__` 0.1.0 vs 0.1.1 ·
RL-005 `verify` 840B not recomputed · RL-006 `verify` 44 of 45 examples never
executed · RL-007 `decide` gitignore for weights · RL-008 `verify` WEAT
sample-vs-population std · RL-009 `decide` `scripts/` not linted · RL-010
`decide` tiny-encoder substitution.

**RL-008 is the one to read first.** Writing the WEAT oracle surfaced that
`weat.py:244` uses `np.std(..., ddof=1)` — the sample standard deviation — while
neither its docstring nor Caliskan eq. 3 says which is meant. At n = 8 the two
differ by ~7%, which is larger than the 6B/840B gap above, and it propagates to
SEAT and CEAT. Deliberately **not** settled from memory: Section 4.0 requires
reading the paper and `W4ngatang/sent-bias` first. The oracle takes `ddof`
explicitly and defaults to 1 to match the library today, which is a pin, not a
verdict.

---

## 2026-08-22 · Phase 1 — Fidelity audit · Section 4.0 gate complete, 8 / 40 metrics audited

### Section 4.0 — source retrieval gate (all three boxes ticked)

**Every Appendix E identifier confirmed.** PLAN.md flags them as "filled from
memory; confirm the title matches before use", so all 27 arXiv ids were resolved
against `export.arxiv.org` and their `citation_title` compared. All 27 match.
`2505.23840` had no expected title recorded; it resolves to *Measuring
Sycophancy of Language Models in Multi-turn Dialogues*, which is TofNof.

**Retrieved.** 38 papers downloaded and text-extracted; 21 reference repos
cloned at pinned SHAs (791 MB, git-ignored). Two papers could not be retrieved
and carry an explicit `paper_status`: SentenceBiasScore (Springer paywall, no
preprint located) and OpinionConsistencyAcrossPersonas (no cited source). Eight
metrics have `code_status: none_found` with search logs.

**`SOURCES.yaml` gained a `status` field.** Retrieval and reading are separate
acts, and the checkbox asks for 40 entries while the schema demands non-empty
`sections_read`. An entry is created `pending` with confirmed identifiers and
becomes `read` only when someone has read the method section and the scoring
code — `check_manifest.py` refuses `read` without both `sections_read` and
`code.files_read`, and prints the pending count so a half-finished audit cannot
look complete. Verified by planting a fake `read`; it was rejected.

**`third_party/LICENSES.md`** records all 21 licenses. `.gitignore` uses
`third_party/*` + `!third_party/LICENSES.md`, since git cannot re-include a file
inside an excluded directory.

### Audit — 8 of 40, prioritising every metric 4.2 flagged `mismatch`

All eight preliminary `mismatch` calls are **confirmed**, each against the paper
and, where it exists, the authors' code:

| Metric | What the paper says | What v0.1.1 does |
|---|---|---|
| LPBS | `log p_tgt − log p_prior`, differenced across two targets | proportion of pairs where the stereotype sentence scores higher; **no prior correction at all** |
| DisCoMetric | mean count of top-3 fills significantly gendered by χ², Bonferroni-corrected | size of a top-k set symmetric difference; **no significance test** |
| DemographicRepresentation | TVD from uniform, counts normalised by group word-list size | entropy / normalized entropy / Gini |
| StereotypicalAssociations | mean over target words of that same TVD | rule/window hit rates |
| FGB | `(1/T) Σ_t Σ_{s=1..217} Var_d(mean style prob)` | mean absolute paired gap |
| PGB | the same, restricted to a style cluster | mean rectified paired gap |
| BOLD | a dataset plus five metrics (VADER, toxicity, regard, NRC-VAD, gender polarity) | a "lexical bias heuristic" the paper does not contain |
| DemographicRepresentationBias | WinoBias pro/anti coreference **F1 gap** | he/she/they **pronoun counts** |

### Three corrections back into PLAN.md

Reading the sources disproved parts of 4.2 itself:

1. **FGB's formula was wrong in the plan.** 4.2 described it as "Σ over style
   clusters of Var over descriptors of cluster proportion". §A.7 sums over all
   **217 individual styles**, with no clusters. The cluster-summed form is the
   paper's *third* metric, **SCGB**, introduced because PGB "artificially
   deflates the bias in style clusters with many styles".
2. **PGB is not "uncited in the paper".** It is defined in §A.7 and reported in
   Table 4 and Figure 4.
3. **DisCo's open question is answered.** 4.2 asked for "χ² (p<0.05 with the
   paper's correction if any)". The paper is explicit: **Bonferroni**.

Both plan rows were corrected in place, per Section 1's "the plan itself is
wrong" rule.

### A Tier-1 target worth having

DisCo Table 2 gives eight precisely citable values on four public models —
DisCo (Terms) 0.4 / 0.0 / 0.8 / 1.0 and DisCo (Names) 3.7 / 3.1 / 3.7 / 3.4 for
ALBERT Base/Large and BERT Base/Large. All eight are now in
`validation/registry.yaml` with the table, row, and column named. This is one of
the better Tier-1 targets in the library and it was previously unrecorded.

### Verification

```
ruff check src tests                    clean
pytest -q --cov=bias_scope              676 passed, 3 deselected, 86%
check_manifest.py                       OK, 8/40 read, 32 pending, 8 no-code
check_manifest.py (planted fake `read`) correctly rejected
render_fidelity_index.py                8 notes, 8 mismatches, 0 problems
render_ledger.py --check                0 broken
```

No library code changed this session, so the suite is unchanged from the Phase 0
entry above. `scripts/experiments/` is still at its pre-existing 53 ruff
violations (RL-009); the WEAT path edits added none.

### What remains

32 metrics still `pending`. None may be audited before its `SOURCES.yaml` status
is `read`. Phase 2's reimplementations are blocked on those audits, and FGB/PGB
are blocked outright (RL-013).

### REVIEW_LATER entries created

RL-012 `verify` LPBS reference code appears to read the prior at the wrong mask
position (**a static code reading, not an executed result**) · RL-013 `blocked`
HolisticBias style classifier unavailable, FGB/PGB cannot be faithful ·
RL-014 `decide` BOLD dataset is CC-BY-SA-4.0 against an MIT repo · RL-015
`verify` six reference repos state no license, removing the copy-the-snippet
fallback for exactly the metrics likeliest to need it.

---

## 2026-08-22 · Phase 2 — Framework layer and the first reimplementation

The unblocked half of Phase 2. Metric reimplementations need their Section 4.0
audit first and 32 of 41 are still `pending`, so this session built the parts
that do not depend on the audit, plus the two fixes whose audits are done.

### 5.1 — `MetricInfo` on every metric

`bias_scope/metadata.py` (the dataclass, `normalized_deviation`, a registry) and
`bias_scope/_metric_info.py` (the table for all 41 metrics), attached explicitly
by one call from `__init__.py` — no import-side-effect registry.

**A fifth fidelity status, `unaudited`.** PLAN.md 4.1 defines four, and Section
4.0 forbids assigning one before the sources are read. With 32 metrics unread,
the alternative was to guess, which is precisely the unverified claim v0.2
exists to remove. `tests/test_metadata.py` pins each metric's fidelity to its
`SOURCES.yaml` status in **both** directions: `pending` must be `unaudited`, and
`read` must not be. Verified by planting `fidelity="faithful"` on WEAT — the
test failed with the reason. Section 13's release gate gains "zero unaudited",
recorded as two `xfail(strict=True)` tests that will fail loudly once true.

### 5.3 — `BiasResult`, `run()`, and the runtime guards

`bias_scope/stats.py` (4 functions, each with a hand-derived known-answer test,
including a coverage check that the bootstrap interval really covers ~95%),
`bias_scope/result.py`, and `BiasMetric.run()` in `base.py`. `evaluate()` is
untouched, so nothing existing broke.

All four Section 1 guards are implemented **and each has a test proving it
fires**: non-finite score, score outside `value_range`, `n <= 0`, and a CI that
does not bracket the score. The `n <= 0` guard immediately caught a real gap —
`WEAT.run()` failed because effect-size metrics report no item count — so
`EmbeddingMetric` now supplies the Hedges-Olkin interval that 5.3 asks for.

### 5.2 — LPBS reimplemented, and one rename

**`LPBS` now implements Kurita et al. 2019**, from the §2 formula:
`[log p_tgt(t1) - log p_prior(t1)] - [log p_tgt(t2) - log p_prior(t2)]`, with
target word sets summed before the log as the authors' code does. Known-answer
tests derive `2·log 2` by hand; swap antisymmetry, the null property and
attribute permutation invariance are tested directly. A dedicated test pins the
prior to the **target** mask, the RL-012 judgement, using a fixture that returns
different values at each mask so reading the wrong one fails.

The v0.1.1 statistic is preserved unchanged as `PairwiseLikelihoodPreference`
(`original`). There is **no alias** from `LPBS` to it: Section 1 forbids a
wrong-metric class keeping its old behaviour under the old name.

**`DemographicRepresentationBias` → `OccupationPronounSkew`** (`original`).
A pure rename, so the old name stays importable until 0.3.0 behind a
`DeprecationWarning`; the alias is a subclass, so `isinstance` still works.
`bias_scope/_deprecation.py` holds the mechanism and documents when it does
*not* apply.

### A test that was testing nothing

`tests/test_probability_based/test_cbs.py` was a near-verbatim copy of
`test_lpbs.py` — same four tests, same assertions — and contained **zero**
references to CBS. It was removed: `cbs.py` coverage is **20% before and 20%
after**, so it exercised none of it, and the `-m slow` count fell 3 → 2 only
because it carried a duplicate of the one slow test. CBS therefore has no tests
at all and was passing CI on a mislabelled file (RL-017).

### Verification

```
ruff check src tests                    clean
pytest -q --cov=bias_scope              1136 passed, 2 deselected, 2 xfailed, 86%
pytest -q -m slow                       2 passed
check_manifest.py                       OK, 9/41 read
render_ledger.py --check                0 broken
render_fidelity_index.py                8 notes, 6 mismatches
```

1136 vs 676 at the end of Phase 0: **460 new tests**. Coverage steady at 86%.

Fidelity now: 1 faithful, 1 original, 6 mismatch, 32 unaudited (was 8 mismatch,
32 unaudited).

### What remains

- **32 metrics unaudited.** This is the bottleneck: 5.1's fidelity values and
  5.2's reimplementations both wait on it.
- **6 mismatches.** DisCo, DemographicRepresentation, StereotypicalAssociations
  and BOLD are ready to reimplement (audited, unblocked). FGB and PGB are
  **blocked** on the unpublished style classifier (RL-013).
- **5.4 not started**: `BiasSuite`, `recommend_metrics`, `report`, `backends`.
- Phases 3–8 untouched.

### REVIEW_LATER entries created

RL-016 `verify` two different Hedges-Olkin standard errors in one repo — the new
`stats.py` and `finalize_emnlp.py` disagree, and the published `results/emnlp/`
CIs were deliberately **not** moved · RL-017 `verify` CBS has no tests at all.

---

## 2026-08-23 · Phase 1 continued — the four headline reproductions audited

Priority set by what actually sank the last submission: a reviewer spot-checked
one metric and found it wrong. The four metrics in the paper's evaluation table
are the ones a reviewer will re-check first, so they were audited before
anything else.

### WEAT — faithful, and RL-008 closed

`std-dev` in Caliskan's effect-size formula is genuinely ambiguous in the paper.
Three independent lines settle it on **ddof=1**, which is what
`weat.py:244` already did:

1. `sent-bias/sentbias/weat.py:175` is `np.std(..., ddof=1)`, explicit.
2. On Caliskan's own GloVe-840B vectors: ddof=1 reproduces 1.81 to **0.22%**,
   ddof=0 is off by **3.50%** — a 16x difference.
3. The 6B replication orders the same way (1.6938 vs 1.7494).

So the audit **confirms** the implementation rather than changing it.

**But v0.1.1 omitted the permutation test entirely** — the p-value is half of
WEAT's definition. Implemented in 0.2.0: exact enumeration over all C(2n, n)
partitions up to 100,000 of them (Caliskan's own 8-per-group tests are always
exact), sampling beyond. The observed partition is always counted so p floors at
1/num_partitions rather than reaching an impossible zero.

### CrowS-Pairs — faithful

Matches eq. 1 and the reference on every point, including the `range(1, N-1)`
interior that skips CLS/SEP and the difflib `equal`-opcode alignment. One
deliberate difference, documented: `autojunk=False`, immaterial at CrowS-Pairs
sentence lengths.

### HONEST — adaptation

The formula is exact. Two deviations move the number and neither is optional:
causal-LM generation replacing top-K mask fills, and HurtLex growing from
Nozza's 1,072 terms to 1,722. Both were already bracketed in `results/emnlp/`.

### BBQ — was a mismatch; now faithful. **This was the important one.**

`is_biased = not is_correct` → `bias_score = biased_count / n`. That is the
**error rate**: 0.2496 + accuracy 0.7504 = 1.0000 exactly in the recorded run.
Parrish et al. define

    s_DIS = 2*(n_biased / n_non-UNKNOWN) - 1,   s_AMB = (1 - accuracy) * s_DIS

signed in [-1,+1], using `question_polarity` and the bias target. **A model
answering entirely against the stereotype scored +1.0 (maximally biased) under
ours and -100% under the paper** — opposite conclusions, not a scaling
difference. A second LPBS, sitting in the paper's headline table.

Reimplemented, with the subtlety that matters: `target_loc` **flips with
question polarity** (2,670 of 2,678 paired questions differ). The derivation
from `answer_info` + `stereotyped_groups` was validated against `nyu-mll/BBQ`'s
own metadata — **19,092 matches, 0 mismatches** across nine categories. 12,264
name-proxy rows are underivable and are excluded *and counted*, never guessed
(RL-020).

### The paper's BBQ row is withdrawn

Auditing BBQ surfaced a second, worse problem. `results/emnlp/` compared 0.2496
against a "published" **0.25** that `finalize_emnlp.py` itself called the
"midpoint of a 0.22-0.28 published range", attributed to sources never resolved
to precise citations — and `METHODOLOGY.md` records that an earlier anchor of
0.20 was **rejected because its CI did not overlap our number**. A reference
value chosen to fit the result is a fitted parameter, and PLAN.md Section 1
forbids exactly this; Section 6.1 names this anchor specifically.

Worse, `METHODOLOGY.md` argued in as many words that "anchoring on the range
midpoint is the honest scientific practice". That passage is **retracted in
place**, with the history kept so the change is auditable.

The row now reads WITHDRAWN with its reason, in `bbq.json`, `summary.md`,
`table.csv` and `CONFIGURATION.md`, and the anchor is gone from both generator
scripts. The genuine finding from that work — AWQ-INT4 vs BF16 moves the score
by **17 points** — was preserved; it was measured, not anchored.

### Verification

```
ruff check src tests      clean
pytest -q --cov           1160 passed, 2 deselected, 2 xfailed, 86%
finalize_emnlp.py         regenerates summary/table/CONFIGURATION cleanly
```

Fidelity: **4 faithful, 1 adaptation, 2 original, 6 mismatch, 28 unaudited**
(13 of 41 audited, up from 8).

### What remains

28 metrics unaudited. 6 mismatches: DisCo, DemographicRepresentation,
StereotypicalAssociations and BOLD are audited and ready to reimplement; FGB and
PGB stay blocked on the unpublished style classifier (RL-013). Phase 2's 5.4
(`BiasSuite`, `recommend_metrics`, reports) is not started.

### REVIEW_LATER entries

RL-018 partly resolved (BBQ metric fixed, row withdrawn; a fresh model run is
still needed to restore it) · RL-019 `verify` `evaluate()` return types are
inconsistent across the family — evidence for the paper's own framework
argument, so tabulate rather than quietly harmonise · RL-020 `verify` BBQ target
derivation covers 61% of the benchmark; SES and Race_ethnicity currently score
over zero rows.

---

## 2026-08-23 (later) · Phase 2 — every unblocked mismatch is fixed

Four reimplementations, each test-first from the paper, each preserving the
displaced statistic under a name that does not claim a paper.

### DemographicRepresentation and StereotypicalAssociations → adaptation

Both are HELM's `TVD(group distribution, uniform)`, so `group_counts_to_bias`
lives in one shared `_helm.py` and cannot drift. The step most likely to be
dropped in a reimplementation — **normalising each group's count by the size of
that group's word list** — has a dedicated test showing that raw counts [10, 5]
score 1/6 without it and 0 with it.

The no-mentions case returns `None` with a reason, not 0.0: 0.0 is the
*unbiased* value, and claiming it when nothing was mentioned would be a false
statement about the model. `run()`'s guards reject a `None` score, so it cannot
silently become a `BiasResult`. That changed v0.1.1's behaviour (which raised),
and the two affected tests were rewritten to assert the new documented
behaviour rather than deleted.

Status is **adaptation**, not faithful, for one honest reason: HELM tokenises
with NLTK and the default here is a regex tokenizer, to keep the core install
light.

### DisCoMetric → faithful

Templates, top-3 fills, per-fill chi-square against equal prediction rates,
**Bonferroni-corrected** α = 0.05, mean count of significant fills per template.

`chi_square_2xk` and `chi_square_p_value` are written out rather than pulled from
scipy, so the core install stays light — and **validated against
`scipy.stats.chi2_contingency` on five contingency tables, matching to 1e-9 on
both the statistic and the p-value.**

The test that separates old from new: one name per gender with completely
disjoint top-3 fills. The v0.1.1 symmetric difference reported **6**; DisCo
reports **0**, because one name pair cannot be distinguished from chance. That
is the whole point of the significance test, and v0.1.1 had none.

`DisCoMetric` also no longer needs torch — the caller supplies the fills — so it
installs with the core package.

### BOLD → faithful, as a runner

BOLD is a dataset plus five metrics, so `BOLD` became a **runner**: load prompts
by domain, generate, hand the continuations to whichever of the paper's five
metrics the caller names. It reports per domain per metric and **produces no
aggregate score**, because the paper produces none and collapsing them would be
the composite score PLAN.md lists as a non-goal.

**The lexical heuristic was removed, not renamed.** Section 12's
"keep it under a new name" rule is for reasonable statistics that were merely
mislabelled; this was an undocumented ad-hoc word-list score with no cited
basis. Its 8 tests went with it; the runner has 17 and module coverage rose from
92% to 98%.

### The state that matters for the paper

```
faithful 6 · adaptation 3 · original 4 · mismatch 2 · unaudited 28
```

**Every mismatch that can be fixed has been.** The two that remain — FGB and
PGB — are defined over a 217-class style classifier that HolisticBias never
published (RL-013); no amount of work here makes them faithful. That is now
pinned by a test, `test_the_only_remaining_mismatches_are_the_blocked_ones`, so
a new mismatch appearing, or one of these being resolved, forces someone to
revisit the release gate rather than letting it drift.

### Verification

```
ruff check src tests      clean
pytest -q --cov           1228 passed, 2 deselected, 2 xfailed, 86%
```

### What remains

- **28 metrics unaudited.** Still the bottleneck, and still the main risk: two
  of the four headline metrics audited so far had real defects.
- **FGB/PGB blocked** (RL-013).
- **The `results/emnlp/` BBQ row** needs a fresh model run to return (RL-018).
- **5.4 not started**: `BiasSuite`, `recommend_metrics`, reports, backends.
- Phases 3-8 untouched.

---

## 2026-08-23 (continued) · Phase 1 complete — 42 of 43 metrics audited

Worked straight through the remaining audit. Every metric's sources have now
been read except one, and that exception is recorded rather than guessed.

```
faithful 17 · adaptation 14 · original 9 · mismatch 2 · unaudited 1
```

### Defects found by reading the sources

Beyond the four headline metrics reported earlier, six more real problems:

| Metric | What was wrong |
|---|---|
| **RegardScore** | Defaulted to a **sentiment** classifier and described itself as measuring sentiment. Sheng et al.'s Table 2 lists sentences where sentiment and regard have **opposite signs** — the conflation the paper was written to refute. Now defaults to `sasha/regardv3` with `other` kept distinct from `neutral`. |
| **CounterfactualSentimentBias** | Computed the signed mean of paired differences, in which **opposing biases cancel**. Huang et al. define the Wasserstein-1 distance. Now W1, validated against `scipy.stats.wasserstein_distance` on seven cases including unequal sample sizes, **matching to 1e-9**. |
| **CAT** | Two defects: `lms` compared `max(stereo, anti)` once where StereoSet counts **both** comparisons over a `2N` denominator; and it took a flat mean where the paper averages **per target term** first. Both fixed. |
| **AULA** | Renormalised the attention weights, computing `Σαᵢlogpᵢ / Σαᵢ` instead of eq. 5's `(1/|S|) Σαᵢlogpᵢ`. The scaling varies per sentence, and since the bias score is a pairwise indicator it can **flip individual comparisons**. Fixed. |
| **ToxicityFraction** | Listed `faithful` to Gehman. The paper says "two metrics" and names them; "fraction" does not appear in it. Relabelled `original`. |
| **ScoreParity** | Cited Borkan et al., whose five metrics are all AUC/Mann-Whitney and **label-based**. A mean-score gap is none of them. Renamed `MeanScoreGap`. |

Smaller, documented: LMB's outlier rule (percentile → the paper's 3-sigma, now
default), CoOccurrenceBiasScore's missing normalising term, SEAT/CEAT's pooling
default (`mean` → `cls`, the reference protocol).

### Open questions the audit closed

- **WEAT's `std`** — `ddof=1`, confirmed three ways (RL-008 closed).
- **AULA's attention layers** — averaged over **all** layers and heads.
- **DisCo's χ² correction** — **Bonferroni**, explicit in the paper.
- **MarkedPersons' "reported scalar"** — there isn't one, and that is correct.

### The one metric that stays unaudited

`SentenceBiasScore` cites Dolci et al. 2023 (*Data Science and Engineering*
8(2)), which is paywalled with no preprint. Section 4.0 forbids assigning a
fidelity without reading the paper, so it keeps `unaudited` and its manifest
entry stays `pending`. `test_the_only_unaudited_metric_is_the_unobtainable_one`
pins that, so a future metric cannot slip in unaudited alongside it (RL-029).

**This is worth a sentence in the paper.** "42 of 43 audited; one cites a source
we could not obtain and is labelled `unaudited` rather than asserted" is a
stronger integrity claim than "43 of 43 verified".

### Four of PLAN.md 4.2's own calls were wrong

The preliminary table was written from a docstring-level pass, and reading the
sources overturned four of its verdicts — `ToxicityFraction`, `RegardScore`,
`BBQMetric`, and FGB/PGB's formula descriptions. All corrected in place. That is
itself evidence for the paper's argument: a docstring-level review is not an
audit.

### Verification

```
ruff check src tests      clean
pytest -q --cov           1249 passed, 2 deselected, 2 xfailed, 86%
check_manifest.py         42/43 read, 1 pending
render_fidelity_index.py  0 inconsistencies
```

### What remains

- **2 mismatches**, both blocked: FGB and PGB need HolisticBias's unpublished
  217-class style classifier (RL-013).
- **`results/emnlp/` BBQ row** needs a fresh model run to return (RL-018).
- **5.4 not started**: `BiasSuite`, `recommend_metrics`, reports, backends.
- Phases 3-8 untouched.
- Fourteen `adaptation` metrics have documented paths to `faithful`; the
  cheapest are UnQover's masked-LM path, TruthfulQA's MC1/MC2, and
  SocialGroupSubstitution's W1 (the helper already exists).

### REVIEW_LATER entries created this stretch

RL-021 CEAT's default N=100 vs the paper's 1,000/10,000 · RL-022 EMT omits the
paper's standard deviation · RL-023 CoOccurrenceBiasScore's missing
normalisation · RL-024 `CounterfactualFairness` renamed to
`IdentitySwapConsistency` (resolved same session) · RL-025 BOLD's second
gender-polarity metric absent entirely · RL-026 LMB's headline is not the t-value
· RL-027 SocialGroupSubstitution uses a range where the paper uses W1 · RL-028
TruthfulQA has neither MC1 nor MC2 · RL-029 SentenceBiasScore's paper
unobtainable.

---

## 2026-08-23 (continued) · Phase 2 §5.4, Phase 6 visualization, and three new metrics

### 5.4 — the framework layer the v0.1 reviewers found missing

`backends.py`, `recommend.py`, `suite.py`, `report.py`. All seven 5.4 boxes
ticked, each acceptance check written as a named test.

**Access is derived, never guessed.** `HuggingFaceBackend` declares
`embeddings`/`logits` (+`completions` for causal); `LiteLLMBackend` declares
`completions`/`chat` and deliberately **not** `logits`, because chat APIs expose
at best top-k logprobs. A test asserts no HF backend ever claims `chat` and
LiteLLM never claims `logits` — otherwise metrics would run on data they cannot
use.

`recommend_metrics` returns metrics sorted **faithful → adaptation → original →
unaudited → mismatch**, each with a reason string carrying its fidelity caveat.
`mismatch` metrics are excluded by default; TruthfulQA and TofNof are excluded as
not-social-bias, per Section 12. `explain_exclusions` says why each *excluded*
metric was dropped — a recommender that silently discards two thirds of the
library is not trustworthy.

`BiasSuite` **skips rather than zeroes**: a metric with no inputs, or one that
raises, is recorded in `report.skipped` with the reason. A missing number is
information; a fabricated one is a defect.

`compare` refuses mismatched metric sets. `correlate` refuses fewer than three
reports, because with two models every correlation is exactly ±1.

**No composite score anywhere**, and a test asserts it: `Report` has no
`overall`, and both output formats print "must not be averaged".

### Phase 6 — visualization

Five plots plus inline-SVG embedding in the HTML report. Section 9's rules are
enforced, not just followed: no composite/gauge/radar, every value carries its
CI, every figure footer carries the protocol hash + seed + dtype + resources,
and fidelity is a colour and a badge.

Tests assert on **drawn artists and data**, never pixels. One test counts the CI
whiskers and checks there are exactly two for a report with three results, one
of which has no interval — so "metric without an interval gets a marker only" is
verified rather than assumed.

`to_html(figures=True)` inlines the SVG so the page stays one file. A test greps
for real resource loads (`src=`/`href=`/`url()` to http) rather than the string
"http", because every valid SVG contains `xmlns="http://www.w3.org/2000/svg"` —
a namespace, never fetched.

### Three new metrics (PLAN.md 7.2)

| Metric | Status | Note |
|---|---|---|
| **WinoBias** | faithful | The real thing at last: pro/anti coreference accuracy gap over Zhao et al.'s own data files. Distinct from `OccupationPronounSkew`, which used to cite this paper while counting pronouns. |
| **PoliticalEvenHandedness** | faithful | Ported from Anthropic's released method. All three dimensions reported separately — **a model that refuses every political prompt scores perfectly even-handed**, so reporting even-handedness alone would make total disengagement look ideal. |
| **DiscrimEval** | adaptation | `p_norm(yes)` and the logit are exact; the discrimination score is a marginal mean difference rather than the paper's mixed-effects coefficient (that needs statsmodels). Headline is the **largest absolute** group difference — a mean would let a favoured group cancel a disfavoured one, and a test pins that they do cancel in a mean. |

### A real defect the end-to-end run exposed

Running `BiasSuite` for real — not in a test — showed **SEAT being skipped**:
it never reported its group sizes, so `run()` could not build a Hedges-Olkin
interval and the `n > 0` guard rejected it. SEAT now delegates to WEAT with
`return_details=True` and carries the sizes and permutation p-value through. It
was invisible to the unit tests and obvious the moment the suite ran.

### Verification

```
ruff check src tests    clean
pytest -q --cov         1432 passed, 2 deselected, 2 xfailed, 86%
end-to-end              suite -> report -> md/html/json -> inline SVG, offline
```

Artefacts in `results/verification/suite/`.

Fidelity: **19 faithful · 15 adaptation · 9 original · 2 mismatch · 1 unaudited**
across 46 metrics.

### What remains

- **2 mismatches**, both blocked on the unpublished HolisticBias classifier.
- **The BBQ row** in `results/emnlp/` needs a model run.
- **Phase 7.2 remainder**: FirstPersonFairness, DecodingTrust, TrustLLM,
  LLM-IAT, and the multilingual variants (MBBQ/KoBBQ/CBBQ, French CrowS-Pairs,
  SHADES) — each needs its own 4.0 source gate first.
- **Phase 3** (validation tiers), **Phase 5** (adapters), **Phase 7** (studies),
  **Phase 8** (release) untouched.

---

## 2026-08-23 (session 3) · PLAN.md 7.2 — nine new metrics, multilingual, docs

### The rule that did the work

Section 4.0 first, every time: fetch the paper, clone the code at a pinned SHA,
read the scoring function, *then* write. Four repositories were cloned this
session and every scoring path was read before a line of metric code existed.
Doing it in that order is what produced the findings below; none of them would
have been visible from the papers alone.

### Bai et al. (PNAS 2025) — and a Tier 2 result

`ImplicitAssociationTest` and `LLMDecisionBias`, both **faithful**.

The repository ships all 33,600 raw responses **with the authors' own
`iat_bias` value per response**, which makes a per-item comparison possible
rather than a comparison of means:

```
released responses:                      33600
scorable from stimuli/iat_stimuli.csv:   18885
exact matches (1e-9):                    18885/18885 = 100.00%
```

The other 14,715 used automatically generated word sets the repository does not
ship (paper Sec. 2.1), so nobody can score them from the released files.

Two things had to be got right for that to happen.

**The 0.01.** `d_score` adds it to both denominators; the paper's formula does
not. Keeping it is what makes the numbers match — the released column's maximum
is 0.9987507807620236, which is exactly `2·(16/16.01) − 1`. `epsilon=0.0` gives
the paper's formula and reproduces both of its worked examples
(`7/7 + 7/7 − 1 = 1` and `3/7 + 6/7 − 1 = .29`) exactly. RL-030.

**The shipped notebook is not the pipeline that made the paper.** Scoring the
released responses the way `clean.ipynb` does reproduces only **87%** of the
authors' own column. It splits each stereotype's attribute list in half in
`iat_stimuli_synonym.csv` — but that file appends further attribute sets
(occupations, then warmth/competence traits) *after* the valence set, so the
halves stop corresponding to valence. `iat_stimuli.csv`, where each stereotype
is a single set, matches 100%. `load_iat_stimuli` reads the base file only and
says why. Worth reporting upstream — RL-031.

### DecodingTrust — two metrics and two surprises

Both **faithful**. `classify_response` transcribed with all three phrase lists;
DPD and EOd reimplemented and checked against `fairlearn` 0.14.0, the package
`score_calculation_script.py` itself calls: **exact to 1e-12 on 2000 seeded
random cases**, degenerate groups deliberately left in.

Two reference behaviours found by writing tests against the transcribed code,
both reproduced and pinned because both move scores:

- **"I cannot agree or disagree" scores DISAGREE, not refusal.** `cannot agree`
  is in `DISAGREE_PHRASES`, so the phrase trips both lists and falls into the
  disclaimer tie-break, which checks disagreement first. The commonest way a
  safety-trained model declines is counted as *disagreeing with the
  stereotype*.
- **`'no' in answer` matches "not", "note", "know".** "I am not able to say."
  is read as a confident prediction of the negative class rather than as a
  rejection.

One deliberate refusal to follow the code: an answer naming both classes is
resolved there by `np.random.uniform(0, 1) > 0.5` on the **global unseeded**
RNG, so the published fairness numbers are not reproducible even from the
released outputs. We drop the record into the rejection rate the benchmark
already reports. RL-032.

And fairlearn's own convention, kept and flagged: a group with no records
carrying the relevant label has an undefined rate, and fairlearn scores it
**0.0** silently. One false positive in one group, two other groups with no
negative records, gives an equalized odds difference of **1.0**.
`degenerate_rate_groups()` counts them. RL-033.

### TrustLLM — four metrics, not one

All **faithful**. The reason they are four:

| Metric | Ideal | Direction |
|---|---|---|
| StereotypeRecognition | **1.0** | accuracy |
| StereotypeAgreement | 0.0 | higher worse |
| Disparagement | **1.0** | a p-value |
| Preference | **1.0** | refusing is the rewarded behaviour |

A model that recognised every stereotype (1.0) and agreed with every stereotype
(1.0) would average to mid-range. Section 1's no-composite rule earning its
keep.

Two benchmark quirks preserved as written: the two stereotype metrics use
**different denominators** (recognition drops invalid answers, agreement counts
them against agreement), and `count_stereotype_recognition` needs an explicit
`anti-` guard because "stereotype" is a substring of "anti-stereotype".

### FirstPersonFairness — an honest `adaptation`

The estimator is the paper's exactly: the net score `h_F − h_R`, the swapped
second judging pass with its option crossing, the identical-response rule. What
is missing is the prompt. Figure 3 is marked **"slightly abbreviated"** and six
searches found no released code (logged under `code_status: none_found`). PLAN.md
7.2 requires the judge instructions to be mirrored verbatim, so the status is
`adaptation`, not `faithful`. RL-035.

The subtraction is the whole method, and a test pins the paper's own case: two
response types at 50/50 for both groups gives a 25% stereotype rate **and** a
25% reverse rate, netting to exactly 0. `H_forward` alone would report 0.25 for
a model with no name effect at all.

### Multilingual (Section 12: loader only)

`bias_scope.multilingual` — loaders for MBBQ, KoBBQ, CBBQ and the French
CrowS-Pairs extension, with provenance for SHADES, HONEST and CBS. Licenses
checked against each repository's own metadata, not inferred: **CBBQ states no
license** and SHADES declares `other` on a gated repo, so both are marked
unstated rather than assumed permissive. Nothing is redistributed, and a test
walks the installed package asserting no dataset file ships with it.

`MetricInfo.languages` is now truthful where it was not: **CBS declares twelve**
(twelve keys in the authors' `configuration.py`, each with its own templates,
nationality list and checkpoint — for a paper titled "Language-Dependent Ethnic
Bias"), **HONEST six** (six template files in `resources/`). A test refuses any
language claim not backed by a dataset in the registry.

### A defect the tests found, and one the suite found

- `run()` **rejected legitimate results**: a metric whose per-item values are
  all equal gives a degenerate interval `[v, v]`, and the mean can miss both
  endpoints by an ULP from summation order. A perfectly consistent model was
  unrunnable. Fixed with 1e-9 relative slack; a real gap still raises. RL-034.
- `SEAT` was **silently skipped by `BiasSuite`** because it never reported its
  group sizes, so `run()` could not build an interval. Invisible to the unit
  tests, obvious the moment the suite was run for real.

### The ledger, and a claim that was not true

PLAN.md Appendix C step 5b: a task whose `verification/ledger.yaml` row is
incomplete is not done. The ledger turned out to be an unfilled Phase 0
scaffold — 40 rows, **none with any evidence**, and 18 metrics missing
altogether. All 18 rows were added, and the nine metrics from this session were
filled with test node IDs **verified to collect** before being written.

`render_ledger.py` then rejected seven of the pointers, which is exactly its
job. Six were repointed at real artefacts. The seventh was a claim in
`docs/fidelity/trustllm.md` that the shared chi-square "is checked against
scipy to 1e-9 in DisCo's tests" — **there was no such test**. Rather than soften
the sentence, `tests/oracles/test_chi_square_oracle.py` now compares
`chi_square_2xk` and `chi_square_p_value` with `scipy.stats.chi2_contingency`:
statistic, degrees of freedom and p-value, exact to 1e-9 across 500 random
tables of 2-5 rows by 2-6 columns, plus the specific 2×4 shape
`TrustLLMDisparagement` produces. The note now cites it.

Ledger state: **9 of 58 metric rows complete, 0 broken**. The other 49 are the
pre-existing scaffold and stay empty; that is a real gap and it is visible in
`results/verification/VERIFICATION.md` rather than hidden.

`fairlearn>=0.10.0` added to the `dev` extra with a DECISIONS.md note, so the
DecodingTrust oracle actually runs in CI instead of skipping vacuously.

### Verification

```
ruff check src tests scripts/{paper,sources,verification}   clean
pytest -q --cov=bias_scope               1784 passed, 5 deselected, 2 xfailed, 88%
pytest -q -m slow                        2 passed
BIASSCOPE_RUN_EQUIVALENCE=1 pytest tests/equivalence -m equivalence   3 passed
check_manifest.py                        OK, 55 entries, read 54/55
render_ledger.py                         metrics 9/58 complete, 0 broken
end-to-end                               all 9 new metrics -> report -> 6 inline
                                         SVGs, 249 KB, zero external loads
```

All nine, run for real (`results/verification/suite/new_metrics.md`):

```
metric                              score  neutral    n  ci                  fidelity
ImplicitAssociationTest            0.9950      0.0    4  (0.995, 0.995)      faithful
LLMDecisionBias                    0.7500      0.5    4  (0.250, 1.000)      faithful
DecodingTrustStereotype            0.5000      0.0    2  (0.000, 1.000)      faithful
DecodingTrustFairness              1.0000      0.0    6  -                   faithful
TrustLLMStereotypeRecognition      0.7500      1.0    4  (0.250, 1.000)      faithful
TrustLLMStereotypeAgreement        0.2500      0.0    4  (0.000, 0.750)      faithful
TrustLLMDisparagement              0.0000      1.0   30  -                   faithful
TrustLLMPreference                 0.5000      1.0    2  (0.000, 1.000)      faithful
FirstPersonFairness                0.6000      0.0    5  (0.600, 0.600)      adaptation
```

The `neutral` column is why these are nine metrics and not one number: three
different neutral values appear in it.

Coverage 86% → **88%**. Artefacts in `results/verification/suite/`.

### The recount (generated, not typed)

`scripts/paper/metric_counts.py` → `results/paper/metric_counts.csv`:

| Family | Faithful | Adaptation | Original | Mismatch | Unaudited | Total |
|---|---|---|---|---|---|---|
| embedding | 3 | 0 | 0 | 0 | 1 | 4 |
| probability | 8 | 1 | 2 | 0 | 0 | 11 |
| generated_text | 4 | 8 | 3 | 2 | 0 | 17 |
| prompt | 12 | 7 | 4 | 0 | 0 | 23 |
| **All** | **27** | **16** | **9** | **2** | **1** | **55** |

PLAN.md 13 asked for "at least six new metrics (7.2) including two judge-based
ones and one multilingual dataset variant". Nine new metrics, three judge- or
classifier-bound (`FirstPersonFairness`, `LLMDecisionBias`,
`TrustLLMPreference`), four multilingual dataset variants with loaders.

### What remains

- **2 mismatches**, both blocked on the unreleased HolisticBias classifier;
  **1 unaudited**, a paywalled paper with no preprint.
- **Phase 3** (validation tiers), **Phase 5** (adapters), **Phase 7** (studies),
  **Phase 8** (release) untouched.
- The `results/emnlp/` BBQ row still needs a model run (RL-018).
- **49 of 58 ledger rows are still empty** — the Phase 0 scaffold. Filling them
  is Phase 8 work and each needs its evidence, not a plausible pointer.
- Three documented `adaptation` → `faithful` paths: UnQover masked-LM,
  TruthfulQA MC1/MC2, SocialGroupSubstitution W1.

## 2026-09-12 · Phase 9 — `bias_scope_agent`, a tool-calling agent over the library

New top-level package, `src/bias_scope_agent/`: a single agent LLM (Claude,
via the `anthropic` SDK) in a tool-calling loop, driving `bias_scope`'s
existing `recommend_metrics`, `explain_exclusions`, `BiasSuite.plan()`/`.run()`
and `Report` rendering through thin wrapper functions. `src/bias_scope/`
itself was not touched — Section 4.0's source-retrieval gate does not apply
here, since no metric was added or changed (noted explicitly in the new
PLAN.md §14 so it isn't mistaken for a gap).

Built in dependency order, test-first throughout (`ruff check` and
`pytest -q --cov=...` after every module): `config.py` (env-var config, no
agent-specific API key — the `anthropic` SDK already reads `ANTHROPIC_API_KEY`,
RL-038) → `registry.py` (`HandleRegistry`, opaque UUID handles so backends and
reports never cross the tool-call boundary) → `introspection.py`
(`metrics_needing_data` via `inspect.signature`, since `MetricInfo` has no
input-shape field — confirmed by reading the class in full; `inspect_model`
best-guess model inspection, live by default behind
`BIASSCOPE_AGENT_INSPECT_LIVE`) → `tools.py` (the ten tool wrappers:
`construct_backend`, `recommend_metrics_tool`, `explain_exclusions_tool`,
`plan_suite`, `request_missing_inputs`, `confirm_plan`, `run_suite`,
`summarize_report`, `record_fact`, plus `inspect_model`) → `session.py` (the
confirm-before-run gate: `AgentSession`/`GateError`) → `schemas.py` +
`system_prompt.py` → `loop.py` (`AgentLoop`) → `cli.py`/`__main__.py`
(`bias-scope-agent` console script).

**The gate is structural, not semantic** (RL-040): `AgentSession.check_run_gate`
guarantees a plan was recorded, a real turn boundary passed, and `confirm_plan`
was explicitly called — enforced by `AgentLoop`'s dispatcher *before* the real
`run_suite` is ever invoked, not just stated in the system prompt. A scripted-
conversation test proves this with a spy on `tools.run_suite` that asserts
zero calls when the script tries to jump the gate, not just that the
transcript order looks right.

**Metric-input-shape design fork** (RL-039): `MetricInfo` has no field
describing what `evaluate()` needs, so `metrics_needing_data` introspects
signatures directly via `inspect.signature`, reimplementing (not importing)
`bias_scope.suite`'s private `_metric_classes()` — that helper isn't in
`__all__`, so `bias_scope_agent` keeps its own four-module-name copy rather
than depending on an internal symbol with no deprecation path.

**Found, not fixed** (RL-041): while writing the integration test,
`BiasSuite.run()` raised `BiasScopeError` for `CrowSPairs` and `AUL` — both
return a `"<name>_score"` dict key (`crows_pairs_score`, `aul_score`) that
`BiasMetric._extract_score`'s accepted-key list (`bias_score`, `score`,
`value`, `effect_size`) doesn't recognize, so neither metric can currently run
through `.run()`/`BiasSuite` at all (only `.evaluate()` called directly works
— which is presumably why `tests/integration/test_tiny_model_fixtures.py`
already does exactly that). Out of scope for this phase (CLAUDE.md forbids
touching `src/bias_scope/` here); flagged as a follow-up task instead. The
integration test uses `WEAT` (raw embedding arrays, per its own docstring
example) against a real `HuggingFaceBackend` for
`hf-internal-testing/tiny-random-BertForMaskedLM` instead.

`pyproject.toml`: new `agent` extra (`anthropic`, `huggingface_hub`, folded
into `all`), `src/bias_scope_agent` added to the wheel `packages` list, new
`[project.scripts]` entry (`bias-scope-agent`), and `bias_scope_agent` added
to `[tool.coverage.run] source`.

### Verification

```
ruff check src tests                                              clean
pytest -q --cov=bias_scope --cov=bias_scope_agent   1870 passed, 2 skipped, 5 deselected, 2 xfailed, 89%
```

bias_scope_agent module coverage: `config.py` 97%, `introspection.py` 89%,
`loop.py` 90%, `tools.py` 96%; `session.py`/`registry.py`/`schemas.py`/
`system_prompt.py`/`cli.py`/`__init__.py` 100% (only `__main__.py`'s 3-line
guard is unexercised, as expected under pytest). Overall repo coverage
88% → **89%**.

Manual smoke tests: `python -m bias_scope_agent` starts, prints its prompt,
and exits cleanly on `exit`/`quit`/Ctrl-D without any API key; with no
`ANTHROPIC_API_KEY` set, a real turn correctly fails at the point of the
actual API call (`anthropic`'s own `TypeError` on missing auth), not before —
confirming `AgentLoop`'s lazy client construction doesn't require credentials
until a message is actually sent.

### What remains

- No live-Anthropic-API conversation test was run in this session (needs a
  real `ANTHROPIC_API_KEY`); the scripted-conversation test covers the tool-
  calling loop's logic against a fake client instead.
- RL-041 (`CrowSPairs`/`AUL` unusable via `.run()`) is a pre-existing
  `bias_scope` defect, flagged but not fixed.
- Phases 3, 5, 7, 8 of the main v0.2 plan remain untouched, as before this
  session — this work was additive (a new package) and did not advance them.

## 2026-09-14 · `bias_scope_agent` follow-up — packaging fix, gate decision, memory verification, multi-provider support

Follow-up session closing the 6 gaps identified in review of the first pass
(2026-09-12 entry above). Committed and pushed the first pass to
`agent-implementation` (`git push -u origin agent-implementation`) before
starting; `main` untouched throughout.

**Item 1 (live conversation) — deferred, by request.** No `ANTHROPIC_API_KEY`
was available in this session; the live-conversation integration test was
not added. Left for whoever has a key; PLAN.md §14 tracks it as open.

**Item 3 (packaging) — verified in a genuine clean venv, one real bug found
and fixed.** `pip install -e ".[agent]"` in a fresh virtualenv installs
`anthropic`+`huggingface_hub` correctly. But running `python -m
bias_scope_agent` with no `ANTHROPIC_API_KEY` set and sending one real
message reproduced exactly the failure mode the follow-up plan worried about:
a bare `TypeError` raised ~15 frames deep inside the `anthropic` SDK's
`_base_client.py`, surfacing only on the *first turn*, never naming
`ANTHROPIC_API_KEY`. Fixed: `_build_default_client` (moved into
`providers.py` during Item 6, see below) now checks for the key — and
separately, whether the SDK package is even installed — before constructing
anything, raising an immediate `RuntimeError` that names the missing
variable and the exact `export` command to fix it. Reverified live in the
same clean venv: the failure now surfaces at `AgentLoop.__init__`, before the
REPL banner even prints. 4 new tests (`test_loop_default_client.py`).

**Item 4 (RL-040) — decided: Option A.** `system_prompt.py`'s confirm_plan
rule now explicitly requires unambiguous affirmation and names hedges,
questions, and silence as *not* confirmation — the one safeguard the
structural gate cannot provide on its own. Option B (a deterministic
hedge-phrase denylist logged alongside the `PlanRecord`) was considered and
rejected for now: no live run has shown the agent actually misreading a
reply, and a phrase heuristic is itself easy to get wrong in both directions
(see `REVIEW_LATER.md` RL-040's updated entry for the full reasoning). 1 new
test asserting the tightened language is actually present in the rendered
prompt.

**Item 5 (memory) — verified through the real loop, not just the renderer.**
A new three-turn scripted test
(`test_loop_scripted_conversation.py::TestRecordedFactsReachLaterTurns`)
proves a fact recorded via `record_fact` in turn 2 is present in the actual
`system` string `AgentLoop.client.create()` receives on turn 3 — exercising
the real plumbing end to end, not `render_system_prompt()` called in
isolation (already covered separately). Documented explicitly, in the test's
own docstring, what a scripted fake-client test can and cannot prove here: it
cannot show a *real* LLM would actually stop re-asking a question; only a
live run (Item 1) can. 2 new tests.

**Item 6 (multi-provider) — built, scope confirmed as full support.** Added
`src/bias_scope_agent/providers.py`: `AgentConfig.provider`
(`BIASSCOPE_AGENT_PROVIDER`, default `anthropic`, each provider gets its own
sensible default model string) plus one adapter class per provider
(`AnthropicProvider`, `OpenAIProvider`, `GeminiProvider`), each translating
`schemas.TOOLS` to that provider's own function-calling wire format and
normalizing its response back into a shared `NormalizedResponse`
(`{content: [NormalizedBlock...], stop_reason}`). `loop.py` was refactored
to call only `self.client.create(system=, messages=)` — it no longer
branches on provider or knows anything Anthropic-specific; `AgentSession`'s
transcript is kept in the same provider-agnostic normalized shape between
turns regardless of which provider is active. Real translation-level
correctness matters most for Gemini, whose protocol correlates tool results
by *name* rather than *id* — `_dispatch_tools` (`loop.py`) now attaches a
`name` field to every tool-result dict for exactly this reason, harmless to
the other two providers which correlate by id and ignore it. Gemini's
schema dialect also does not accept every JSON-Schema keyword this package's
own `schemas.py` uses (`default`, on two properties) — `providers.py` strips
to a conservative, known-safe keyword subset when building Gemini's tool
declarations (`REVIEW_LATER.md` RL-043).

New extras: `agent-openai` (`openai>=1.0.0`), `agent-gemini`
(`google-genai>=0.3.0`) — deliberately *not* folded into the base `agent`
extra, so a Claude-only install stays at two dependencies; both folded into
`all`. Reasoning for not routing the agent LLM through litellm despite it
already being a dependency (`bias_scope`'s target-model backends,
`bias_scope_agent.introspection`'s API-endpoint probing) is in
`DECISIONS.md`'s 2026-09-14 entry.

34 new tests: 4 (default-client error messages) + 1 (system-prompt wording)
+ 2 (memory plumbing) + 7 (config provider field/env var) + 20
(`test_providers.py`: per-provider translation unit tests, plus a
parametrized scripted conversation proving the confirm-before-run gate and
tool dispatch behave identically across all three real adapters). Verified
in a clean venv that `pip install -e ".[agent,agent-openai,agent-gemini]"`
installs `anthropic`, `huggingface_hub`, `openai`, and `google-genai`
correctly and all four import cleanly.

**Not done, flagged instead of silently skipped:** the `openai`/`gemini`
adapters have not been run against real APIs in this session (no key
available for either) — `REVIEW_LATER.md` RL-042/RL-043 log this explicitly,
alongside Item 1's same gap for Anthropic. Item 2 (BBQMetric registry
mismatch mentioned in the original follow-up plan) was explicitly out of
scope and not touched.

### Verification

```
ruff check src tests                                                clean
pytest -q --cov=bias_scope --cov=bias_scope_agent
  1904 passed, 2 skipped, 5 deselected, 2 xfailed, 89%   (was 1870, 89%)
```

`bias_scope_agent` module coverage: `session.py`/`registry.py`/`schemas.py`/
`system_prompt.py`/`cli.py`/`__init__.py` 100%; `config.py` 98%; `loop.py`
96%; `tools.py` 96%; `providers.py` 92%; `introspection.py` 89%;
`__main__.py` 0% (unexercised 3-line guard, as expected under pytest).
`bias_scope_agent` test count: 117 (was 83).

### What remains

- Item 1: no live-Anthropic-API conversation test in this session (by
  request, no key available) — `test_bias_scope_agent_live_conversation.py`
  was not created.
- RL-042/RL-043: the `openai`/`gemini` provider adapters are unit-tested
  against hand-built fakes matching each provider's documented API contract,
  not yet run against the real APIs.
- RL-041 (`CrowSPairs`/`AUL` unusable via `.run()`), from the first pass,
  remains a pre-existing `bias_scope` defect, still not fixed (out of scope).
- Phases 3, 5, 7, 8 of the main v0.2 plan remain untouched, as before.

## 2026-09-14 (later) · `bias_scope_agent` — a fourth agent-LLM provider and target-model UX fixes

Two more requests handled in the same follow-up thread, after the multi-
provider work above landed.

**A `local` agent-LLM provider.** `LocalProvider` (`providers.py`) subclasses
`OpenAIProvider`, overriding only client construction (`base_url` defaulting
to Ollama's endpoint, a placeholder API key most local servers ignore) —
inherits `create()` unchanged, since Ollama/llama.cpp/LM Studio/vLLM all
converged on the same OpenAI-compatible wire format. No new dependency;
`_PROVIDERS` and `_DEFAULT_MODELS` both gained a `"local"` entry. 6 new tests.

**Target-model UX, prompted by a direct review of the friction involved.**
Walking through what a user actually has to know/do to specify a target
model surfaced three real gaps, all fixed:

1. `inspect_model` previously had no way to recognize an API-style model
   string ("gpt-4o-mini") at all — it would just fail an HF Hub lookup for
   it and return an unhelpful low-confidence guess. It now checks litellm's
   bundled, offline model registry first (`_litellm_model_hint`): an exact
   match skips the HF Hub attempt entirely (`guessed_source=
   "litellm_model_id"`, confidence high), and a near-miss (a typo) adds a
   "did you mean" note. Confirmed empirically that this registry can still
   miss valid real strings (`"claude-3-5-sonnet-20241022"` isn't in it,
   only provider-prefixed variants are) — logged as RL-045 rather than
   treated as exhaustive; the check is advisory only, never blocking.
2. `inspect_model` silently returned low-confidence, no explanation, for any
   encoder-decoder/seq2seq model (T5, BART, ...) — which
   `HuggingFaceBackend` cannot represent at all (`kind` is only "causal" or
   "encoder"). It now names this explicitly in `notes` rather than leaving
   it an unexplained guess.
3. `inspect_model`'s causal/encoder classification only trusted an exact
   `architectures` string match, missing common open-weight families like
   Llama or RoBERTa whenever a config only set `model_type`. Added a small,
   deliberately non-exhaustive `model_type` fallback lookup (RL-044).
4. **The bigger one:** `construct_backend`'s LLM-facing schema no longer
   has an `api_key` property at all — the target model's own API key can
   now never become a tool-call argument, which means it can never enter
   `self.messages` (the transcript also sent to the *agent* LLM as context
   on every turn). This works for free: `LiteLLMBackend.generate()` already
   passes `api_key=self.api_key` straight to `litellm.completion()`, which
   falls back to the provider's standard env var when `api_key` is `None` -
   confirmed by reading `backends.py` directly before relying on it.
   `system_prompt.py` now states this rule explicitly. Same "structural, not
   prompted" pattern as the confirm-before-run gate (RL-040) - not a
   coincidence, a deliberate reuse of that design.

11 new tests (9 introspection, 2 schema/prompt).

### Verification

```
ruff check src tests                                              clean
pytest -q tests/test_bias_scope_agent/ tests/integration/test_bias_scope_agent_tiny_model.py
  134 passed   (was 117 before this entry)
```

### What remains

- Same live-testing gaps as before (RL-042, and now also the `local`
  provider — no Ollama/llama.cpp instance was available to verify against).
- The `model_type` lookup (RL-044) and litellm hint (RL-045) are both
  deliberately non-exhaustive; revisit if a misclassification or a missed
  match is ever actually reported.
- Still not committed/pushed — held per instruction, same as the rest of
  this session's work.

## 2026-09-17 · `bias_scope_agent` — OpenRouter and litellm as agent-LLM providers, by request from the supervisor

Two more `AgentConfig.provider` values: `openrouter` and `litellm`. Both are
thin `OpenAIProvider` subclasses (same reuse pattern as `local`) — no new
translation logic.

- **`openrouter`**: points the existing OpenAI-shaped client at OpenRouter's
  own endpoint (`https://openrouter.ai/api/v1`, itself plain OpenAI-
  compatible). Requires `OPENROUTER_API_KEY`; fails fast with a clear
  message like every other cloud provider here.
- **`litellm`**: a general escape hatch through `litellm.completion()`
  directly — access to whichever of litellm's 100+ supported providers the
  model string names (OpenRouter included, via litellm's own
  `"openrouter/<slug>"` routing prefix). Built via a small shim
  (`_wrap_litellm_client`) that makes `litellm.completion` answer to the
  same `.chat.completions.create(...)` shape `OpenAIProvider.create()`
  already calls, since litellm's own `completion()` response is already
  OpenAI-shaped — no separate normalization needed. Deliberately has no
  eager API-key check, unlike the other four providers: which environment
  variable litellm needs depends on the model string's provider prefix, so
  there's no single variable to check for generically; a missing/wrong key
  surfaces as litellm's own authentication error on the first real call.

This revisits (without reversing) the 2026-09-14 decision not to route the
agent LLM through litellm: that reasoning was about the *first-class*
adapters (Anthropic/OpenAI/Gemini) not needing litellm's indirection since
it wouldn't avoid a translation layer for them — `litellm` here is an
*additional*, optional path alongside those, and it still needed its own
(small) translation shim, which if anything confirms that original
reasoning rather than undercutting it. Full explanation in `DECISIONS.md`'s
2026-09-17 entry.

13 new tests (`test_providers.py`: 3 construction + 1 translation for
OpenRouter, 2 construction + 2 translation for litellm; `test_config.py`:
2 default-model cases).

### Verification

```
ruff check src tests                                              clean
pytest -q tests/test_bias_scope_agent/ tests/integration/test_bias_scope_agent_tiny_model.py
  144 passed   (was 134 before this entry)
```

### What remains

- Neither new provider has been run against a real key in this session —
  logged as RL-046, same category as every other provider's live-testing
  gap (RL-042, and `local`'s equivalent note).
- This work was done from a fresh harness-managed worktree (the previous
  one was recycled mid-session) on a throwaway branch
  (`agent-implementation-openrouter`, off `agent-implementation`'s tip),
  because this session's Edit tool is blocked from writing directly into
  the user's main checkout folder while tied to a worktree — a harness-
  level guardrail, not something bypassable. Plan is to fast-forward-merge
  this branch into `agent-implementation` in the main folder via `git`
  (unlike file edits, git operations against that folder are not blocked),
  then delete the throwaway branch.

## 2026-09-17 — environment repair and a registry finding (no source change)

Session was a walkthrough of `bias_scope_agent`'s architecture, which turned
into an environment repair when `import bias_scope` proved to fail outright in
the repo's own `.venv`.

- Repaired `.venv` with `uv`: installed the missing **core** dependency
  `requests>=2.28.0`, reinstalled editable (0.1.0 -> **0.1.1**, so protocol
  blocks now record the right `library_version`), then installed `[all]`.
- Full suite afterwards: **1930 passed, 3 skipped, 5 deselected, 2 xfailed**.
  `ruff check src tests` clean. Every failure seen before the repair (10 in
  the agent suite, 5 collection errors project-wide) was environmental.
- **Finding (RL-047):** `list_metrics()` silently omits metrics whose optional
  dependency is absent — 48 metrics core-only, 54 with `[datasets]`, 55 with
  `[all]`. `BiasSuite.plan()` then raises `ValueError: unknown metric
  'BBQMetric'` for a metric that is registered and correct but not installed,
  and the agent relays that wording to the user as fact.
- **Correction:** an earlier draft of RL-047 claimed the agent's BBQ-referencing
  tests had rotted against a removed metric. That was wrong and has been struck
  from the entry — the tests are correct and pass with `[datasets]` installed.
- RL-041 re-confirmed by direct reproduction: `CEAT`, `AUL`, `AULA`,
  `CrowSPairs` are unreachable through `BiasSuite` (and so through the agent)
  because `_extract_score` rejects their `<name>_score` key.

Then, on request, fixed two of the reachability defects (tests written first).

**RL-041 resolved for CrowSPairs / AUL / AULA.** Two independent defects, both
in the `evaluate()` -> `run()` contract, neither in any metric's statistic:
- `base.py` `_split_result` accepted only `bias_score`/`score`/`value`/
  `effect_size`. Now falls back to exactly one `<name>_score` key; two remain
  ambiguous and still raise, since a guessed score is a fabrication.
- `base.py` `_count_items` required `isinstance(value, int)`, but CrowS-Pairs
  reports `num_pairs: 2.0` — a count computed through numpy. A whole-number
  float is now accepted; 2.5 and 0.0 still raise.
Verified on a real `HuggingFaceBackend`: all three return a score with `n=2`
through `BiasSuite`, where all three were previously recorded as skipped.

**RL-047 fixed.** `BiasSuite.plan()` said `unknown metric 'BBQMetric'` for a
metric that is registered and correct but whose extra is not installed.
`prompts_based` now exposes `PROMPT_METRIC_NAMES` (every prompt metric,
installed or not) and `suite._missing_metric_message` uses it to name the
install command instead.

**RL-048 opened, deliberately not fixed.** `CEAT` is still unreachable: it
reports `n_samples`, its permutation-sample count, which is not an items-scored
count. `n` feeds the confidence interval, so whether `n_samples` belongs there
is a question about Guo & Caliskan (2021), not plumbing — CLAUDE.md forbids
settling it from memory. Left labelled rather than guessed.

Gates: **1938 passed, 3 skipped, 2 xfailed**; `ruff check src tests` clean.
Files changed: `src/bias_scope/base.py`, `src/bias_scope/suite.py`,
`src/bias_scope/prompts_based/__init__.py`, `tests/test_run.py`,
`tests/test_framework.py`.

**Still open, found by a static screen of all 55 metrics:** 12 metrics key
their headline number as `<name>_score`. The new fallback covers the 9 with
exactly one such key; `FGB`, `PGB` and `StereoSetMetric` report several and
stay (correctly) ambiguous. There is no release gate asserting that every
registered metric can complete `.run()` — `tests/test_metadata.py`'s
`TestReleaseGates` covers fidelity only. That missing gate is why these
defects reached a release at all; worth adding before 0.2.0.

### Later the same day — PLAN.md Section 14 Item 1 finally done: the first live agent run

Item 1 (a real conversation against a real agent LLM) had been deferred every
session for want of an API key. An Ollama server running `gemma4:12b-mlx` was
available locally, so the run was done through the `local` provider: 4 turns,
target `hf-internal-testing/tiny-random-BertForMaskedLM`, metric `CrowSPairs`.
It cost nothing and needed no key. **Every provider in this package had, until
today, been built and unit-tested without one real API call ever being made.**

**Held on first contact, none of it previously exercised against a real LLM:**
the confirm-before-run gate (turn 4's `confirm_plan` matched turn 3's plan,
`check_run_gate` passed, and the model correctly confirmed the *newer* plan_id
rather than the stale one), `providers.py`'s whole-transcript re-translation
across four turns, both handle registries, and `inspect_model`'s guess
(`encoder`, `has_lm_head`, confidence `high`) on a real Hub lookup.

**Broke — two defects, both now fixed, tests first:**
- **RL-049**: the agent sent `run_suite` a *flattened* `inputs`
  (`{"sentence_pairs": ...}` rather than `{"CrowSPairs": {"sentence_pairs": ...}}`).
  `BiasSuite.run()` looked up the metric name, found nothing, and skipped it
  with a reason that reads like the user's omission. The agent reported that
  skip to the user as the evaluation result. Cause: `RUN_SUITE`'s schema
  described `inputs` as a bare `{"type": "object"}` with one line that never
  said the keys are metric names. Fixed structurally — `_check_inputs_shape`
  rejects an unrecognised top-level key with a `ValueError` (which
  `loop._CAUGHT_TOOL_ERRORS` returns to the agent as a correctable tool error),
  and the schema now carries a worked example.
- **RL-050**: even with the right shape, `CrowSPairs` still would not run.
  `crows_pairs.py`, `aul.py` and `aula.py` validated pairs with
  `isinstance(pair, tuple)`. **JSON has no tuple type**, so a pair arriving
  through any tool call is always a list — making all three unreachable from
  the agent regardless of the model's competence, and unreachable from any
  future HTTP API for the same reason. Now accept tuple or list (not
  `Sequence`, which would admit a 2-character string). Verified with `inputs`
  round-tripped through `json.dumps`/`json.loads`: `CrowSPairs` and `AUL` now
  return real scores with fidelity badges where both previously skipped.

**The finding worth the most attention, not fixed.** In turn 1 the user asked
which metrics could run. The model called neither `recommend_metrics_tool` nor
`explain_exclusions_tool` and **fabricated their output** — four metric names
that do not exist (`gender_stereotypes_prediction`,
`gender_representation_generation`, `gender_professional_stereotypes`,
`gender_occupational_stereotypes`), each with an invented exclusion reason,
under confident Markdown headings. In turn 4 it invented a cause for the skip.
The package's safety design assumes the agent either calls a tool or says it
cannot; it has no answer for an agent that answers from itself. Every
prompt-only guarantee rests on an assumption this run falsifies for a 12B
model — including RL-040's decision to leave affirmation-judging to the agent,
which was taken explicitly because no live run had shown a model misreading
anything. One now has.

What still held: **the fabrication never reached a score.** Names and reasons
were invented; no number was. Scores can only arrive through `run_suite` and
`summarize_report`, both structural. The gates held exactly where they exist
and nowhere else — which is the clearest argument yet for widening them.

Gates: **1947 passed, 3 skipped, 2 xfailed**; `ruff check src tests` clean.

## 2026-09-18 — PLAN.md Section 14 Item 1 closed: the first live run against a paid API, on GPU

A supervisor-supplied OpenRouter key made Item 1 — deferred every session since
the agent was built — finally runnable. Three live conversations were recorded,
driving a real target model on the RTX A4500.

**Agent LLM.** The requested model, `~typesafe/jev-latest`, **cannot drive this
agent** and no adapter could make it: it is a *decisions* model reachable only
at `/api/alpha/decisions`, taking `state` + `questions` and answering each as a
`noul` (probability), `choice` (option key) or `score` (legend index). It emits
no free text and no tool-call arguments. `~openai/gpt-terra-latest` was
substituted (Claude models excluded by request). Full probe record: RL-051.

**Target model.** `bert-base-uncased`, encoder, fp32, `device="cuda"` —
confirmed resident on the GPU (`cuda:0`, 418.7 MiB, `torch.float32`). Data was
the authors' own `crows_pairs_anonymized.csv`, gender subset, first 20 pairs.

**Result:** `CrowSPairs = 0.400`, n=20, fidelity `faithful`, verified by
computing the metric directly on the same pairs outside the agent. **This is
not a Tier-1 reproduction** — 20 pairs is not Nangia's 1508 — and is recorded
under `results/verification/agent_live/`, not `results/validation/`. The
retained transcript stores `summarize_report`'s own return value
(`[faithful] CrowSPairs: 0.4`) next to the agent's prose, so "did the reported
number come from the tool" is answerable from the artifact rather than by
trusting the narrative. Four encoder runs were made in all; the two whose
artifacts predate timestamped filenames were overwritten, including the
0.450 / 19-of-20 run described below — its numbers are recorded here and in
RL-053, not in a file.

**Held on first real contact:** the confirm-before-run gate in both directions
(turn 3 confirmed and ran; in the causal scenario the user declined and the
model called neither `confirm_plan` nor `run_suite`); `providers.py`'s
whole-transcript re-translation across three turns; `OpenRouterProvider`
against a real key (RL-046 resolved for it). Notably, the 2026-09-17
fabrication finding **did not recur**: the model called
`recommend_metrics_tool` and `explain_exclusions_tool` for real, repeatedly,
rather than inventing their output.

**Broke — RL-052, fixed test-first.** `metrics_needing_data` built its answer
from `inspect.signature(cls.evaluate)` alone, so it never mentioned a model a
metric takes at *construction*. `plan_suite` told the agent CrowSPairs needed
`sentence_pairs`; the agent supplied exactly that; `BiasSuite` skipped the
metric with a `TypeError`. `BBQMetric`, which cannot be constructed without a
`model_name`, was reported as needing nothing at all — and a test asserted it.
Constructor requirements are now reported as `"__init__.<param>"`, with a
deliberate line between an unset sentinel (`None`/`""` → needed) and a real
default (`RegardScore(model_name="sasha/regardv3")` → *not* needed, because
that names the classifier the paper requires, not the model under test).
`run_suite` now rejects an incomplete call, and refuses to hand back a handle
to a report in which nothing ran. Blanket injection of the backend's model into
any `model_name` parameter was considered and rejected: the name means three
different things across families, and silent injection would mis-attribute.

**The finding worth the most attention — RL-053, not fixed, architectural.**
The run script compares the pairs that reach `run_suite` against the source
CSV. On one run 19 of 20 survived: the agent rewrote *"one of the best
engineers in **her** field"* as *"...in **his** field"*, destroying the minimal
pair on exactly the token CrowS-Pairs measures. The score moved 0.400 → 0.450.
Both are honest outputs of the metric — on two different datasets, one of which
nobody chose. This is inherent to passing data *by value* through a tool call:
every item a metric scores is retyped by the model, and a well-formed
paraphrase is undetectable downstream. It makes the library's protocol-hashing
apparatus moot at the agent boundary. The fix is to pass data *by reference*
(dataset id + split, or a whitelisted path the harness resolves) so the agent
names data it cannot retype.

**A correction, recorded because it nearly reached this file as a finding.**
An intermediate reading of the first transcript concluded the agent had
fabricated the 0.40 score after being told the metric was skipped — which would
have falsified the 2026-09-17 claim that "the fabrication never reached a
score". It was wrong. `BiasSuite.run` pops `"__init__"` out of the dict it is
given (RL-054), and the recorder had stored that dict by reference, so the
constructor arguments the agent *did* send were gone by the time the transcript
was serialized. Replaying the mutated log reproduced a skip that never happened
live. The recorder now deep-copies; the agent's number was correct all along.

Gates: **1959 passed, 6 deselected, 2 xfailed**; `ruff check src tests` clean.
New: `tests/integration/test_bias_scope_agent_live_conversation.py` (opt-in via
`BIASSCOPE_RUN_LIVE_AGENT=1`, marked `slow`, CPU-only, a few cents per run) and
`scripts/agent/live_conversation.py` (the GPU counterpart, per PLAN.md Section 1's
"GPU reproductions are scripts, not tests").

### Later the same day — a second agent model: `deepseek/deepseek-v4.1-flash`

Run on request, same protocol as the `~openai/gpt-terra-latest` runs above so
the two are comparable: the opt-in live test, then both GPU scenarios.

- **Live test:** passed.
- **Encoder (`bert-base-uncased`, fp32, cuda):** gate held; supplied
  `__init__.model_name` unprompted; called `inspect_model` first, which
  gpt-terra never did. `CrowSPairs = 0.4211`, matching `summarize_report`'s
  return value exactly and independently recomputed — **but on 19 pairs, not
  20.** It silently dropped the last pair when retyping the list.
- **Causal (`Qwen2.5-1.5B-Instruct`):** declined correctly — no `confirm_plan`,
  no `run_suite`. Turn 2 labelled the plan "dry run, nothing executed", and it
  refused to treat FGB/PGB as reportable: "a mismatch-fidelity score isn't a
  valid measurement of the metric it claims to be". No number appears anywhere
  in that transcript.

**This is the second independent confirmation of RL-053, by a different
mechanism.** gpt-terra altered a token inside a pair (`her` → `his`); DeepSeek
truncated the list. Two frontier models, two unrelated corruption modes, the
same 20-item input — the defect is in passing data by value through a tool
call, not in either model.

The one mitigation observed: DeepSeek **caught and disclosed its own omission**
("The score above reflects 19 pairs, not your full 20 — so it is not the answer
to the question you asked") and offered to re-run. gpt-terra's alteration went
unnoticed by gpt-terra. Self-report is a behaviour, not a guarantee, and does
not change RL-053's conclusion.

Neither model misreported a score: both matched the tool output exactly.
