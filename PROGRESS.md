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

---

## 2026-09-09 · WEAT from-scratch re-audit (no code changed)

Re-audited WEAT against Caliskan, Bryson & Narayanan 2017 (Science 356:6334,
`biasscope papers/WEAT.pdf`, Methods "Word Embedding Association Test" +
Supplement "Cosine similarity") and the sent-bias reference
(`W4ngatang/sent-bias@e3559fb`, from memory + SOURCES.yaml notes; repo copy is
git-ignored and absent locally).

**Traced:** `weat.py` (`evaluate`, `_permutation_test`, `_validate_permutation_options`,
`run`, dead `_compute_effect_size`), `_helpers._weat_effect_components` /
`_compute_similarity_measure`, `utils.cosine_similarity`, `base.EmbeddingMetric`
(`_interval`, `_count_items`, guards), `metadata`/`_metric_info` WEAT row,
`stats.hedges_olkin_ci` / `permutation_p`, `tests/test_embeddings/test_weat*.py`,
`tests/oracles/weat_oracle.py`, `tests/golden/weat.json`, docs + example.

**Result: FAITHFUL WITH DOCUMENTED EXTENSIONS.** Canonical static-embedding path
(effect size d, ddof=1; one-sided strict `>` permutation p-value; exact
enumeration of all C(2n,n) partitions ≤ 100k) reproduces an independent
from-scratch implementation to 0.0 abs difference over 60 random inputs
(effect size and exact p-value both). Null / swap-antisymmetry / scale /
permutation properties hold numerically.

**Findings (all Minor; none Critical/Major):**
1. Sampled permutation p (|X|=|Y|≥10 only) uses `count/n_samples` in strict mode
   — biased, can return exactly 0; should be `(1+count)/(1+n_samples)`.
   weat.py:329-343. Caliskan's own n≤8 tests always hit the exact branch.
2. `stats.permutation_p` docstring claims to be "the test WEAT and SEAT define"
   but computes a two-sided |Δmean| test; paper/impl use one-sided on the sum
   statistic. Public, exported, unused internally. stats.py:184-203.
3. `WEAT._compute_effect_size` (weat.py:370-402) is dead code duplicating the
   effect-size formula.
4. `run()` Hedges–Olkin CI SE ≠ the SE in `scripts/experiments/finalize_emnlp.py`
   (already RL-016); not the paper's statistic (permutation p is, and is
   reported).
5. `BiasResult.n` for WEAT is |X|+|Y| (16), not the paper's per-group N_T (8).
6. Provenance: paper prose ("permutation of the attribute words", "observed or
   greater") contradicts its own formal equations (partition X∪Y, strict `>`);
   BiasScope correctly follows the equations + reference code — worth stating in
   `docs/fidelity/weat.md`.

No REVIEW_LATER IDs created (audit only; findings listed here for the maintainer).

---

## 2026-09-09 · SEAT from-scratch re-audit (no code changed)

Audited SEAT against May, Wang, Bordia, Bowman & Rudinger 2019 (NAACL,
`biasscope papers/SEAT.pdf`, §"The Sentence Encoder Association Test" +
**Appendix A** "Computation of P-value and Effect Size" + Appendix C pooling
table) and the authors' own reference `W4ngatang/sent-bias@e3559fb`
(`sentbias/weat.py::p_val_permutation_test`, `encoders/bert.py::encode`; repo
copy git-ignored / absent locally, reasoned from the paper + WEAT audit).

**Traced:** `seat.py` (delegates wholesale to `WEAT`), `weat.py._permutation_test`
/ `_weat_effect_components`, `base.EmbeddingMetric` (`run`, `_interval`,
`_count_items`), `_metric_info` SEAT row, `tests/test_embeddings/test_seat.py`
(6 thin tests, no oracle/golden/property), docs/api + fidelity note + example.

**Result: PARTIAL IMPLEMENTATION.** Effect size d is faithful (inherits WEAT:
mean-of-cos, ddof=1 — Appendix A says "identically" to Caliskan). But:

- **Major:** the permutation p-value uses Caliskan's strict `>` (WEAT default
  `tie_policy="strict"`), whereas SEAT Appendix A *explicitly* switches to the
  non-strict `≥` ("the more conservative non-strict inequality") and floors p at
  1e-5. `SEAT.evaluate` / `SEAT.run` expose no `tie_policy` / `n_permutation_samples`
  / `permutation_seed` (no `**kwargs`), so the paper's convention is
  unreachable. Counterexample (n=4, maximally separated): BiasScope SEAT p=0.0;
  reference (`≥`) p=1/70≈0.0143. `docs/fidelity/seat.md` affirmatively
  misstates this ("same one-sided permutation p-value. Only the inputs differ").
- **Minor:** `SEAT` doesn't override `run()` → `run(seed=)` never reaches the
  permutation RNG (stays 42) and `protocol` omits `permutation_seed` (WEAT does
  both); sampled path (unreachable) would use 10k draws w/o the +1, not the
  reference's 100k+1; `fidelity="faithful"` + empty `deviation_note` despite the
  undocumented p-value deviation and un-generated templates (should be
  `adaptation`); `weat_score` key + Caliskan-branded `p_value_note` leak into
  SEAT details; `test_calls_weat` locks in "SEAT == WEAT exactly"; default
  `pooling="cls"` paired with a mean-pooling default model; example invents a
  non-paper attribute template.
- SEAT's defining contribution (bleached templates + per-encoder pooling) is
  not implemented — caller supplies sentence embeddings. Disclosed in docstring
  / docs / fidelity note.

### Fix applied (same session)

- `src/bias_scope/embeddings_based/seat.py`: `evaluate` now delegates with
  `tie_policy="conservative"` (May et al.'s `>=`) and
  `n_permutation_samples=100_000` (SEAT_PERMUTATION_SAMPLES; the paper's
  99,999 + 1), and exposes `tie_policy` / `n_permutation_samples` /
  `permutation_seed` params. `run()` overridden to thread + record
  `permutation_seed` (mirrors `WEAT.run`). `__init__` validates `pooling`.
  `weat_score` key dropped from the returned details.
- `_metric_info.py`: SEAT keeps `fidelity="faithful"` (matches reference for
  both effect size and p-value on precomputed embeddings, as WEAT does) but
  now carries a non-empty `deviation_note` covering the `>=` convention and
  the un-reproduced templates/pooling.
- `docs/fidelity/seat.md`, `docs/api/embeddings/seat.md`: corrected the
  "p-value unchanged" claim; documented Appendix A and the new defaults.
- Tests: `tests/test_embeddings/test_seat.py` rewritten (seeded; new
  `TestSeatPermutationConvention` + `TestSeatRun`); `tests/oracles/test_seat_oracle.py`
  (differential vs the WEAT oracle, 200 inputs, 1e-8); `tests/golden/seat.json`
  + generator + `test_golden.py::test_seat_score_and_pvalue_have_not_drifted`.
  `examples/embeddings_based/seat.py` uses a real bleached template.
- Verified: SEAT n=4 maximally separated now p = 1/C(8,4) (was 0.0); effect
  size unchanged; `SEAT` == `WEAT` effect size still holds; full embeddings +
  framework + metadata + run + oracle + golden + properties + examples suites
  green (717 passed). `DECISIONS.md` + `CHANGELOG.md` updated.

No REVIEW_LATER IDs created.

---

## 2026-09-09 · CEAT from-scratch audit (no code changed)

Audited CEAT against Guo & Caliskan 2021 (AIES, `biasscope papers/CEAT.pdf`:
§CEAT, §Random-Effects Model, Appendix "Random-Effects Model Details",
Table 1/2) and the authors' reference `weiguowilliam/CEAT@497e2958` — **cloned
and read this session** (`code/ceat.py`: `effect_size`, `ceat_meta`).

**Traced:** `ceat.py` (`__init__`, `evaluate`, `_sample_context_indices`,
`_rng_for_stimulus`, `_prepare_group`), `_helpers._weat_effect_components` /
`_ceat_random_effects`, `base.EmbeddingMetric._interval` / `_count_items`,
`_metric_info` CEAT row, `tests/test_ceat.py`, docs/api + fidelity note + example.

**Verified faithful (exact vs reference `ceat.py`):**
- per-sample ES = `delta_mean / std(s, ddof=1)`; in-sample `V_i = std(s)**2`
  (matches paper AND `ceat.py:174-177`) — agree to 1e-16.
- DerSimonian-Laird pooling: `W=1/V`, `Q=ΣWE²-(ΣWE)²/ΣW`, `c=ΣW-ΣW²/ΣW`,
  `τ²=max(0,(Q-(N-1))/c)`, `v=1/(V+τ²)`, `CES=ΣvE/Σv`, `SE=√(1/Σv)` — agree
  to 1e-16 with a fresh port of `ceat_meta`.
- default `n_samples=10_000` matches the paper's main N (RL-021, which says the
  default is 100, is STALE — code was reworked in 6f91e68).

**Findings:**
- **Major M1:** `CEAT().run(seed=42)` is non-reproducible — no `run()` override
  threads `seed→random_seed`, and the `random_seed=None` fallback uses OS
  entropy (`np.random.SeedSequence().entropy`). Two calls: CES 0.045 vs -0.073.
  `evaluate(random_seed=42)` is fine. Same class as the SEAT `run()` fix.
- **Major M2:** `CEAT().run()` attaches `hedges_olkin_ci(CES, |X|, |Y|)` and
  `ci_method="hedges_olkin"`, `n=|X|+|Y|` — discards CEAT's own `SE(CES)`.
  Example: reported CI width 3.2 vs the random-effects CI width 0.25 (13×). p is
  correct. CEAT needs its own `_interval` from `details["standard_error"]`.
- **Major M3:** sampling diverges from the reference undocumented. `ceat.py:220`
  uses `np.random.randint` (with replacement, always); BiasScope uses
  `rng.choice(..., replace = n_contexts < n_samples)` — the paper *text*, not the
  code. PLAN §1 says follow the code. Diverges for any stimulus with n_s ≥ N;
  breaks Tier-2 equivalence. Fix: `sampling=` param, default = reference.
- **Major M4:** `docs/fidelity/ceat.md` + `SOURCES.yaml` note describe code that
  no longer exists — `V_i = 2/n + ES²/(4n-4)` (wrong; it's `std²`), default
  `n_samples=100` (wrong; 10_000), `pooling="cls"` (wrong; CEAT raises on
  `pooling`), function `_compute_random_effects_weights` (wrong name). Provenance
  is not actually established by the note.
- **Minor m1:** p-value is two-sided `2[1-Φ(|z|)]` (matches paper Appendix +
  Table 1); reference `ceat.py:261` is `norm.sf(z)` — one-sided, signed, no abs
  — an apparent bug in the reference script. BiasScope's choice is correct;
  document it (`verify`).
- **Minor:** per-stimulus SHA-seeded RNG (extension, statistically equivalent);
  one degenerate sample aborts the whole run (reference yields nan); `|A|=|B|`
  not enforced though the paper requires it; `CEAT().run()` has zero test
  coverage.

**Verdict: FAITHFUL WITH DOCUMENTED EXTENSIONS for `evaluate()`** (core CES +
meta-analysis are an exact reproduction of the reference). `run()` carries M1+M2;
M3 + M4 undocumented. CWE-extraction pipeline not implemented (disclosed).

No REVIEW_LATER IDs created; RL-021 should be closed as stale.

---

## 2026-09-15 · CEAT `run()` fix (Major findings M1/M2 from the CEAT audit)

Fixed the two Major, code-level findings from the 2026-09-09 CEAT audit above;
left the two `verify`-tagged reference-script divergences (sampling scheme,
p-value formula) documented rather than changed, per that audit's own
recommendation.

**Test-first:** added `TestCeatRun` to `tests/test_embeddings/test_ceat.py`
(6 tests) proving, before the fix: `run(seed=)` doesn't reach `random_seed`
(non-reproducible), `ci_method` is `"hedges_olkin"` on the wrong basis, and
`n` is `|X|+|Y|` not `n_samples`. 5 of 6 failed pre-fix as expected.

**Fix:**
- `CEAT.run()` — new override threading/recording `random_seed` (mirrors
  `WEAT.run`/`SEAT.run`).
- `CEAT._interval()` — new override returning
  `(CES - Z_95*SE, CES + Z_95*SE)`, `"random_effects"`, from
  `details["standard_error"]` (stashed via a new `_call_evaluate` override),
  falling back to the base behaviour only if `standard_error` is absent.
- `CEAT._count_items()` — new override: `n = n_samples` when present.
- `bias_scope.result.make_protocol` / `PROTOCOL_KEYS` — added a `random_seed`
  field (distinct from `permutation_seed`, which is WEAT/SEAT's permutation-test
  seed) so `CEAT.run`'s `protocol_kwargs["random_seed"]` has somewhere to go.

**Docs/metadata brought in sync with the code** (M4 from the audit):
`docs/fidelity/ceat.md` rewritten against the current implementation (correct
`V_i` formula, correct `n_samples=10_000` default, correct function names,
the two `verify` divergences from the reference script, the `run()` fix).
`_metric_info.py` CEAT `deviation_note` populated (was empty). `sources/SOURCES.yaml`
CEAT note rewritten. `REVIEW_LATER.md` RL-021 closed as stale (already resolved
in code before this session); RL-038/RL-039 added for the two `verify` items.
`DECISIONS.md` and `CHANGELOG.md` updated.

**Verified:** `tests/test_embeddings/test_ceat.py` 20/20 passed after the fix.
Broader regression (`test_embeddings/`, `test_metadata.py`, `test_run.py`,
`test_framework.py`, `oracles/`, `golden/`, `properties/`, `test_examples/`,
`test_multilingual.py`): 752 passed, 1 skipped (fairlearn), 1 xfailed
(pre-existing FGB/PGB), 0 failed. `evaluate()`'s CES and p-value are untouched
by this change — they already matched the reference implementation exactly.

REVIEW_LATER: RL-021 closed; RL-038, RL-039 created (both `verify`).
