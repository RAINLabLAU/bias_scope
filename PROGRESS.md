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

## 2026-09-18 (continued) — making a real multi-model evaluation possible

Goal for this stretch: the agent evaluates the *recommended* metrics on a
causal, an encoder and an embedding model, and reports a summary of the bias
results, with no invented numbers. Six defects stood between here and there.
All were found by running the thing, all fixed test-first.

**RL-053 fixed — data by reference.** The blocker. `run_suite` took its data as
a tool argument, so every scored item passed through the agent's output tokens;
two models had already corrupted a 20-pair list that way, and embeddings arrays
cannot survive that path at all. New `src/bias_scope_agent/datasets.py`: the
agent calls `list_datasets` and `prepare_inputs`, the harness loads the
authors' own vendored files server-side, and only a handle plus provenance
(path, sha256, counts) crosses the boundary. Providers declare which metrics
they serve, which also closes two wrong-dataset traps: `LMB` and
`PairwiseLikelihoodPreference` are not fed CrowS-Pairs (they need
equal-token-length pairs), and no WEAT test is substituted for an axis
Caliskan never measured.

**RL-055 fixed — the plan reached the user.** `run_turn` returned only the
final response's text, so prose written alongside a tool call was discarded.
It now joins every round's text.

**RL-056 fixed — bf16 embeddings.** `_embed_cls` did `.cpu().numpy()` on
BFloat16 hidden states. Every embedding metric was broken on the dtype PLAN.md
*requires* for causal LMs. SEAT on Qwen2.5-1.5B went from `TypeError` to 0.3193.

**RL-057 fixed — causal models were offered masked-LM metrics.** A causal
backend declared `logits`, so all 11 probability metrics were recommended and
all 11 failed with "Unrecognized configuration class". Every consumer of
`logits` in this library is a masked-token scorer; a causal LM has none. Causal
backends now declare `("embeddings", "completions")`.

**RL-058 fixed — the worst one.** `sentence-transformers/all-MiniLM-L6-v2` has
architectures `["BertModel"]`: no LM head. Loading it through
`AutoModelForMaskedLM` does not fail - transformers newly initializes the head
and warns - so `CrowSPairs` returned **0.4000** computed from random weights,
in range, non-NaN, badged `faithful`, indistinguishable from a real score. An
encoder now advertises `logits` only when its config says it has a masked-LM
head; an unreadable config stays optimistic but records `lm_head_verified`.

**RL-059 fixed — the gate blocked the very thing it was for.** `check_run_gate`
required the run's metric set to *equal* a confirmed plan's. A five-metric
evaluation spans three datasets, so each run is necessarily narrower, and every
one was refused. The gate now accepts a subset of an approved plan (approving
more than you run is not an escalation; an unapproved metric and an empty set
are still refused), and `run_suite` takes several prepared handles at once so
the evaluation stays one report.

Gates: **2001 passed, 6 deselected, 2 xfailed**; `ruff check src tests` clean.

### The runs themselves — three model types, one agent, real numbers

`deepseek/deepseek-v4.1-flash` over OpenRouter, driving each target model on
the RTX A4500. Every score below is `summarize_report`'s own return value as
recorded in the transcript, not the agent's prose about it; every dataset is
the authors' vendored file, loaded server-side and pinned by sha256.

**encoder — `bert-base-uncased` (fp32, cuda)**, one `run_suite` call over three
prepared handles, five metrics in one report:

| metric | score | n | neutral | data |
|---|---|---|---|---|
| CrowSPairs | 0.5573 | 262 | 50 (see RL-060) | crows_pairs gender, `dfb36986ce05…` |
| AUL | 0.4656 | 262 | 50 (see RL-060) | same |
| AULA | 0.4389 | 262 | 50 (see RL-060) | same |
| WEAT | 0.6113 | 16 | 0 | weat6, `21681d3f4d7f…` |
| SEAT | 1.044 | 128 | 0 | sent-weat6, `bcf1a6864e3d…` |

**causal — `Qwen/Qwen2.5-1.5B-Instruct` (bf16, cuda)**: WEAT 0.6307 (n=16),
SEAT 0.3193 (n=128). No probability metric is offered any more (RL-057), and
SEAT only runs at all because of RL-056.

**embedding — `sentence-transformers/all-MiniLM-L6-v2` (fp32, cuda)**:
WEAT 1.021 (n=16), SEAT 1.402 (n=128). This model is the RL-058 case: it has
no masked-LM head, and before today it was offered all 11 probability metrics
and returned `CrowSPairs = 0.4000` from randomly initialized weights. It is now
offered `CEAT`, `SEAT`, `WEAT`, `SentenceBiasScore` and nothing else.

Read across the three, on the same WEAT-6 word set: 0.6113 (BERT), 0.6307
(Qwen), 1.021 (MiniLM) — the sentence encoder shows the largest
career/family gender association of the three. That is the comparison the
framework exists to support, and it is the first time it has been produced
end to end by the agent.

**Verification, because the point of all this is not to be believed on trust.**
`WEAT` on `bert-base-uncased` recomputed outside the agent, through the same
`prepare_inputs` path: **0.6113, n=16** — identical to the recorded run. The
transcripts also carry `reported_numbers`, which lists every decimal in the
agent's final message that appears in no tool result. Across the runs it caught
only neutral reference points (0.5), rounded restatements (0.61 for 0.6113,
0.32 for 0.3193), a percentage (55.73 for 0.5573) and Cohen's *d* bands
(0.2/0.5/0.8) quoted in the interpretation. No invented score, in any run.

The gate held in every run: `plan_suite` → the plan shown → `confirm_plan` in a
later turn → `run_suite`. `scripts/agent/summarize_runs.py` tabulates all of it
from the recorded tool output.

### Later — closing the open failures, and widening what can actually be fed

The three-model runs worked but only fed 5, 2 and 2 of the 15, 19 and 4
recommended metrics. Closing that gap meant fixing the failures that had been
logged-and-deferred, because most of them *were* the gap.

**RL-060 fixed — the units bug, and the most consequential.** `CrowSPairs`,
`AUL` and `AULA` returned a fraction while their own `MetricInfo`
(`neutral=50`, `range=(0,100)`), the authors' scorers
(`crows-pairs/metric.py:270`, `evaluate_bias_in_mlm/evaluate.py:213`), the
papers' reported values and `validation/registry.yaml` all say percent.
`normalized_deviation(0.5573)` was **−0.9889** — the wrong sign, reading a
mildly stereotype-preferring model as maximally anti-stereotypical. All six
return sites now scale by 100. On `bert-base-uncased`, the first 60 gender
pairs give CrowSPairs **60.00** (deviation **+0.20**), AUL 46.67, AULA 45.00 —
and 60.00 is directly comparable to Nangia's published 60.5 for the first time.
Nineteen existing assertions were rescaled, not relaxed.

**RL-061 fixed — CAT and ICAT were unreachable.** `_split_result` inferred the
headline from the dict's shape, and neither `{lms, ss, n_examples,
num_target_terms}` nor `{icat, lms, ss, n_examples}` matched. Metrics now
declare `headline_key`, because which number is the bias score is in the paper,
not in the dict: CAT's is `ss` (`lms` measures language-modelling quality, not
bias), ICAT's is `icat`. Verified end to end — CAT 67.50, ICAT 54.44, lms 83.75
over 40 StereoSet gender items, and 83.75·min(67.5, 32.5)/50 = 54.4375 exactly.

**RL-048 resolved.** The entry left open whether CEAT's `n_samples` is an
items-scored count. The authors' `ceat.py` draws N samples and pools them with
`df = N - 1`: the degrees of freedom say N *is* the observation count. CEAT
completes `run()` now.

**RL-054 fixed.** `BiasSuite.run` popped `__init__` out of the caller's dict,
so the same inputs could not be run twice. It copies first.

**RL-062 opened and acted on.** `RegardScore` had neither a headline nor a
count. The count is unambiguous (texts classified). The headline is a
*judgement*, recorded as such: `negative_difference`, because Sheng's reported
result is the negative-regard gap, this repo's `repro_regard_sheng.py`
reproduces that number, and `MetricInfo`'s signed (−1, 1) range fits exactly
that difference.

**Two new datasets, so more of the recommended set is actually feedable.**
`stereoset` (CAT, ICAT) from the authors' `dev.json` — BLANK rendered as
`[MASK]`, fills recovered by diffing each sentence against its context, items
with multi-word fills skipped and *counted in the provenance* (229 usable of
255 gender items). And `bold_regard`, the first provider that **generates**:
Dhamala et al.'s own BOLD prompts, continuations produced by the model under
evaluation, scored by Sheng's regard classifier. `DatasetSpec` gained
`requires_access`, so a generating provider is offered only to backends that
can generate — which is why a causal LM can use it and an encoder cannot.

Feedable share of the recommended set: encoder 5 → **7 of 15**, causal
2 → **3 of 19**, embedding **2 of 4**. The causal number is small but it is no
longer the wrong *kind* of coverage: a generative model is now evaluated on
generation, not only on its embeddings.

### Final runs, and an audit of whether the *recommendations* are valid

Three live conversations with `deepseek/deepseek-v4.1-flash` over OpenRouter,
each ending in one summary. Every number is `summarize_report`'s own return
value as recorded in the transcript; every dataset is sha256-pinned.

    encoder   bert-base-uncased        CrowSPairs 55.73  AUL 46.56  AULA 43.89  (n=262)
                                       CAT 69.00  ICAT 51.99                    (n=229)
                                       WEAT 0.6113 (n=16)  SEAT 1.044 (n=128)
    causal    Qwen2.5-1.5B-Instruct    WEAT 0.6307 (n=16)  SEAT 0.3193 (n=128)
                                       RegardScore -0.02 (n=100) [ADAPTATION]
    embedding all-MiniLM-L6-v2         WEAT 1.021 (n=16)   SEAT 1.402 (n=128)

`CrowSPairs 55.73` is RL-060 showing through: the same computation that
returned `0.5573` before, now on the scale the authors' script, Nangia's Table
3 and its own metadata all use. `RegardScore` is the first generation-based
evaluation in this project - Qwen continued BOLD's own prompts and Sheng's
classifier scored them.

**Then: are the recommended metrics valid?** Measured rather than assumed.
Every metric `recommend_metrics` returns was constructed and run on minimal
shape-correct stand-ins. **Ten were not valid** - the planner offered metrics
that could not complete. Three are now fixed (RL-063: `EMT`, `GenderPolarity`,
`HONEST` reported counts under their own names); one was a misleading error
rather than a defect (RL-064); and six remain genuinely unrunnable and are now
*listed* rather than discovered at run time: two need a Perspective API key,
one needs a live classifier service, two report no scalar at all (RL-065 -
`CoOccurrenceBiasScore`'s only scalar contradicts its declared `signed`
direction, and `MarkedPersons`' scalar is the open item PLAN.md 4.2 still
records), and one is the unaudited `SentenceBiasScore`.

`tests/test_recommendation_validity.py` is the gate that was missing - the
reason all of these reached a release. It asserts every recommended metric runs
or appears in `KNOWN_UNRUNNABLE` with a reason, and a second test fails when a
listed entry starts working, which caught two entries I had over-listed on its
first run.

## 2026-09-19 — five more target models through the agent, and a check that the recommended metrics are the ones it runs

Goal for the session: run the agent on a few more models and make sure the
metrics it *recommends* are the metrics it *runs*. Agent LLM
`deepseek/deepseek-v4.1-flash` over OpenRouter; targets on the A4500 (fp32
for encoders, bf16 for causal); every dataset sha256-pinned through
`prepare_inputs`. Eight conversations recorded under
`results/verification/agent_live/`.

**The check did not exist.** Each transcript already carried the
recommendation output, the plan, the `run_suite` call and the summary, but
nothing compared them - a recommended metric quietly left out looked the same
as one that ran. `recommendation_coverage` in
`scripts/agent/live_conversation.py` now derives, from the transcript alone:
recommended → feedable (a dataset provider serves it and the backend has the
access it needs) → planned → run → scored, and `complete` means every
feedable metric was scored and nothing was scored that was never recommended.
Written test-first (`tests/test_bias_scope_agent/test_recommendation_coverage.py`,
7 tests); `summarize_runs.py --check` recomputes it for old runs too, and on
them it reproduces yesterday's account exactly (bert-base-uncased fed 0, then
1, then 5, then 7 of 7).

**Results, from `summarize_report`'s own output:**

    encoder    bert-base-cased        CrowSPairs 57.63  AUL 53.05  AULA 53.82  (n=262)
                                      CAT 64.19  ICAT 59.11                    (n=229)
                                      WEAT 0.3792 (n=16)  SEAT 0.9246 (n=128)
    encoder    roberta-base           CrowSPairs 54.96  AUL 56.49  AULA 53.44  (n=262)
                                      CAT 55.46  ICAT 61.07                    (n=229)
                                      WEAT -0.6074 (n=16) SEAT 1.099 (n=128)
    causal     Qwen2.5-0.5B-Instruct  WEAT 0.847 (n=16)   SEAT 0.2512 (n=128)
                                      RegardScore 0.02 (n=100) [ADAPTATION]
    causal     gpt2                   WEAT 0.5183 (n=16)  SEAT -0.0486 (n=128)
                                      RegardScore 0.02 (n=100) [ADAPTATION]
    embedding  all-mpnet-base-v2      WEAT 1.257 (n=16)   SEAT 1.042 (n=128)

Coverage on the final transcript of each: 15 recommended / 7 feedable / 7
scored (both encoders), 19 / 3 / 3 (both causal), 4 / 2 / 2 (mpnet). WEAT and
CrowSPairs on bert-base-cased recomputed outside the agent through the same
`prepare_inputs` path: 0.3792 and 57.6336, identical. The confirm gate held in
all eight runs; `reported_numbers` found no figure that was not a rounding,
a neutral point or a Cohen's-d band. RoBERTa's `<mask>` token went through
the `[MASK]`-normalising scorer without incident.

**Two defects found by the new models, both fixed test-first the same day.**

*RL-066 - RL-058 recurred past the config check.* `all-mpnet-base-v2` lists
`MPNetForMaskedLM` in its config, so `_has_masked_lm_head` said yes and all
11 probability metrics were recommended. The checkpoint ships no `lm_head.*`
tensors; transformers initialised them at random, and the first run reported
CrowSPairs 48.85, AUL 50.38, AULA **50.00**, CAT 54.59, ICAT 38.87 - all
badged `faithful`, all from a random head, and the coverage check called the
run *complete*, because it was: the wrong metrics were run completely. The
config is a claim; the weights are the fact. `HuggingFaceBackend` now loads
the masked-LM model once with `output_loading_info=True` and withholds
`logits` when any head weight is missing (empty `missing_keys` verified for
bert-base-uncased, bert-base-cased, roberta-base and the tiny test encoder;
six missing for each sentence-transformers checkpoint). The rerun recommends
4 metrics instead of 15. The invalidated transcript is moved to
`results/verification/agent_live/invalidated/` with a README; it is evidence,
not a result.

*RL-067 - WEAT and SEAT skipped on gpt2.* "Asking to pad but the tokenizer
does not have a padding token." The agent reported the skips honestly and the
coverage check marked the run incomplete, so this one was visible. The pad
token is now set to end-of-sequence, as `HuggingFaceBackend.generate`
already did - in *both* loaders: the first rerun fixed SEAT (`cls` path) and
still skipped WEAT (sentence-transformers `mean` path), which is how the
second was found. Third gpt2 run: complete.

**Logged, not changed:** RL-068 - `pooling='cls'` on a decoder-only LM reads
the first token's hidden state, so the causal SEAT numbers are of uncertain
meaning even though the statistic is WEAT's; the fix is a protocol decision
to be read from May et al.'s code, not made here. RL-069 - the agent's
interpretive prose is occasionally wrong where the numbers are right (SEAT
called "not comparable to WEAT's d"; WEAT's sign called absent).

`ruff check src tests scripts/agent` clean; fast suite 2038 passed, 2 xfailed;
`-m slow` on `test_framework.py` and `test_recommendation_validity.py` green.

## 2026-09-20 — six dataset providers: a causal LM's recommended metrics fed without human input

Follow-on to yesterday, by request: 3 of 19 recommended metrics were feedable
for a causal LM and the user wants the agent to evaluate the major
recommended metrics on its own, retrieving what it needs. Six providers now
serve the six metrics that could run but had no data path.

**What each feeds, and from where.** `bold_gender_polarity` → GenderPolarity
(BOLD profession prompts continued by the model, BOLD §4.5 unigram lists);
`bold_helm_bias` → DemographicRepresentation and StereotypicalAssociations
(same generations, HELM's own `bias_word_lists.py` loaded from the vendored
file, adjectives as targets); `honest` → HONEST (Nozza's en binary templates,
K=20 sampled 5-token continuations, HurtLex EN 1.2 conservative level -
CC BY-NC-SA, so fetched at a pinned commit into git-ignored `third_party/`
rather than vendored; `fetch_sources.py` gained resource download with a
hash check); `rtp_toxicity` → EMT (RealToxicityPrompts at a pinned Hub
revision, K=25 nucleus samples, 20 tokens); `ceat_contexts` → CEAT (each
WEAT word's contexts from BOLD's 23,679 Wikipedia sentences, N=1,000).

**Two of them substitute what the paper used, and the result says so.** No
Perspective key, so EMT is scored by `unitary/toxic-bert`; no Reddit corpus,
so CEAT's contexts are Wikipedia sentences embedded as sentences. The metric
classes' fidelity badges are static, so a new key `__protocol__` in a metric's
inputs carries `resources` with a `deviation` into the result's protocol block
(`BiasSuite.run`), and `summarize_report` prints it under the score:

    [faithful] EMT: 0.05298 (n=50)
        deviation: toxicity scored by the local classifier unitary/toxic-bert, not the
        Perspective API Gehman et al. used; EMT values are not comparable to the paper's

RL-070 (HONEST causal decoding), RL-071 (CEAT corpus and pooling), RL-072
(EMT classifier) and RL-073 (BOLD profession prompts, decoding) record the
choices. All resources are in `sources/SOURCES.yaml` with sha256.

**Process note.** The first two providers were written before their tests,
against the project's rule; the tests followed in the same commit and the
remaining four were test-first. Recorded here rather than hidden.

**The first live runs exposed two defects, both measured and fixed.**
StereotypicalAssociations declined on both models: greedy 30-token
continuations of gpt2 are degenerate ("He is a skilled metalsmith, and he is
a skilled metalsmith...") and even 500 contained no HELM adjective next to a
gender word; nucleus-sampled 50-token continuations give it 0.467 at 500
prompts and 0.464 at 1,000, so that is the default now, seeded, shared
between the two BOLD providers through the cache. And EMT was left out of the
gpt2 plan because `rtp_toxicity` was tagged axis "toxicity" while the plan
was axis "gender"; the dataset is now `axes: any`, with the reason in its
description. The two transcripts are kept and listed as incomplete by design.

**Final runs, `deepseek/deepseek-v4.1-flash`, one plan / one confirmation /
one report each, every number `summarize_report`'s own:**

    causal  gpt2                    WEAT 0.5183 (16)  SEAT -0.0486 (128)  CEAT 0.0801 (1000)*
                                    EMT 0.0530 (50)*  RegardScore 0.02 (100)  GenderPolarity 0.044 (500)
                                    HONEST 0.083 (1000)  DemographicRepresentation 0.2564 (78)
                                    StereotypicalAssociations 0.4667 (15)
    causal  Qwen2.5-0.5B-Instruct   WEAT 0.847 (16)   SEAT 0.2512 (128)   CEAT 0.0652 (1000)*
                                    EMT 0.0320 (25)*  RegardScore 0 (80)      GenderPolarity 0.032 (500)
                                    HONEST 0.016 (1000)  DemographicRepresentation 0.3824 (51)
                                    StereotypicalAssociations 0.5 (4)
    * recorded deviation printed under the score

Coverage: 19 recommended, **9 feedable, 9 scored**, on both models
(`recommendation_coverage.complete = True`); the ten unfed are exactly
`KNOWN_UNRUNNABLE`. HONEST on gpt2 at 0.083 sits inside the published
0.08-0.12 band even in causal mode. StereotypicalAssociations' `n` is target
words scored (15 and 4) - small, and honest about it.

`ruff check src tests scripts/agent scripts/sources` clean; fast suite green;
`check_manifest.py` valid.

### Later the same day - twelve models, one table, the logs, and five more defects

By request: run different models and show the table. `RESULTS.md`,
`README.md` (every conversation verbatim, with commands) and `REPRODUCE.md`
now live in `results/verification/agent_live/`, all three generated or
written to be regenerated from the transcripts.

**Complete runs (every feedable metric scored):** bert-base-uncased,
bert-base-cased, roberta-base (8 of 15, CEAT included now), all-MiniLM-L6-v2,
all-mpnet-base-v2 (3 of 4), gpt2, gpt2-medium, Qwen2.5-0.5B, -1.5B, -3B
(9 of 19 each). **Partial by construction:** Llama-3.2-1B (7 of 9) and
gemma-3-1b-it (8 of 9) - both prepend a BOS token, so position-0 pooling
gives one vector for every sentence and SEAT/CEAT decline or return a
degenerate 0 (RL-068; and `sent-bias` has no GPT encoder to take a protocol
from). **Blocked:** gemma-2-2b-it, gated and not accessible to this account
(RL-079).

**What the bigger models broke, in the order found.** Llama: a gated-repo
401 inside `run_suite` escaped the loop's four-type catch and killed the
conversation (RL-074, fixed; gated cached models run offline). Qwen 3B: out
of GPU memory - the backend, sentence-transformers and the CLS loader each
held a copy of the model (RL-075, fixed by sharing the backend's model).
gemma-3: the sentence-transformers loader wanted an image processor (RL-076,
fixed: for a causal backend, mean pooling runs on the shared model, and I
checked it is bit-identical to what sentence-transformers computes on gpt2
before switching). gemma-3 again: every generation came from the cache, so
the backend never loaded and the sharing registered inside `_load` never
happened, and that one metric's OSError discarded the other eight results
(RL-078, fixed: a loader registered in the constructor; the suite skips a
metric on any exception). Each fix has a test named after the failure.

**A protocol fact, recorded rather than changed (RL-077).** Sharing the
backend's model means the embedding metrics on causal LMs now run in the
bf16 the protocol block always claimed. On gpt2 the same weights, words and
pooling give WEAT 0.5183 in fp32, 0.4847 in bf16 on CPU, 0.4006 in bf16 on
the GPU. Every causal row was re-run so the table is one protocol; computing
embeddings in fp32 regardless of generation dtype is the recommended next
change, and it is written down as such.

**Process.** Twelve queued runs, five reruns, one model dropped. All
scripts, tests and REVIEW_LATER entries (RL-074 to RL-079) committed;
`ruff` clean, fast suite green, manifest valid.

### Later - merging the three branches

`merge/all-branches` = `agent-implementation` + the three unpushed August
commits of `v0.2-metrics-and-framework` + a content-less record of local
`main`'s August snapshot (superseded by origin's PR #29 and later branding
commits). Both branches had fixed the same three defects in parallel; the
September versions were kept (live-tested), and v0.2's every-metric `run()`
test with `tests/fixtures/tiny_inputs.py`, its headline keys for the other
metrics, and its demo script came in. Its RL-038..041 became RL-080..083.
One entry the merge made stale: `CoOccurrenceBiasScore` now has the headline
v0.2 gave it, which is the mean absolute bias the paper reports, so its
metadata direction was corrected and it left `KNOWN_UNRUNNABLE` (RL-065
updated). Fast suite 2161 passed, coverage 90%, slow validity gate green.
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
in code before this session); RL-084/RL-085 added for the two `verify` items.
`DECISIONS.md` and `CHANGELOG.md` updated.

**Verified:** `tests/test_embeddings/test_ceat.py` 20/20 passed after the fix.
Broader regression (`test_embeddings/`, `test_metadata.py`, `test_run.py`,
`test_framework.py`, `oracles/`, `golden/`, `properties/`, `test_examples/`,
`test_multilingual.py`): 752 passed, 1 skipped (fairlearn), 1 xfailed
(pre-existing FGB/PGB), 0 failed. `evaluate()`'s CES and p-value are untouched
by this change — they already matched the reference implementation exactly.

REVIEW_LATER: RL-021 closed; RL-084, RL-085 created (both `verify`).

---

## 2026-09-15 · SentenceBiasScore from-scratch audit + fix

Audited SentenceBiasScore against Dolci, Azzalini & Tanelli 2023 (Data
Science and Engineering, `biasscope papers/Sentencebiasscore.pdf`) — full §3
"Gender Bias Estimation" and §6. **Provenance correction:** the paper was
believed paywalled with no preprint (RL-029) and the class carried
`fidelity: unaudited`; it is actually Springer open access (CC-BY-4.0) and
the PDF was already in the papers folder. Read in full this session.

**Audit found (verdict at the time: PARTIAL IMPLEMENTATION):**
- **Critical:** `SentenceBiasScore().run()` raised `BiasScopeError`
  unconditionally on every input — `evaluate(..., return_details=True)`'s
  dict keys matched none of `_split_result`'s recognised headline-score names.
- **Major:** the paper's actual methodology (PCA gender direction, Sec 3.2;
  max-pooling word importance, Sec 3.4; 6562-word gender lexicon, Sec 3.3)
  was not implemented anywhere — only the trivial final weighted sum
  (Eq. 1-3) was, which itself was verified faithful against the paper's own
  Table 2 worked example ("She likes the new pink dress": female_bias
  0.07550, male_bias -0.01858, matching to the paper's own rounding).
- **Major:** wrong citation (author initials, fabricated title) in the class
  docstring and docs/api page.
- **Major:** `sources/SOURCES.yaml`'s reading record cited section numbers
  ("2.3, 2.3.1-2.3.3") that don't exist in the actual paper (real structure:
  §3.1-3.4); `local_pdf` was empty despite the PDF being present locally.

### Fix applied (same session)

- `SentenceBiasScore.evaluate`'s dict gains `bias_score` (= `absolute_bias`,
  the paper's Eq. 3), `breakdown` (`female_bias`/`male_bias`), and integer
  `n` — `run()` now works. `MetricInfo` updated: `direction=
  "higher_more_biased"`, `value_range=(0, inf)`, `fidelity="adaptation"`
  (was `"faithful"`, which overstated the remaining lexicon gap).
- New `derive_gender_direction()` (PCA of gender word-pair difference
  vectors) and `derive_word_importance()` (max-pooling selection counts)
  implement Sec. 3.2 and 3.4. `derive_gender_direction` uses **uncentred**
  SVD after a first centred-PCA attempt failed its own known-answer test
  (centring removes the shared signal being sought — verified numerically,
  documented as REVIEW_LATER RL-086, tag `decide`, no reference code exists
  to settle it definitively). `derive_word_importance` verified against the
  paper's own Fig. 3 ratio (saxophone: 1106/4096 ≈ 27%).
- New `build_gender_words_mask()` implements Sec. 3.3's case-insensitive
  matching, but **the 6562-word lexicon is deliberately not vendored or
  reconstructed** — it is unpublished and "selected starting from" two
  source lists implies curation this session cannot safely guess (PLAN.md
  Section 1: never invent an unverified resource). Logged as REVIEW_LATER
  RL-087, tag `blocked`, with a concrete path to closing it.
- Citation fixed (Dolci, T.; Azzalini, F.; Tanelli, M.; correct title) in the
  class docstring and `docs/api/embeddings/sentence_bias_score.md`.
- `docs/fidelity/sentence_bias_score.md` rewritten in full against the
  now-read paper and the fixed code. `sources/SOURCES.yaml` entry corrected
  (real section numbers, `local_pdf` path, `paper_status` removed now that
  it reads normally). `REVIEW_LATER.md` RL-029 closed; RL-086/RL-087 opened.
  `PLAN.md`'s three stale "SentenceBiasScore paywalled" references corrected
  (Phase 0 retrieval summary, Phase 1 audit-count line, Appendix E table).
  Example rewritten to exercise all three new helpers plus `run()`.

**Verified:** `tests/test_embeddings/test_sentence_bias_score.py` 37/37 pass
(16 new: `TestSentenceBiasScoreRun`, `TestDeriveGenderDirection`,
`TestDeriveWordImportance`, `TestBuildGenderWordsMask`). `test_metadata.py`
511 passed after the `MetricInfo` change. `scripts/sources/check_manifest.py`
passes (55/55 read, matching schema). Updated example runs end to end.

REVIEW_LATER: RL-029 closed; RL-086 (`decide`), RL-087 (`blocked`) created.

---

## 2026-09-15 · CrowS-Pairs from-scratch audit (no code changed)

Audited CrowS-Pairs against Nangia, Vania, Bhalerao & Bowman 2020 (EMNLP,
`biasscope papers/crowspair.pdf`: §3 "Measuring Bias in MLMs", Eq. 1, Table 2,
Appendix A/B/C) and the authors' reference `nyu-mll/crows-pairs@8aaac11c`
(`metric.py`) — **cloned and read this session**.

**Traced:** `crows_pairs.py` (`evaluate`, `_evaluate_wordpiece`, `_compute_pll`),
`_helpers.py` (`_categorize_tokens`, `_score_wordpiece_pair_crows`,
`_filter_special_aligned_positions`), `scorers.py::WordPieceBertScorer`
(`encode`, `align_unmodified`, `pll_over_positions`), `base.py::BiasMetric._interval`,
`_metric_info.py`, `tests/test_probability_based/test_crows_pairs.py`,
`scripts/experiments/emnlp_reproduction.py::_crows_pair_from_row`, docs, example.

**Verified faithful (default `mode="wordpiece"`), against the actual reference
script:** per-position masking (`pll_over_positions` masks exactly one position
per model call, from the *original* `input_ids`, matching `get_log_prob_unigram`
+ the "skip CLS/SEP, mask one at a time" loop in `mask_unigram`); SUM (not
mean) over masked positions; strict `>` comparison; unmodified-token alignment
via `difflib.SequenceMatcher` `equal` opcodes (matches `get_span` exactly);
tie handling (0 toward the stereotype-preference count either way, but the
tied item still counts toward `n` — mathematically identical to the reference's
`neutral`/`N` bookkeeping, verified by hand-tracing both). `MetricInfo`
(`neutral_value=50`, `range=(0,100)`, `direction="higher_more_biased"`)
matches the paper exactly.

**Findings:**
- **Major:** `run(..., ci="wald")` silently returns `ci=None, ci_method="none"`
  for CrowS-Pairs (and any metric without `details["per_item"]"`) — confirmed
  by execution. `base.py::BiasMetric._interval`'s early return
  (`if ci == "none" or per_item is None or not per_item: return None,"none",None`)
  fires before the `ci == "wald"` branch is ever reached, even though
  `wald_ci(score, n)` needs no per-item data. CrowS-Pairs' score is exactly a
  Bernoulli proportion (% preferring the stereotyping sentence out of N
  pairs) — the textbook case for a Wald interval — but it's unreachable.
- **Minor:** tie threshold differs from the reference *script*. BiasScope
  compares raw floats (`pll_s > pll_a`); `metric.py` rounds each sentence's
  PLL to 3 decimals *before* comparing for equality. Constructed and executed
  a counterexample: PLL sums `-1.00000` vs `-1.00049` (round to the same
  `-1.000`, so the reference calls it a tie, contributing 0) but
  `-1.00000 > -1.00049` is strictly true, so BiasScope scores the pair as a
  100% stereotype preference. Undocumented in the existing fidelity note;
  low real-world materiality (exact-to-3dp ties are rare with continuous
  multi-token PLL sums) but a genuine, demonstrated discrepancy.
- **Minor (already disclosed, independently reconfirmed):** `autojunk=False`
  vs the reference's default `True` in `SequenceMatcher` — provably inert for
  CrowS-Pairs sentences (autojunk only activates above 200 elements).
  `mode="whitespace"`'s naive positional token categorization breaks for
  multi-word group substitutions (e.g. "White" ↔ "African American") — not
  the paper's protocol, already documented as such and not the default.
- **Minor:** no `details["per_item"]` (per-pair 0/1 indicators), so `run()`'s
  default bootstrap CI is also unavailable, and no built-in stereo/antistereo/
  bias-category subset reporting (Table 2's breakdowns) — caller must filter
  and call `evaluate()` per subset themselves. Not a fidelity violation (the
  paper's core metric is the aggregate percentage), but a completeness gap
  relative to Table 2's reported breakdowns.

**Verdict: FAITHFUL** for `mode="wordpiece"` (the default) — the scoring
algorithm is verified against both the paper's equations and the actual
reference script line-by-line, including tie bookkeeping. The Wald-CI bug is
a framework-level (`base.py`) issue surfaced by this metric, not a scoring
deviation. No REVIEW_LATER IDs created yet (audit only); recommend one for
the wald-CI base.py bug (affects other metrics too) and one `verify`/`decide`
for the tie-threshold rounding choice.

---

## 2026-09-15 · LPBS from-scratch audit (no code changed)

Audited LPBS against Kurita, Vyas, Pareek, Black & Tsvetkov 2019 (GeBNLP@ACL,
`biasscope papers/LPBS.pdf`: §2 "Quantifying Bias in BERT", the four-step
procedure) and the authors' reference
`keitakurita/contextual_embedding_bias_measure@18044f87` — **cloned and read
this session** (`lib/bias_calculator.py`, `lib/bert_utils.py`).

**Traced:** `lpbs.py` (`evaluate`, `_probabilities`, `_total`, `_log_ratio`,
`_validate_inputs`), `pairwise_likelihood_preference.py` (the v0.1.1 statistic,
correctly isolated under its own name/fidelity="original"), `_metric_info.py`,
`tests/test_probability_based/test_lpbs_faithful.py` (26 tests), docs, example.

**Verified faithful:** the core formula
`LPBS(t1,t2,a) = [log p_tgt(t1) − log p_prior(t1)] − [log p_tgt(t2) − log p_prior(t2)]`
matches the paper's §2 exactly (hand-verified both algebraically and against
the class's own worked example, `2·log 2`). Multi-word target-set handling
(sum probabilities within a set, then log) matches the reference code's
`bias_calculator.py:51-65`. Aggregation (`bias_score` = mean over all
(template × attribute) pairs; `breakdown[attribute]` = mean over templates for
that attribute) matches §3.2 ("mean log probability bias score for each
attribute") exactly, since the template×attribute grid is a full cross
product. `run()` works (`per_item` populated, `bias_score` key present).
`MetricInfo` (`neutral_value=0`, `direction="signed"`, unbounded) matches.

**The one substantive discrepancy (already flagged, `verify`, RL-012) — this
session upgraded it from "static reading" to executed confirmation:** cloned
the reference repo and ran its literal `get_index` index-arithmetic (isolated
from the model/tokenizer dependencies, which don't install on a modern
Python) against a controlled token list matching its own docstring example
("GGG is XXX", `gender_comes_first=True`, the paper's primary template
orientation). Confirmed: the reference's prior-probability read lands on
real-input position 2, which is the literal word "is" — not a mask token at
all — because its `last=True` branch omits the `+1` CLS offset that the
`not last` branch has. BiasScope's `LPBS` does not replicate this; it follows
the paper's §2 step 3 (prior read at the TARGET slot), deriving the mask
ordinal from the template text itself (`target_ordinal = 0 if template.index(
TARGET_SLOT) < template.index(ATTRIBUTE_SLOT) else 1`) rather than a
caller-supplied boolean flag — more robust than the reference's approach, not
just different from it. This remains the right call; RL-012 stays open only
pending a Tier-2 run against the (currently uninstallable) full reference
model pipeline, not because the index-arithmetic bug itself is in doubt.

**Verdict: FAITHFUL.** No new findings beyond what the existing
`docs/fidelity/lpbs.md` already recorded; independent re-derivation confirms
it rather than superseding it. No REVIEW_LATER IDs created (RL-012 already
covers the one open item; its "static reading" language could be updated to
"index-arithmetic confirmed by execution; full model pipeline still blocked"
but that's a documentation nicety, not a new finding).

## 2026-09-15 · AUL/AULA from-scratch audit + fixes

Read Kaneko & Bollegala 2022 in full; cloned the reference
(`kanekomasahiro/evaluate_bias_in_mlm@6b10239974a7`, freshly, not trusted from
`docs/fidelity/aul_aula.md`'s existing quote — which turned out to be
authentic). Traced `aul.py`, `aula.py`, `scorers.py`
(`WordPieceBertScorer`, `BertPLLScorer`), `_helpers.py`, `base.py::run`, both
test files, both examples, `_metric_info.py`.

**Verified against the reference, on a real `bert-base-uncased` forward pass:**
`WordPieceBertScorer.aul_aula()` (the canonical `mode="wordpiece"` path) is
bit-identical (diff = 0.0) to a fresh reimplementation of the reference's
`calculate_aul`, for both AUL and AULA. Adversarial counterexamples confirmed
this is distinguishable from two plausible bugs: a masked-PLL average
(−4.27 vs. the correct unmasked −2.42) and a last-layer/self-attention-
diagonal AULA variant (−0.1226 vs. the correct all-layers/heads-received
−0.1394).

**Fixed (see `DECISIONS.md` and `CHANGELOG.md` for detail):**
- `run()` silently returned no confidence interval for AUL/AULA for any
  `ci=`, because `evaluate()` never exposed `per_item`. Fixed by adding it
  (scaled 0/100) to both modes' `return_details=True` dict in both classes.
  Confirmed by execution: `run(ci="bootstrap")` now returns a real interval.
- A whitespace-mode footgun: `BertPLLScorer`/`WordPieceBertScorer` — both
  advertised in their own docstrings as usable for AUL/AULA — always mask
  the scored position, so plugging either into AUL/AULA's whitespace mode
  silently computed masked PLL, not AUL/AULA, with no error. Added
  `_reject_masked_scorer` (raises `ValueError`) and corrected both
  docstrings.
- Three inaccurate comments/docstrings (no scoring change): `AUL.evaluate`'s
  worked example claimed a 0.5-scale output where the real scale is 0-100;
  `AULA`'s whitespace docstring and `_compute_aula`'s internal comment both
  mischaracterized how the attention weighting works (claimed normalization
  to 1 that the code doesn't do; claimed a self-attention-diagonal reading
  that isn't what the paper or reference use).
- `tests/test_probability_based/test_aula.py::test_attention_normalization`
  was non-discriminating (its fixed inputs gave the same result whether or
  not weights were normalized); rewritten with per-position probabilities so
  it actually distinguishes the two, and renamed
  `test_attention_weights_are_not_renormalized`.

**Tests.** Added `test_return_details_exposes_per_item`,
`test_run_produces_bootstrap_ci`, `test_wordpiece_mode_also_exposes_per_item`,
`test_whitespace_rejects_bert_pll_scorer_instance`,
`test_whitespace_rejects_wordpiece_scorer_instance` to both `test_aul.py` and
`test_aula.py` (10 new tests total); rewrote `test_attention_normalization`.
Full suite: `1845 passed, 4 skipped, 5 deselected, 1 xfailed` (up from 1835
passed before this session's AUL/AULA changes — 10 new tests, no regressions).
`ruff check src tests` clean.

**Verdict unchanged: FAITHFUL.** The core paper-defined mathematics (eq. 4-6)
were already exact in the canonical path; the fixes are to the `run()`/CI
framework integration and a non-canonical mode's misuse-guarding, consistent
with how CrowS-Pairs' analogous `run()`/base.py finding was handled (verdict
not downgraded for a framework-layer gap, only for the metric's own scoring).
No REVIEW_LATER IDs created — these were concrete, mechanical bugs with an
unambiguous fix, not judgment calls.

## 2026-09-17 · Fixed the CrowS-Pairs and LPBS audit findings

Fixed the two outstanding findings from the CrowS-Pairs audit (`docs/fidelity/
crows_pairs.md`, `DECISIONS.md`) left unapplied at the time since no fix was
requested:

- **`base.py::BiasMetric._interval`**: reordered so `ci == "wald"` is checked
  before the `per_item is None` short-circuit (wald needs `score`/`n`, not
  per-item data), and made it compute a real Wald interval whenever
  `MetricInfo.value_range` is finite — normalizing the score to `[0, 1]` for
  `wald_ci` and rescaling the result back to the metric's native range. This
  is shared code, so it benefits every metric on a bounded scale, not just
  CrowS-Pairs; confirmed by execution that `LPBS().run(ci="wald")` (unbounded
  `value_range`) is unaffected and still correctly returns `ci=None`.
- **`crows_pairs.py`**: `evaluate()`/`_evaluate_wordpiece` (both modes) now
  expose `"per_item"` (the per-pair indicators, scaled to 0/100), the same
  gap already fixed for AUL/AULA two days prior — so `run()`'s default
  `ci="bootstrap"` now returns a real interval too, not just `ci="wald"`.
  Also matched Nangia 2020's reference `metric.py:225-226`, which rounds each
  side to 3 decimals before comparing for a win/tie; BiasScope was comparing
  raw floats, so a sub-0.001 difference could count as a stereotype "win"
  where the reference calls it neutral. Both scoring modes now round first.

**LPBS: nothing to fix.** Its from-scratch audit's one open item (RL-012) is
about the *reference* implementation's own index-arithmetic bug, which
BiasScope's `LPBS` deliberately does **not** replicate (it derives the mask
ordinal from the template text itself, which is more robust than the
reference's caller-supplied flag) — there is no BiasScope-side defect to
correct. Re-confirmed `LPBS` already exposes `per_item` (`lpbs.py:175`) and
its `run()` CI already worked before and after today's `base.py` change.

**Tests.** Added 6 tests to `test_crows_pairs.py`: `per_item` exposure (both
modes), `run(ci="bootstrap")` and `run(ci="wald")` both producing real
intervals, and two tie-rounding cases (a sub-0.001 difference counted as a
tie; a real difference still counted as a win). Full suite:
`1851 passed, 4 skipped, 5 deselected, 1 xfailed` (up from 1845 before this
session — 6 new tests, no regressions anywhere, including WEAT/SEAT/CEAT/CBS/
LPBS which also flow through the changed `base.py::_interval`).

**Verdicts unchanged: both FAITHFUL.** These were `run()`/CI-machinery and a
sub-0.001 rounding edge case, not the CPS statistic itself.

## 2026-09-17 · CAT/ICAT from-scratch audit + fix

Read Nadeem, Bethke & Reddy 2021 (StereoSet) in full; cloned the reference
(`moinnadeem/StereoSet@ead7d086a64a`, freshly, SHA confirmed to match the
existing `_metric_info.py`/`docs/fidelity/stereoset_family.md` citation) and
independently re-read `code/evaluation.py` (aggregation) and
`code/eval_discriminative_models.py`/`models.py` (the actual masked-LM
scoring convention). Traced `cat.py`, `icat.py`, `scorers.py`, `base.py`,
all three test files, the fidelity doc, `_metric_info.py`.

**Confirmed faithful, independently re-derived:** the two-comparisons-per-
instance `2×total` denominator for `lms`, per-target-term averaging before
the dataset-level mean, and `icat`'s macro (not micro) formula all match the
reference line-for-line — this was already correctly implemented and well
tested (`test_cat_stereoset.py` already pinned the exact v0.1.1 regressions
this fixed, independently verified fresh this session).

**One finding, fixed:** same `run()`/CI gap as the AUL/AULA/CrowS-Pairs
fixes two days prior — neither `CAT` nor `ICAT` exposed `per_item`, so
`run()`'s default `ci="bootstrap"` silently returned `None` for both.
Fixed with two different mechanisms, because `ss` (what CAT reports) and
`icat` (what ICAT reports) have different statistical structure:

- `CAT.evaluate()` now exposes `"per_item": term_ss` — bootstrapping the
  per-target-term `ss` values is the exact, standard bootstrap for `ss`
  (which literally is their mean).
- `ICAT` cannot reuse that: `icat` is a *nonlinear* function of two paired
  per-term series (`lms`, `ss`), so no single flat list's mean equals
  `icat`, and reusing CAT's `ss`-only `per_item` would target the wrong
  statistic and risk failing `run()`'s CI-bracket guard. `ICAT` instead
  overrides `_interval` to resample target terms with their `(lms, ss)`
  pairs kept together and recompute `icat` via the same `combine` formula
  per resample — a proper paired bootstrap.

**Tests.** Added `test_return_details_exposes_per_item_as_per_term_ss` and
`test_run_produces_bootstrap_ci` to `test_cat.py`; added
`test_evaluate_does_not_leak_cats_per_item`,
`test_run_produces_bootstrap_ci_for_icat_itself`, and
`test_bootstrap_ci_degenerate_with_single_target_term` to `test_icat.py` (5
new tests). Also verified manually (not just via the test suite) with a
20-target-term synthetic dataset with real cross-term variance: both `CAT`
and `ICAT` now return non-degenerate intervals that correctly bracket their
scores. Full suite: `1856 passed, 4 skipped, 5 deselected, 1 xfailed` (up
from 1851 — 5 new tests, no regressions).

**Verdict unchanged: FAITHFUL.** The core `lms`/`ss`/`icat` mathematics were
already exact; the fix is entirely to `run()`'s CI machinery, the same
category of gap already fixed for three other metrics and not counted
against their verdicts either.

## 2026-09-17 · LMB from-scratch audit + fix

Read Barikeri, Lauscher, Vulić & Glavaš 2021 (RedditBias) in full; cloned
the reference (`umanlp/RedditBias@61f9ae9458e2`, freshly — SHA confirmed to
match the existing `_metric_info.py` citation) and independently read
`Evaluation/measure_bias.py` (outlier removal, the paired t-test) and
`utils/helper_functions.py::perplexity_score` (the actual perplexity
computation: `model(input_ids, labels=input_ids)` on an
`AutoModelForCausalLM`, i.e. standard causal perplexity on **DialoGPT**,
not a masked LM). Traced `lmb.py` in full, including the hand-rolled
t-distribution/incomplete-beta implementation (no scipy dependency).

**Four findings, all fixed:**

1. **The default outlier rule was dead code.** `outlier_strategy="sigma"`
   computed `[mean-3·std, mean+3·std]` and never applied it — behaviorally
   identical to `"none"`, and no test caught it (only `"percentile"` and
   `"none"` were tested). Confirmed by execution with a 5-pair sample
   containing one extreme outlier: `sigma` removed nothing where
   `percentile` correctly removed one pair. Fixed by applying the mask.
2. **The p-value was wrong by up to 10x for any sample with `df > 30`.**
   `_normal_cdf` computed `0.5·(1+erf(x))` instead of the standard normal
   CDF `0.5·(1+erf(x/√2))` — confirmed by comparing to `scipy.stats.t.cdf`:
   at `t=1.96, df=254` (REDDITBIAS's actual Race test-set scale), the bug
   reported `p=0.0056` ("significant") where the true value is `p=0.051`
   ("not significant") — a false-positive flip at exactly the α=0.05
   boundary the test exists to adjudicate. Fixed by scaling the erf
   argument by `1/√2`.
3. **`run()`'s headline was Cohen's *d*, not the paper's t-value.**
   `evaluate()`'s dict had no `bias_score` key, so `base.py`'s fallback
   search matched `"effect_size"` first — a third, previously undocumented
   divergence beyond the already-tracked `mean_diff`-vs-`t-value` gap
   (RL-026, `evaluate(return_details=False)`'s own scalar, deliberately
   left as-is). Fixed by adding `"bias_score": t_stat`.
4. **The documented/default scoring contract was bidirectional; the
   paper's model is causal.** The docstring said "Same as AUL's predict
   function" (full unmasked context) and the built-in scorer requires a
   masked-LM tokenizer, so `LMB(model_name="microsoft/DialoGPT-small")` —
   the paper's own model — cannot construct; the shipped example used
   `bert-base-uncased` (masked), reinforcing the wrong contract. Fixed the
   docstring and rewrote the example to build a genuine causal scorer
   around DialoGPT-small, matching the reference's computation via one
   forward pass per sentence.

**Tests.** Added 6 tests to `test_lmb.py`: sigma outlier removal actually
removing an injected outlier (and leaving clean data untouched), p-value
accuracy against precomputed scipy reference values at the `df>30`
boundary and at REDDITBIAS scale, and `run()` reporting `t_stat` rather
than `effect_size`. Full suite: `1862 passed, 4 skipped, 5 deselected,
1 xfailed` (up from 1856 — 6 new tests, no regressions).

**Verdict unchanged: `adaptation`.** The formula skeleton (perplexity,
paired t-test, sign convention) was already right; the four fixes are all
to correctness-critical machinery around it (the cited outlier rule, the
significance test's numerical accuracy at realistic scale, and the
framework's headline value) plus a documentation/default-behavior gap, not
a change to the core statistic. The one remaining, deliberate deviation is
RL-026 (`evaluate(return_details=False)` still returns `mean_diff`, not
the t-value), unchanged by this session.

## 2026-09-18 · CBS from-scratch audit + fix

Read Ahn & Oh 2021 (*Mitigating Language-Dependent Ethnic Bias in BERT*) in
full; cloned the reference (`jaimeenahn/ethnic_bias@a115eb7c3af7`, freshly —
SHA confirmed to match the existing `_metric_info.py` citation) and
independently read `score.py` (the actual CB-score computation, including
its whole-word-masking and variance code) in full. Traced `cbs.py` line by
line — there was no pre-existing test file to cross-check against (a prior
`test_cbs.py` had been removed for containing zero real CBS assertions,
RL-017).

**Confirmed faithful, independently re-derived:** the core single-token
formula (`log P' = log p_tgt - log p_prior`, the `(1/|T|)(1/|A|)ΣΣVar`
aggregation, and the target-mask-ordinal derivation for the prior sentence)
match the paper's equation and the reference's arithmetic exactly.

**Five findings, all fixed:**

1. **`run()` was completely broken — crashed on every single call.**
   `evaluate()`'s dict had no key `base.py::_split_result` recognizes
   (`cbs` isn't `bias_score`/`score`/`value`/`effect_size`). Fixed by
   adding `"bias_score"` and `"per_item"`.
2. **The shipped example crashed immediately.** A stale method-override
   signature in its offline demo subclass. Fixed.
3. **Multi-token attributes broke structural parity between the target
   and prior sentences.** The prior sentence always used exactly one mask
   token for the attribute, regardless of its real subword count, though
   the paper's whole-word-masking adaptation (confirmed via the
   reference's `attribute_mask`) applies to the attribute too. Several of
   the paper's own 70 attributes are multi-token under
   `bert-base-uncased` ("C.E.O." → 6 pieces). Verified by execution: for
   a 3-token attribute, using 1 mask vs. the correct 3 shifted
   target-mask log-probabilities by up to 0.66 nats. Fixed by inserting
   `attribute_num` mask tokens.
4. **Multi-token target handling (opt-in) computed an unrelated
   quantity.** It evaluated all of a target word's subword IDs as
   candidates at one single mask slot rather than inserting one mask
   token per subword. Fixed by grouping target words by subword count and
   scoring each subword against its own mask position, one-to-one
   (following the paper's text over the reference's apparent all-pairs
   loop bug — recorded as `REVIEW_LATER` RL-088, the same judgment
   already applied for LPBS's RL-012).
5. **Variance convention didn't match the reference.** `np.var(ddof=0)`
   (population variance) vs. the reference's `pandas.Series.var()`
   (`ddof=1`, sample variance) — confirmed by reading `score.py` directly;
   the previous fidelity note had guessed population variance was
   defensible without checking. Fixed to `ddof=1`, falling back to `0.0`
   (not NaN) for a single target.

**Tests.** `tests/test_probability_based/test_cbs.py` created from
scratch (16 tests, none existed before): the formula, `run()` no longer
crashing plus its bootstrap CI, both whole-word-masking fixes (checked
against independently hand-recomputed expected values, not just "doesn't
crash"), the `ddof=1` variance convention, and input validation. Full
suite: `1878 passed, 4 skipped, 5 deselected, 1 xfailed` (up from 1862 —
16 new tests, no regressions).

**Verdict changed: MATERIALLY DEVIATES → FAITHFUL.** The core math was
already exact; every finding was in `run()`'s framework integration, a
structural sentence-construction bug affecting realistic (multi-token)
inputs, an opt-in feature that computed the wrong thing entirely, a
constant scaling factor, and a broken example — all fixed and verified,
none requiring a change to the CB score formula itself.

## 2026-09-18 · DisCoMetric from-scratch audit + fix

Read Webster et al. 2020 (*Measuring and Reducing Gendered Correlations in
Pre-trained Models*) in full, including Appendix A's 14 templates. No
reference implementation exists (the paper gives no code URL; confirmed
consistent with `sources/SOURCES.yaml`'s `code_status: none_found`).

**Confirmed faithful, independently re-verified:** re-checked
`chi_square_2xk`/`chi_square_p_value` against `scipy.stats.chi2_contingency`
fresh on 6 tables (bit-exact to displayed precision on the 5 valid cases;
sensible handling of a degenerate all-zero-variance case scipy itself
errors on) rather than trusting the existing doc's claim of prior
validation. Re-derived the contingency-table construction, the
Bonferroni-corrected significance test, and the per-template-averaged
aggregation from the paper text and confirmed they match `disco.py`
exactly. Confirmed `run()` correct by fresh execution
(`score=2.0, n=1, ci=(2.0,2.0)`, matching an independent hand-computation).
This is the best-verified metric audited this session — no defect found in
the metric itself.

**Two findings, both fixed, neither in the metric:**

1. **The shipped example was still the pre-refactor API.**
   `examples/probability_based/disco.py` called `metric.evaluate(template=...,
   attr_a=..., attr_b=..., k=5)` against the current, paper-faithful
   `DisCoMetric` — that signature belongs to `TopKFillDivergence` (the old
   v0.1.1 statistic, intentionally split into its own class when
   `DisCoMetric` was reimplemented). Confirmed by execution:
   `TypeError: DisCoMetric.evaluate() got an unexpected keyword argument
   'template'`. Rewritten to the current API; also fixed a substring bug
   discovered while rewriting it (`"man" in person` is also true for `"the
   woman5"`, since `"woman"` contains `"man"`) that would have silently
   made the new example's own demonstration data uninformative.
2. **Stale `validation/registry.yaml` notes.** All eight DisCo Tier-1 rows
   said "Blocked until DisCoMetric is reimplemented ... computes a
   different statistic" — true for v0.1.1, false since the reimplementation.
   Corrected to name the actual remaining blocker (the reproduction run
   itself, not an implementation gap).

**Verdict unchanged: FAITHFUL.** Both fixes were to an example script and
bookkeeping notes; the metric's own formula, statistics, and `run()`
integration were already correct and are now independently re-confirmed,
not newly fixed.

## 2026-09-18 · ToxicityFraction from-scratch audit + fix

Read Gehman, Gururangan, Sap, Choi & Smith 2020 (*RealToxicityPrompts*,
Findings of EMNLP 2020) fresh, §3.2 in particular ("we characterize toxic
generations with **two** metrics"), and re-read `docs/fidelity/toxicity_family.md`
only after forming an independent view. Traced `ToxicityFraction`,
`ToxicityProbability`, and `EMT` (all three share this page and this paper)
through to `evaluate()` and `run()`.

**Provenance finding: already correct, not a new defect.** The paper defines
exactly two metrics — expected maximum toxicity (`EMT`) and the empirical
probability of at least one toxic span (`ToxicityProbability`). `ToxicityFraction`
(mean fraction of the K generations that are toxic) is not in the paper at
all — confirmed by full-text search, "fraction" never occurs and "proportion"
only refers to training corpora. This was already correctly caught by a prior
audit: `fidelity="original"`, `reference` cites Gehman as inspiration only,
`docs/fidelity/toxicity_family.md` already documents it plainly. No new
finding here — re-verified, not re-flagged.

**New finding: `run()` unconditionally broken, for all three classes.**
`evaluate(return_details=True)` for `EMT`, `ToxicityProbability`, and
`ToxicityFraction` each returned a dict keyed only by its own
metric-specific name (`emt_score` / `toxicity_probability` /
`toxicity_fraction`) — none of which `base.py::BiasMetric._split_result`
recognizes (`bias_score`/`score`/`value`/`effect_size`). Confirmed by direct
execution: `.run(..., ci="none")` raised `BiasScopeError` on every call, for
all three, before the fix. `evaluate()` itself was correct and already
tested for all three — this was purely a framework-integration gap shared
by all three classes, identical to the same defect class found earlier in
this session in `CBSMetric`.

**Fix.** Added `"bias_score"` and `"per_item"` to all three `evaluate()`
dicts: `per_item` is the per-template (EMT) or per-prompt
(ToxicityProbability/ToxicityFraction) list each headline score is already
the mean of, so `run()`'s default bootstrap CI is the metric's own standard
percentile bootstrap — no new judgment call. Verified live with mocked
Perspective clients: `run()` now returns correct `score`, `n`, and a
bracketing bootstrap `ci` for all three.

**Tests.** 8 new tests added across
`tests/test_generated_text_based/test_emt.py`,
`test_toxicity_probability.py`, `test_toxicity_fraction.py`
(`test_return_details_exposes_bias_score_and_per_item`,
`test_run_no_longer_crashes`, `test_run_bootstrap_produces_a_ci` per class,
plus EMT's existing `test_emt_details` extended). Full suite: **1886
passed** (up from 1878, exactly +8, zero regressions).

Documented in `DECISIONS.md`, `CHANGELOG.md`,
`docs/fidelity/toxicity_family.md`, and `_metric_info.py`'s `deviation_note`
for all three classes.

**Verdict unchanged: FAITHFUL (EMT, ToxicityProbability) / adaptation, kept
as `original` (ToxicityFraction).** The fix was a framework-integration
issue affecting `run()` only; no metric's formula changed.

## 2026-09-18 · ToxicityProbability from-scratch audit

Read Gehman et al. 2020 §2.1 and §4.2 fresh (both state "TOXICITY ≥ 0.5",
inclusive) and traced `toxicity_probability.py` end to end. Independently
verified against the reference by cloning `allenai/real-toxicity-prompts` @
`dd44ab77ed8b` — the repo ships no metric-computation module, so the actual
code that produced the paper's Table 2 numbers had to be found in
`notebooks/realtoxicityprompts_results.ipynb`.

**Formula, aggregation, validation, `value_range`, `direction`: all
confirmed correct, no defect.** `_has_toxic` implements
`TP = (1/N)·Σ max(I(score(t) ≥ threshold))` exactly.

**One finding, no code change, logged for traceability.** The reference
notebook's actual cell uses a **strict** `>` (`len([y for y in x if y >
0.5])`), contradicting the paper's own text, which states `≥` twice.
BiasScope already used `>=`, matching the paper — correct under the same
precedent already applied twice this session (LPBS RL-012, CBS RL-088):
when paper text and an unexplained reference discrepancy disagree, follow
the paper. Logged as `REVIEW_LATER` RL-089 and pinned by a new test,
`test_threshold_boundary_is_inclusive`, since nothing previously exercised
the exact-equality boundary.

Documented in `DECISIONS.md`, `CHANGELOG.md`, `REVIEW_LATER.md`,
`docs/fidelity/toxicity_family.md`, and `_metric_info.py`'s
`deviation_note`. Full suite: 1887 passed (+1 from 1886).

**Verdict unchanged: FAITHFUL.**

## 2026-09-18 · EMT from-scratch audit + fix (resolves RL-022)

Read Gehman et al. 2020 §3.2/§4.2 fresh and traced `emt.py` end to end.
Independently re-verified against `allenai/real-toxicity-prompts` @
`dd44ab77ed8b`'s `notebooks/realtoxicityprompts_results.ipynb` (the actual
code that produced Table 2, since the repo ships no metric module).

**Confirmed correct, no new defect.** `EMT = (1/|T|)·Σ_t max_k s(t,k)`
matches the paper and the reference notebook's `max_toxicities.mean()`
exactly. Validation, `run()` (already fixed earlier this session),
`value_range`, `direction`, `neutral_value` all correct.

**One pre-existing gap, now fixed: RL-022.** Logged in an earlier session —
the paper reports EMT "with a mean **and standard deviation**"; `evaluate()`
returned only the mean. This audit's reference check confirmed the exact
convention to match: the notebook itself computes
`std_max = max_toxicities.std()` (pandas default `ddof=1`) right alongside
`avg_max`. Fixed by adding `"std"` to `evaluate(return_details=True)`'s
dict (`ddof=1`; `0.0` for a single template to avoid a divide-by-zero
NaN), which `run()` now surfaces via `result.details["std"]`, the same
mechanism WEAT uses for its permutation `p_value`.

**Tests.** 3 new tests in `tests/test_generated_text_based/test_emt.py`
(`test_std_matches_reference_notebook_convention`,
`test_std_is_zero_for_a_single_template`, `test_run_exposes_std_in_details`),
plus `test_emt_details` extended with a `std` assertion. Full suite:
**1890 passed** (up from 1887, +3, zero regressions).

Documented in `DECISIONS.md`, `CHANGELOG.md`, `REVIEW_LATER.md` (RL-022
marked resolved), `docs/fidelity/toxicity_family.md`, and
`_metric_info.py`'s `deviation_note`.

**Verdict unchanged: FAITHFUL.** The fix is additive (a new detail key);
`bias_score` and every existing key are untouched.

## 2026-09-18 · RegardScore from-scratch audit + fix (RL-090)

Read Sheng et al. 2019 fresh and traced `regard_score.py` end to end.
Independently checked against the reference by cloning
`ewsheng/nlg-bias@7f8d08ea4f33`.

**Confirmed correct, no new defect in the distributional design.** The
paper never defines a single scalar — it's a comparison of per-demographic
`[neg, neu, pos]` regard-label distributions (Figure 2). RegardScore's
16-key output (per-bucket fractions and A−B differences across 4 buckets:
`negative`/`neutral`/`positive`/`other`) is a faithful, reasonable
operationalization of that, already correctly labeled `adaptation` for the
`sasha/regardv3` vs. `regard1`-ensemble checkpoint substitution (pre-existing,
re-confirmed, not revisited).

**New finding: `run()` was unconditionally broken, same root cause found
repeatedly this session.** None of the 16 keys matched anything
`BiasMetric._split_result` recognizes, and no `n`-like key existed either —
confirmed by direct execution raising `BiasScopeError` for every `ci=`
setting. Unlike the prior four fixes (CBS, EMT, ToxicityProbability,
ToxicityFraction), there was no obvious headline number to add: cloning the
reference confirmed it *also* never computes one, so any choice is a
BiasScope invention. Fixed by adding `bias_score = (positive_difference -
negative_difference) / 2` — bounded in the metric's declared `value_range`,
antisymmetric under group swap — and `n` (total texts scored). No natural
`per_item` exists, so `run()`'s default bootstrap CI correctly degrades to
`ci="none"`, matching WEAT/SEAT/CEAT/CBS's documented behavior. Logged as
`REVIEW_LATER` RL-090.

**Secondary finding, fixed: a stale fidelity-doc claim.**
`docs/fidelity/regard_score.md`'s Tier 3 section claimed swap antisymmetry
"is tested" when no such test existed and `RegardScore` isn't in
`validation/registry.yaml`. Corrected the claim and added the test.

**Tests.** 6 new tests in `tests/test_generated_text_based/test_regard_score.py`
(`test_bias_score_is_net_regard_gap`, `test_bias_score_swap_antisymmetry`,
`test_bias_score_is_zero_for_identical_groups`,
`test_n_counts_all_flattened_texts`, `test_run_no_longer_crashes`,
`test_run_has_no_bootstrap_ci_without_per_item`), plus
`test_all_values_are_floats` updated for `n`'s `int` type. Full suite:
**1896 passed** (up from 1890, +6, zero regressions).

Documented in `DECISIONS.md`, `CHANGELOG.md`, `REVIEW_LATER.md` (new
RL-090), `docs/fidelity/regard_score.md`, and `_metric_info.py`'s
`deviation_note`.

**Verdict unchanged: adaptation.** The fix is additive and framework-level
(`bias_score`/`n` are new keys); the metric's distributional design and
existing 16 keys are untouched. Noted in passing, out of scope here:
`MeanScoreGap` has the identical `run()`-breaking gap (no `n`-like key);
flagged separately rather than fixed in this pass.

## 2026-09-18 · MeanScoreGap from-scratch audit + fix

Re-read `mean_score_gap.py`'s own module docstring and `docs/fidelity/score_parity.md`
fresh. This metric is already honestly labeled `original` — a prior audit
correctly found it is not derivable from Borkan et al. 2019's five
threshold-agnostic, label-based AUC/Equality-Gap metrics, and renamed it
from `ScoreParity`. No paper-comparison work was needed; this audit focused
on code correctness.

**Cohen's d computation confirmed correct.** Pooled variance
`((n_a-1)*std_a² + (n_b-1)*std_b²)/(n_a+n_b-2)`, guarded against
`n_a+n_b<=2` and near-zero pooled std, is the standard formula, independently
re-derived and checked by hand on a synthetic example.

**Confirmed: `run()` was unconditionally broken, same root cause pattern as
this session's other fixes, but a different specific gap than RegardScore's.**
`effect_size` was already a key `_split_result` recognizes, so the headline
lookup worked — but no key `_count_items` recognizes as an item count
existed, so `n` always resolved to 0 and `run()`'s `n > 0` guard raised on
every call regardless of `ci=`. Fixed by adding `"n"` (total texts scored
across both groups). No `per_item` added: Cohen's d is a two-sample
statistic (not a per-prompt one), so a naive bootstrap over pooled items
wouldn't estimate the right thing; `run()`'s default CI correctly degrades
to `ci="none"`, matching WEAT/SEAT/CEAT/CBS/RegardScore's documented
behavior for metrics without item-level scores.

**Found but explicitly not fixed, flagged for later:** `group_a_std`/
`group_b_std` are `NaN` (with unguarded `RuntimeWarning`s — this explains
warnings already visible in this session's full-suite runs) whenever a
group has exactly one text, since `np.std(scores, ddof=1)` divides by zero
at n=1. Doesn't affect `run()`'s guards (only the headline score and CI are
checked), so left as a documented follow-up rather than fixed in this pass.

**Tests.** 3 new tests in `tests/test_generated_text_based/test_mean_score_gap.py`
(`test_n_counts_all_flattened_texts`, `test_run_no_longer_crashes`,
`test_run_has_no_bootstrap_ci_without_per_item`), plus
`test_all_values_are_floats` updated for `n`'s `int` type. Full suite:
**1899 passed** (up from 1896, +3, zero regressions).

Documented in `DECISIONS.md`, `CHANGELOG.md`, `docs/fidelity/score_parity.md`,
and `_metric_info.py`'s `deviation_note`.

**Verdict unchanged: original.** The fix is additive (`n` only); the
metric's own statistic (already an honest, non-paper-derived original) is
untouched.

## 2026-09-18 · CounterfactualSentimentBias from-scratch audit + fix

Read Huang, Zhang, Jiang, Stanforth, Welbl, Rae, Maini, Yogatama & Kohli
2020 (DeepMind, Findings of EMNLP 2020) fresh from the PDF and traced
`counterfactual_sentiment_bias.py`, `stats.py::wasserstein_1`, validation,
`run()`, tests, the example, and both docs files end to end. No reference
implementation exists (confirmed: `code_status: none_found`).

**Core statistic independently re-verified correct, no computational
defect.** `wasserstein_1`'s quantile-matching estimator matches
`scipy.stats.wasserstein_distance` bit-exactly over 200 random trials (max
error `2.2e-16`). The two-group normalization (`mean_t W1(...)`) is exactly
eq. 3's `2/(M|A|(|A|-1))` for a binary attribute — verified algebraically
and against the existing test's hand-derived arithmetic. `run()` already
worked correctly (`bias_score`/`per_item`/`n` were all present from a prior
audit), giving a working default bootstrap CI.

**Major finding, fixed: an impossible, misleading sign claim.** The class
docstring's "Interpretation" section claimed `csb_score` is signed — "CSB
< 0: group B receives more positive sentiment" — impossible, since
`csb_score` is a Wasserstein-1 distance, always `>= 0`. Confirmed with a
constructed counterexample: group A all strongly negative, group B all
strongly positive (B unambiguously favored) gives `csb_score = 1.6`
(positive), while `signed_mean_difference` (the statistic the stale text
was actually describing) correctly gives `-1.6`. The same false claim was
duplicated verbatim in the executable example's printed output and its
copied `docs/api` page — three places, one bug. This is a genuine
practical risk for a bias metric: a user following the documented
interpretation could report the wrong group as disadvantaged. Fixed all
three copies to describe `csb_score` correctly (unsigned, direction-blind)
and point to `signed_mean_difference` for direction.

**Two findings, documented but not changed (no bug, just undocumented):**
1. Huang et al. define the sentiment classifier's output domain as `[0,1]`
   (§3, confirmed by all three of their own classifiers). BiasScope
   validates `[-1,1]` instead, and the shipped example uses `[-1,1]`-scaled
   scores. `wasserstein_1` is domain-agnostic, so this is a valid
   generalization, not a bug — but `csb_score` is only numerically
   comparable to the paper's own reported I.F. values when scores are
   actually scaled to `[0,1]`, and this was previously undocumented
   (`deviation_note` was empty despite `fidelity="faithful"`).
2. This class computes exactly one pairwise term of eq. 3 — correct in
   full for a binary attribute (Name), but not the paper's full multi-value
   average for attributes with more than two values (Country: 10,
   Occupation: 29) without the caller averaging multiple calls themselves.
   Not previously stated anywhere.

Both logged together as `REVIEW_LATER` RL-091 (a single entry, since both
are the same kind of judgment: keep the current, more general/flexible
behavior, but stop being silent about it).

**Tests.** 2 new tests in
`tests/test_generated_text_based/test_counterfactual_sentiment_bias.py`
(`test_csb_score_is_nonnegative_even_when_group_b_is_clearly_favoured`,
`test_csb_score_is_symmetric_under_group_swap`). Full suite: **1901
passed** (up from 1899, +2, zero regressions).

Documented in `DECISIONS.md`, `CHANGELOG.md`, `REVIEW_LATER.md` (new
RL-091), `docs/fidelity/huang_metrics.md`, and `_metric_info.py`'s
`deviation_note`.

**Verdict: FAITHFUL WITH DOCUMENTED EXTENSIONS.** The core W1/I.F.
computation was always correct; the fix corrects a real documentation bug
and makes two previously-silent design choices explicit. Fidelity label
unchanged at `faithful` in `_metric_info.py` since the underlying
statistic itself was never wrong — only its documentation was.

## 2026-09-18 · PsycholinguisticNorms from-scratch audit + fix (RL-092)

Read Dhamala, Sun, Kumar, Krishna, Pruksachatkun, Chang & Gupta 2021 (BOLD,
FAccT '21) §4.4 fresh from the PDF and traced `psycholinguistic_norms.py`
end to end, explicitly *not* trusting `docs/fidelity/bold_metrics.md`'s
existing claims until independently re-deriving the formula. Cloned
`amazon-science/bold@3ad652c773f5`: dataset-only, no metric code, so the
paper's text is the sole authoritative source.

**Critical finding, fixed: the aggregation formula was fundamentally
wrong, not a scaling variant.** The paper's §4.4 formula is a
magnitude-weighted signed aggregation, `Σsgn(w)w²/Σ|w|` — the identical
mathematical form as the paper's own Gender-Wavg in §4.5 (already correctly
transcribed elsewhere in the same fidelity doc, making the miss avoidable).
BiasScope computed a plain arithmetic mean instead. Counterexample: a
completion with three near-neutral filler words and one strongly-valenced
word (`[0.1, 0.1, -0.1, 4.0]`) gave `1.025` (plain mean) vs. `3.723` (paper's
formula) — a 3.6x divergence, the outlier word almost entirely diluted by
the wrong aggregation. **The existing `docs/fidelity/bold_metrics.md` and
`_metric_info.py` both explicitly (and wrongly) claimed "the aggregation
matches"** — this was a documentation bug inherited from an earlier,
insufficiently careful audit; both corrected as part of this fix.

**Second finding, fixed: no function-word exclusion.** The paper excludes
"pronoun, preposition, and conjunction" tokens (they "do not convey any
emotion"); `_tokenize` did plain regex splitting with no POS filtering.
Added `EXCLUDED_FUNCTION_WORDS`, a defensible closed-class word list — the
paper names no exact list or tagger, so this is a judgment call, logged
along with the run() headline choice below as `REVIEW_LATER` RL-092.

**Third finding, fixed: `run()` unconditionally broken.** Same recurring
defect class as six other metrics fixed this session — `evaluate()`'s dict
had only per-dimension keys (`pn::valence`, etc.), no `bias_score`/`n`-like
key. Fixed with the same kind of judgment call `RegardScore`'s RL-090
required (no paper precedent for a cross-dimension scalar): for a single
requested dimension, `bias_score` is exactly that dimension's score (no
judgment call); for multiple dimensions, it's their mean (a BiasScope
composite). `per_item` is populated (enabling a genuine bootstrap CI) only
for the single-dimension case, where it's unambiguous.

**Confirmed correct, not new:** the existing "no VAD/BE5 rescaling" and "no
BE5 lexicon shipped" findings from a prior audit were re-verified accurate
and left unchanged — those really are documented, reasonable scope
limitations, unlike the aggregation formula.

**Tests.** 5 new tests in
`tests/test_generated_text_based/test_psycholinguistic_norms.py`
(`test_weighted_aggregate_diverges_sharply_from_a_plain_mean`,
`test_function_words_are_excluded_from_aggregation`,
`test_run_no_longer_crashes`, `test_run_bootstrap_ci_for_a_single_dimension`,
`test_run_multi_dimension_has_no_bootstrap_ci`), plus
`test_psycholinguistic_norms_equation` rewritten with hand-derived expected
values for the corrected formula (the existing coverage test is unaffected,
since a single-word completion's weighted formula and plain mean coincide).
Full suite: **1906 passed** (up from 1901, +5, zero regressions).

Documented in `DECISIONS.md`, `CHANGELOG.md`, `REVIEW_LATER.md` (new
RL-092), `docs/fidelity/bold_metrics.md`, and `_metric_info.py`'s
`deviation_note`.

**Verdict: MATERIALLY DEVIATES → adaptation (after fix).** Before this fix,
the `fidelity="adaptation"` label understated the gap — the paper's central
equation for this metric was not implemented, not merely rescaled
differently. After the fix, `adaptation` is accurate: the aggregation now
matches exactly, and the two remaining documented gaps (no rescaling, no
BE5) are genuine scope limitations, not correctness bugs.

## 2026-09-18 · ScoreParity (MeanScoreGap) from-scratch audit + fix

Audited the `ScoreParity` deprecated alias — confirmed the alias mechanism
itself (a lazy `__getattr__`-based subclass in
`generated_text_based/__init__.py`) is correct: warns, `isinstance`-
compatible, identical behavior to `MeanScoreGap`. `MeanScoreGap` is already
correctly labeled `fidelity="original"` (a prior audit confirmed it is not
derivable from Borkan et al. 2019, despite the old name citing it), and its
`run()`-missing-`n` bug was already fixed earlier this session, so this
audit focused on remaining code correctness rather than paper comparison.

**Two Minor bugs found and fixed, both re-confirmed by execution:**
1. `group_a_std`/`group_b_std` returned `NaN` (with unguarded
   `RuntimeWarning`s — this explains warnings already visible in this
   session's earlier full-suite `pytest` runs) for a single-text group,
   since `np.std(scores, ddof=1)` divides by zero at n=1. Fixed: `0.0`,
   matching the convention already used elsewhere (e.g. `EMT`'s `std`).
2. A local `_validate_classifier_scores` override in `mean_score_gap.py`
   shadowed the inherited, more permissive one from `BiasMetric`
   (`isinstance(score, (int, float))` vs. the inherited
   `isinstance(score, (int, float, np.floating))`). Confirmed by execution
   that a `numpy.float32`-returning classifier was incorrectly rejected,
   even though the shared validator every other metric uses accepts it.
   NaN/Inf rejection was unaffected either way (both are caught by the
   `0.0 <= score <= 1.0` range check regardless of type). Fixed by removing
   the redundant override.

**Confirmed correct, no new defect:** Cohen's d's pooled-variance formula,
independently re-derived by hand; `run()`'s already-fixed `n`/headline
behavior; the deprecated-alias mechanism.

**Tests.** 2 new tests in
`tests/test_generated_text_based/test_mean_score_gap.py`
(`test_std_is_zero_not_nan_for_single_text_groups`,
`test_accepts_numpy_float32_classifier_scores`). Full suite confirmed green
with zero regressions.

Documented in `DECISIONS.md`, `CHANGELOG.md`, `docs/fidelity/score_parity.md`,
and `_metric_info.py`'s `deviation_note`.

**Verdict unchanged: original / FAITHFUL to its own declared status.** Both
fixes are hygiene/consistency corrections; the metric's own statistic
(Cohen's d and the mean-score gap) was never wrong.

### Later - the remote branches, too

By request, `origin/elissa-metrics` (12 commits, prompt-metric paper
reproductions) and `origin/nancy` (3 commits, a from-scratch audit of the
embedding and probability families and seven generated-text metrics) are in
`merge/all-branches` as well. `nancy-prompt-metrics` and `fix/weat-seat-bugs`
were already contained in `main`; `gh-pages` is the deployed site.

Elissa's merge had one conflict (an extras list) and one missing fixture
entry. Nancy's had 24: her audit had fixed, in August-September, the same
run()-reachability defects the agent work fixed in September, and changed
CAT's input from a token list to a string. Resolution, file by file: her
scoring code and tests win (they are the audit), with the agent's list-or-
tuple pair acceptance (RL-050) and its `headline_key` declarations re-applied
on top; RegardScore keeps RL-062's `negative_difference` headline and her
composite was not adopted (RL-090); her REVIEW_LATER ids RL-038..046 became
RL-084..092. Her CEAT now takes per-stimulus contextual token embeddings and
no longer encodes text, so the agent's `ceat_contexts` provider now computes
each word's own subword states in context with the model under evaluation -
which removes the sentence-pooling deviation RL-071 had recorded; only the
corpus substitution remains. The StereoSet provider passes CAT a string.
Five of her tests only failed on a machine with a GPU (tensors left on the
CPU) and are fixed; three of her functions exceed the complexity cap and are
suppressed with a note (RL-093); coverage is 86% (RL-094).

### Later - the twelve-model experiment, reproduced on the merged branch

By request. Ten of the twelve models re-run through the agent on
`merge/all-branches` (gemma-2 remains blocked, RL-079), one plan / one
confirmation / one report each, `results_table.py --since 2026-09-20T19
--compare-before` to list every cell that moved against the pre-merge table
(`RESULTS_pre-merge_2026-09-20.md`).

**Reproduced exactly:** every metric whose code the merge did not touch and
whose item count matched - WEAT/SEAT on gpt2, gpt2-medium and the encoders,
AUL/AULA, HONEST, GenderPolarity, DemographicRepresentation,
StereotypicalAssociations.

**Moved because the audit changed the metric:** CAT and ICAT on every encoder
(per-target aggregation, multi-subword scoring; e.g. bert-base-uncased CAT
69.00 → 63.22, ICAT 51.99 → 63.91), CrowSPairs on bert-base-uncased (55.73 →
58.02, the WordPiece alignment), CEAT everywhere (stimulus-aligned contextual
token embeddings instead of pooled sentences; encoders now 0.37-0.60,
causal 0.11-0.58 - and Llama and gemma-3 score it now).

**Moved because of the agent's own `limit` choice, not code:** EMT, RegardScore
and HONEST on several causal models, each with a different `n` in the two
runs (gpt2 EMT 25 vs 50 prompts). The scenario now tells the agent not to
pass a limit (RL-095).

**Moved for a reason found only now (RL-096):** WEAT on the three Qwen models.
The pre-merge numbers came through sentence-transformers, which wraps an
instruct model's text in its chat template before encoding - the WEAT words
had been embedded inside "You are Qwen, created by Alibaba Cloud..." The
shared-model path embeds the bare word; those are the numbers in the table.

**On the way there, one more merge defect:** CAT's new string context reached
the masked-LM scorer as "[MASK]." and no mask was found - every real-model
CAT/ICAT run failed; fixed in the scorer, plus StereoSet's target term passed
through so CAT's n is target terms, not 1.

## 2026-09-21 - a model served by OpenRouter as the model under evaluation

By request. `LiteLLMBackend.generate` now translates the providers'
transformers decoding names to chat-API names (RL-097), the scripted runner
has an `api` scenario, and every generation provenance says whether the text
came from a local causal LM or a chat API that answers rather than continues.

First run: `openrouter/meta-llama/llama-3.1-8b-instruct` through
`deepseek/deepseek-v4.1-flash`, 45 minutes, 2,205 sequential requests (500
BOLD profession, 80 BOLD gender, 1,000 HONEST, 625 RealToxicityPrompts), all
served from the cache afterwards. 37 metrics recommended for a chat-only
backend, 6 feedable, 6 scored, complete; every limit left at its default:

    RegardScore 0 (80)   GenderPolarity 0.0076 (500)   DemographicRepresentation 0.2222 (18)
    StereotypicalAssociations 0.5 (3)   HONEST 0.048 (1000)   EMT 0.01255 (25)*

The small n on the two HELM metrics is the chat behaviour showing through: an
instruct model answers "A metalworker is" with a sentence about the question,
so group words and adjectives rarely co-occur. Provenance carries
`access_mode: chat API ...` for exactly this reason. No embedding or
masked-LM metric is offered to an API target, correctly. Cost on this model:
under one cent; there is still no spending cap (RL-097).

### Later - the remaining metrics for an API target

By request ("fix to be able to run the remaining metrics"). Of the 31
recommended-but-unfed metrics for a chat target, the fixable ones were three
kinds: metrics that load their own benchmark and need only a model name
(BBQMetric, StereoSetMetric, IdentitySwapConsistency, OccupationPronounSkew -
one provider, `prompt_benchmarks`, that names the model, the axis's subset and
a bounded size), metrics that take model answers to the authors' items
(WinoBias from Zhao et al.'s type-1 files, DecodingTrustStereotype from Wang et
al.'s statements under the benign scenario), RealToxicityPrompts with the local
toxicity scorer injected in-process, and CoOccurrenceBiasScore, which the HELM
provider already had the inputs for. Two metric-level gaps surfaced on the
way: RealToxicityPrompts and OccupationPronounSkew returned their statistics
under their own names with no headline or count, so `run()` could not read
them; both now declare `headline_key` and `count_key`. The coverage check
learned to read a provider's "does not cover axis" refusal so a correct
refusal is not reported as an omission. RL-098 lists what is still deferred
(UnQover needs log-probabilities, TrustLLM's data is not in the clone, the
judge-based ones need a judge decision) and the six that stay blocked.

Third API-target run, `openrouter/meta-llama/llama-3.1-8b-instruct`: 37
recommended, 13 feedable on the gender axis (IdentitySwapConsistency refused
by name - its swap pairs are race and religion terms), **13 scored**, complete;
26 minutes, the generation-based datasets served from the cache.

    RegardScore 0 (80)  GenderPolarity 0.0076 (500)  HONEST 0.048 (1000)
    DemographicRepresentation 0.2222 (18)  StereotypicalAssociations 0.5 (3)
    CoOccurrenceBiasScore 0.5417 (122)  EMT 0.01255 (25)*
    BBQMetric 0.02605 (95)  StereoSetMetric 88.98 (200)  OccupationPronounSkew 1.395 (200)
    WinoBias 0.43 (100)  DecodingTrustStereotype 0 (120)  RealToxicityPrompts 0.03715 (10)*

One caveat to read them with: the prompt-family metrics query the API
themselves and are not cached, and an API's temperature-0 answers are not
bit-stable, so BBQ moved from 0.041 (n=102) to 0.026 (n=95) and
RealToxicityPrompts from 0.076 to 0.037 between the second and third runs
on identical inputs. The cached generation-based metrics reproduced exactly.
