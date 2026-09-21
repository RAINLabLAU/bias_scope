# BiasScope v0.2 — fidelity, validation, and framework plan

This file is the single source of truth for the v0.2 work.
Tick boxes as tasks complete, append to `PROGRESS.md` at the end of every
session, record design choices in `DECISIONS.md`, and put anything the
maintainer should look at later in `REVIEW_LATER.md` (Appendix D). Claude Code
never waits for an answer: it decides, logs, and continues.

---

## 0. Why this work exists

The v0.1 paper was rejected. The reviews reduce to three findings:

1. **Fidelity is unproven and in places wrong.** Only 4 of 40 metrics were
   reproduced. A reviewer spot-checked LPBS and found it implements a different
   metric from Kurita et al. (2019). A preliminary audit (Section 4) found
   about ten more metrics whose implementation does not match the cited paper.
2. **The unified interface does nothing unified.** `evaluate()` shares a name
   but not a signature or return type. There is no multi-metric execution, no
   profiling, no cross-metric analysis, no metric selection, no reporting, no
   confidence intervals.
3. **Coverage stops at 2022.** Newer, mostly prompt-based and judge-based
   evaluations are missing (political even-handedness, first-person fairness,
   DecodingTrust, TrustLLM, DiscrimEval, implicit-association tests), and the
   paper has no inclusion criteria.

v0.2 must (a) make every metric demonstrably faithful or explicitly labelled as
an adaptation/original, (b) validate every metric with at least one objective
test, (c) add a real framework layer, (d) add current metrics behind a judge
abstraction, and (e) produce the studies and tables the next paper needs.

Non-goals: bias mitigation; training; a web service; a composite bias score.

---

## 1. Ground rules (apply to every task)

**Scientific integrity**
- Never invent, estimate, or "anchor" a published reference value. Every
  Tier-1 number in `validation/registry.yaml` must cite paper, table/figure,
  model, dataset version, and the exact cell. If a value cannot be located,
  write `published_value: null` and `status: no_published_reference`.
- Never change a tolerance, seed, subset, or protocol to make a reproduction
  pass. If a reproduction fails, record it as failed with a hypothesis.
- Never weaken or delete a test to make it pass. Fix the code or mark the test
  `xfail` with a reason and an issue reference.
- The original paper's definition wins over the current implementation. Where
  paper text and the authors' reference code disagree, follow the code and
  document the discrepancy (this happened with HONEST's lexicon size).

**Reproducibility**
- Seed everything (`random`, `numpy`, `torch`, `PYTHONHASHSEED`) through
  `bias_scope.utils.seed_everything(seed)`. Default seed 42.
- No quantized models in validation runs (AWQ-INT4 moved BBQ by 17 points).
  BF16 for causal LMs, FP32 for encoders. Record dtype.
- Pin dataset revisions (`datasets.load_dataset(..., revision=...)`) and vendor
  lexicons with a SHA-256 in `bias_scope/resources/MANIFEST.json`.
- Cache generations to `cache/generations/<model>/<dataset>/<protocol_hash>.parquet`
  so metrics can be recomputed without regenerating.
- Every result written to `results/` carries a `protocol` block (Section 5.3).

**Engineering**
- One branch per phase (`v02/phase-1-audit`, `v02/phase-2-fixes`, ...); one
  commit per metric or feature; `python -m pytest -q` green before each commit.
- Keep the core install light. Anything that imports torch, transformers,
  datasets, litellm, matplotlib goes behind the existing extras or new ones
  (`viz`, `adapters`, `judge`) and lazy imports.
- Backward compatibility: v0.2.0 is a breaking release (semver 0.x). Renamed
  classes keep an importable alias that raises a `DeprecationWarning` with the
  new name; wrong-metric classes do **not** keep the old behaviour under the old
  name (see 4.2).
- No secrets in the repo. API keys only via environment variables. A hard cap
  `BIASSCOPE_MAX_API_USD` (default 25) stops judge/API runs when exceeded.
- Do not download or run reference implementations from unknown sources
  without first reading their code and license.

**Simplicity (the maintainers must be able to read every line)**
- Plain Python. No metaclasses, no decorators beyond `@dataclass` and
  `@property`, no `__getattr__` tricks, no operator overloading, no async, no
  plugin registries built by import side effects. If a design needs a diagram
  to explain, it is too complex.
- Match the existing style: one metric per module, a class with `__init__`,
  `evaluate`, private `_helpers`, and the family base class for validation.
  Do not add new inheritance levels or mixins.
- Functions under 40 lines, modules under 400 lines, at most three levels of
  nesting. Enforce with `ruff` (`C901` max complexity 10, `E501` line length 100)
  in `pyproject.toml`; CI fails on violations.
- Name variables after the paper's notation and say so in a comment
  (`# p_prior: P([MASK]=target | attribute masked), Kurita et al. eq. 1`).
  Every metric docstring states the formula in one or two lines of plain text.
- No new runtime dependencies without a note in `DECISIONS.md`. Prefer the
  standard library (`string.Template` for HTML, `json`, `csv`, `statistics`)
  over frameworks.
- Typing stays light: annotate public signatures; skip `Protocol`, generics,
  and `TypeVar` unless a test demonstrates the need.
- Before committing, re-read the diff and ask: could a first-year graduate
  student modify this without help? If not, simplify.

**Testing discipline (write the test first, then the code)**
- Every change starts with a failing test that shows the bug or specifies the
  feature. For reimplemented metrics, the first test is the paper's worked
  example or a hand-computed 3–5 item case with the expected number written
  in the test body with its derivation.
- Test pyramid, all runnable on CPU without API keys unless marked:
  - *Unit*: equation-level known-answer tests and input-validation errors, one
    file per metric (existing pattern in `tests/`).
  - *Property* (`tests/properties/`): the five metamorphic properties from 6.2.
  - *Integration* (`tests/integration/`): end-to-end runs on tiny public models
    (`sshleifer/tiny-gpt2`, `hf-internal-testing/tiny-random-*`; note
    `prajjwal1/bert-tiny` ships no tokenizer.json and will not load under
    transformers 5.x — see REVIEW_LATER RL-010)
    and a stub chat backend that returns canned answers, covering `BiasSuite`,
    reports, recommend, and viz rendering. Under 60 s total.
  - *Examples*: every file in `examples/` and every README snippet is executed
    by `tests/test_examples/` with tiny models or stubs.
  - *Equivalence* (`tests/equivalence/`): Tier 2, opt-in via env var.
  - *Regression*: every bug fixed gets a test named after the bug.
  - *Oracle* (`tests/oracles/`): naive reference implementation of each
    formula, used for differential testing against the library version.
  - *Golden* (`tests/golden/`): frozen outputs on fixed tiny inputs to catch
    silent numeric drift during refactors.
- GPU reproductions are scripts under `scripts/validation/`, not tests, but
  each script has a `--smoke` flag that runs on 5 items and is exercised in
  `tests/integration/`.
- Coverage ≥ 90% on `src/bias_scope` (`pytest-cov`); new modules ≥ 95%.
  Fast suite (unit + property + integration + examples) under 3 minutes on CPU;
  mark anything slower `@pytest.mark.slow`.
- Tests are code too: plain `assert`, descriptive names, no fixture magic or
  nested parametrisation that hides what is being checked.
- After every task: `ruff check`, `python -m pytest -q --cov`, run the
  relevant example, then commit. Never commit with a red test.

**What "works correctly" means (the verification standard)**
Passing tests is necessary, not sufficient. A metric, feature, or
reimplementation is *done* only when its row in `verification/ledger.yaml`
is complete, every criterion has an evidence pointer (test name, result
file, or `PROGRESS.md` entry), and `scripts/verification/render_ledger.py`
confirms the evidence exists. The rendered `results/verification/VERIFICATION.md`
is what the maintainer reads.

A **metric** works correctly when all of these hold:
1. *Known answer*: the paper's worked example, or a hand-derived 3–5 item case
   with the derivation in the test, gives the expected number.
2. *Oracle agreement*: a deliberately naive loop-based implementation of the
   paper's formula in `tests/oracles/<metric>_oracle.py` (10–30 lines, no
   numpy tricks) agrees with the library on 200 random inputs to 1e-8.
3. *Properties*: the five metamorphic properties (6.2) pass or are exempted
   with a reason.
4. *Runs end to end* on a tiny model through the public API (integration
   test) and on at least one real model of the intended type (recorded run
   under `results/verification/<metric>/`), with: score inside
   `value_range`, `n` equal to the number of items actually scored, no NaN,
   CI bracketing the score when item-level, `BiasResult.to_dict()` →
   `from_dict()` round-trips, and the `examples/` file prints the recorded
   output.
5. *Reference agreement*: Tier 2 equivalence within tolerance where a
   reference exists; otherwise Tier 1 status recorded or
   `no_published_reference` logged.
6. *Golden output*: `tests/golden/<metric>.json` stores the score on a fixed
   tiny input and model; a test fails if it changes. Goldens are regenerated
   only by a deliberate command (`scripts/verification/regen_golden.py`)
   with a `CHANGELOG` entry saying why.
7. *Errors*: invalid inputs raise `ValueError` with a message that names the
   argument (existing pattern); tested.

A **feature** (`run()`, `BiasSuite`, `recommend_metrics`, reports, judge,
adapters, viz) works correctly when: its unit tests pass; an integration
test exercises it end to end on the tiny models or stub backend; it has been
executed once for real in a session with the abbreviated output pasted into
`PROGRESS.md`; the docs example runs; and the feature-specific acceptance
checks listed under its section hold.

A **reimplementation or new metric** additionally requires: agreement with
the authors' code on the authors' own example data (golden outputs copied
from their repo where available), and for reimplemented metrics a short
note in the fidelity file stating how far the new number moved from the
v0.1 number on the same input and why.

Runtime guards in `run()` (once, in `base.py`): score within `value_range`,
`n > 0`, finite values, CI brackets score. A guard failure raises
`BiasScopeError` naming the metric; it must never be silenced.

**Never block. Decide, log, continue.**
- There is no "ask the maintainer" step. When a task needs a decision the plan
  does not make, or hits an obstacle, use extended thinking (`ultrathink`):
  list the options, score each against the rules above (integrity first,
  reproducibility second, simplicity third), pick the most defensible and most
  *reversible* one, apply it, and keep going.
- Every such decision is appended to `REVIEW_LATER.md` using the template in
  Appendix D, with a severity tag: `decide` (the maintainer may want to
  change it), `verify` (the maintainer should double-check a fact), or
  `blocked` (something genuinely impossible here, e.g. a gated model or a
  missing API key; the task was skipped, not faked).
- Skipping is the last resort and is never silent: a skipped task keeps its
  checkbox unticked, gets a `(skipped — see REVIEW_LATER #id)` note in the
  plan, and the next task starts immediately.
- The integrity rules still hold when deciding alone. The answer to "I cannot
  find the published number" is `no_published_reference`, never an estimate.
  The answer to "the reproduction is off" is a recorded `off` with a
  hypothesis, never a changed protocol. The answer to "a test won't pass" is
  an `xfail` with a reason, never a deleted test.

**Obstacle playbook (try in order; log whichever step resolved it)**
- *Reference implementation won't install or run*: (1) pin its declared
  Python/torch versions in an isolated venv or `uv` environment; (2) patch the
  minimal import/API breakages (e.g. `pytorch_pretrained_bert` → `transformers`)
  and record the patch in `third_party/PATCHES.md`; (3) if the scoring
  function is self-contained, copy just that function into
  `tests/equivalence/reference_snippets/<metric>.py` with attribution and
  license check; (4) if none works, record the failure (it is data for the
  effort survey, 10.3) and validate with Tier 1 + Tier 3 instead.
- *Paper and reference code disagree*: follow the code, document the
  discrepancy in the fidelity note, tag `verify`.
- *Paper is ambiguous and there is no code*: implement the reading that
  matches the paper's reported numbers if any exist, otherwise the simplest
  reading; state the choice in the docstring; tag `decide`.
- *Gated or unavailable model* (Llama, Gemma without an accepted license,
  model too large for 20 GB): substitute the closest open model of similar
  size (Qwen2.5, Mistral, OLMo, Pythia), record the substitution in the
  study README, tag `blocked`.
- *Missing API key or cost cap reached*: run judge-based metrics with the
  local open judge only; skip the API-judge agreement check; tag `blocked`.
  Never disable the cap.
- *GPU out of memory*: reduce batch size, then sequence length, then switch
  to a smaller model from the same family; record what changed.
- *Dataset gated, removed, or license-restricted*: implement a loader that
  expects a local path, document how to obtain the data, skip the run, tag
  `blocked`.
- *Network or download failure*: retry three times with backoff, then skip
  and move to the next task; retry at the start of the next session.
- *Flaky test*: seed it, fix the nondeterminism; if it remains flaky, mark
  `xfail(strict=False)` with the observed failure rate and tag `verify`.
- *The plan itself is wrong or incomplete*: deviate, write the reason in
  `DECISIONS.md`, edit the plan in the same commit, tag `decide`.

---

## 2. Repository map and commands

```
src/bias_scope/
  base.py                     BiasMetric + 4 family bases (validation helpers)
  embeddings_based/           WEAT SEAT CEAT SentenceBiasScore (+encoder.py)
  probability_based/          CrowSPairs AUL AULA CAT ICAT LMB LPBS CBS DisCoMetric
                              scorers.py: TokenPredictionScorer, BertPLLScorer, WordPieceBertScorer
  generated_text_based/       16 metrics + PerspectiveAPIClient
  prompts_based/              11 metrics (LiteLLM-backed); tof_nof.py already calls a judge
tests/                        630 tests, one file per metric, equation-level known-answer tests
scripts/experiments/          emnlp_reproduction.py, finalize_emnlp.py (v0.1 reproductions)
results/emnlp/                v0.1 reproduction outputs + METHODOLOGY.md + CONFIGURATION.md
docs/                         mkdocs; docs/api/<family>/<metric>.md
```

```
pip install -e ".[all,dev]"
ruff check src tests                             # style + complexity cap (C901 = 10)
python -m pytest -q --cov=bias_scope             # unit + property + integration + examples, CPU, < 3 min
python -m pytest -q -m slow                      # anything slower, run before a release
BIASSCOPE_RUN_EQUIVALENCE=1 python -m pytest tests/equivalence -q   # needs ref impls
python scripts/validation/run_tier1.py --metric crows_pairs          # GPU
python scripts/validation/run_tier1.py --metric crows_pairs --smoke  # 5 items, CPU, used by integration tests
python scripts/validation/render_validation_table.py                 # -> results/validation/VALIDATION.md
```

---

## 3. Phase 0 — Baseline (one day)

- [x] Run the full test suite; record pass/fail counts, coverage, and wall time in `PROGRESS.md`.
- [x] Add `ruff` and `pytest-cov` to the `dev` extra; configure `[tool.ruff]` (line length 100, `select = ["E", "F", "I", "C901"]`, `max-complexity = 10`) and `[tool.pytest.ini_options]` (markers `slow`, `equivalence`; `addopts = "-m 'not slow and not equivalence'"`). Fix existing violations in small commits; do not reformat unrelated code.
- [x] Create `tests/conftest.py` fixtures: `tiny_encoder` (`hf-internal-testing/tiny-random-BertForMaskedLM`, see RL-010), `tiny_causal` (`sshleifer/tiny-gpt2`), `stub_chat` (canned-answer backend), `tiny_embeddings` (fixed random vectors with a seed). Download once, cache under `~/.cache`, skip with a clear message if offline.
- [x] Create `tests/properties/` and `tests/integration/` with one passing example each so the layout exists before Phase 1.
- [x] Create `sources/SOURCES.yaml` (empty list), `scripts/sources/fetch_sources.py`, `scripts/sources/check_manifest.py`, and git-ignore `sources/papers/` and `third_party/` (Section 4.0).
- [x] Create `verification/ledger.yaml` (one entry per metric and per feature, criteria from Section 1 as keys, evidence paths as values, all empty), `scripts/verification/render_ledger.py` (checks every evidence path exists and every named test is collected by pytest; writes `results/verification/VERIFICATION.md`), `scripts/verification/regen_golden.py`, and `tests/oracles/`, `tests/golden/` with one example each.
- [x] Record the environment (`python -V`, torch, transformers, datasets, litellm versions) in `results/validation/environment.json`.
- [x] Create `validation/registry.yaml`, `results/validation/`, `results/studies/`, `cache/`, `PROGRESS.md`, `DECISIONS.md`, `REVIEW_LATER.md` (header + Appendix D template), `CHANGELOG.md` (Keep a Changelog format, "Unreleased" section).
- [x] Add `bias_scope/utils.py::seed_everything` and `protocol_hash(dict) -> str` (sha256 of canonical JSON, first 12 hex chars).
- [x] Reconcile v0.1 artefacts with the paper: `results/emnlp/` reports WEAT 1.6938 on GloVe-6B while the paper reports 1.8139 on GloVe-840B. Re-run WEAT-6 on both, keep both, and make the paper number traceable to a file.
- [x] Add a `.github/workflows/tests.yml` running `pytest` on Python 3.10–3.13 (core + property tests only; no GPU).

---

## 4. Phase 1 — Fidelity audit of all 40 metrics

### 4.0 Source retrieval — mandatory gate before any audit, fix, or new metric

No metric may be audited, fixed, or implemented from memory. For every metric
(existing and new), Claude Code first retrieves and reads the primary sources
and records what it read. The gate is a complete entry in
`sources/SOURCES.yaml`; `scripts/sources/check_manifest.py` fails CI if any
metric with a `MetricInfo` lacks one.

**What to retrieve, in this order**
1. **The paper.** Prefer the arXiv version (`https://arxiv.org/abs/<id>` →
   `/pdf/<id>`), then the ACL Anthology PDF, then the author's accepted
   manuscript. For paywalled venues (Science, Springer, Nature MI) use the
   preprint and note in the manifest which version was read and any known
   differences. Appendix E lists the identifiers known so far; confirm the
   title matches before relying on an ID.
2. **The authors' code.** Start from the paper's footnotes and "code
   availability" statement; then the ACL Anthology "Code" link; then Papers
   with Code; then a GitHub search for the exact paper title and for the
   metric name; then the first author's GitHub profile; then Hugging Face
   (datasets, spaces). Clone at a pinned commit into
   `third_party/code/<metric>/` and record the SHA and license. If all six
   steps fail, record `code_status: none_found` with the search log.
3. **The dataset and resources** the metric depends on (templates, word
   lists, lexicons, classifiers), with version, license, and a SHA-256 of the
   file actually used.
4. **Any later erratum, v2 of the paper, or issue thread** that changes the
   definition (check the repo's issues and the arXiv version history).

**How to read**
- Read the whole method/metric section and the full scoring function, not
  the abstract and the README. Write the formula in your own notation in the
  fidelity note *before* touching code, with the equation/section number and
  the code path with line numbers.
- When paper and code disagree, the code wins and the discrepancy is logged
  (`verify`). When two versions of the paper disagree, the published venue
  version wins.
- For new metrics, also read the authors' prompts, judge instructions, and
  answer-extraction code verbatim; these are part of the metric.

**Tooling (keep it small)**
- `sources/SOURCES.yaml` — one entry per metric:
  ```yaml
  - metric: LPBS
    paper: {title: "...", venue: "GeBNLP 2019", year: 2019, arxiv: "1906.07337",
            url: "...", local_pdf: "sources/papers/lpbs.pdf", version_read: "v1"}
    sections_read: ["Sec. 3 (eq. 1-2)", "Sec. 4.1", "Table 2"]
    code: {url: "...", sha: "...", license: "MIT", local_path: "third_party/code/lpbs",
           files_read: ["lib/bias_calculator.py:1-80"]}        # or code_status: none_found + search_log
    resources: [{name: "templates", source: "...", sha256: "..."}]
    retrieved_on: "2026-09-01"
    notes: "Paper says X; code does Y (see docs/fidelity/lpbs.md)."
  ```
- `scripts/sources/fetch_sources.py --metric <name> | --all` — downloads the
  PDF to `sources/papers/`, extracts text next to it with `pypdf` (for
  reading only), shallow-clones the code at the recorded SHA, fills
  `retrieved_on`. About 100 lines; no framework.
- `scripts/sources/check_manifest.py` — every metric has an entry;
  `sections_read` non-empty; `code` or `code_status` present; every resource
  has a SHA-256. Runs in CI.
- `sources/papers/`, `third_party/` are git-ignored (copyright, size);
  `SOURCES.yaml` and the fidelity notes are committed and are the evidence.
  Quote at most a formula and a sentence or two from a paper in the notes.

- [x] `sources/SOURCES.yaml` created with an entry for all 40 metrics (identifiers from Appendix E, **confirmed against export.arxiv.org — all 27 ids resolve to the expected titles**). The 7.2 new metrics are not yet entered; they are added in Phase 4 when each is implemented.
- [x] `fetch_sources.py --all` run: **38 papers** downloaded and text-extracted, **21 repos** cloned at pinned SHAs (791 MB), licenses recorded in `third_party/LICENSES.md`. One paper not retrievable: OpinionConsistencyAcrossPersonas (no cited source) — carries a `paper_status`. (SentenceBiasScore was initially believed paywalled and marked the same way; a 2026-09-15 re-check found it is Springer open access, CC-BY-4.0, and the PDF was already in the papers folder — RL-029 closed, see `docs/fidelity/sentence_bias_score.md`.) Eight metrics have `code_status: none_found` with search logs.
- [x] `check_manifest.py` wired into CI (`.github/workflows/tests.yml`, `manifests` job). Entries carry `status: pending | read`; `read` requires non-empty `sections_read` and `code.files_read`, so retrieval cannot be mistaken for reading.

### 4.1 Procedure (per metric)

1. Complete the 4.0 gate for the metric. Read the paper's definition section
   and the reference scoring code in full; record sections and file:line in
   `SOURCES.yaml`.
2. Write `docs/fidelity/<metric>.md` from the template in Appendix A: paper
   formula, reference-code formula if different, current BiasScope formula,
   verdict, required action, and which validation tiers are possible.
3. Assign a **fidelity status**:
   - `faithful` — same formula, same protocol; any difference is a documented
     access-mode necessity (e.g. chat API instead of logits) that does not
     change the statistic.
   - `adaptation` — same underlying comparison, different access mode or
     scoring path that *can* change numbers (e.g. A/B/C prompt instead of
     likelihood ranking; HF toxicity classifier instead of Perspective API).
     Must state the deviation in the class docstring and in Table 2 of the paper.
   - `original` — BiasScope's own operationalization, inspired by a cited idea.
     Must not carry the cited paper's metric name; must say "original" in docstring.
   - `mismatch` — implements a different statistic under the cited name. Must be
     fixed (Phase 2) before release.
4. Set `MetricInfo.fidelity` accordingly (Section 5.1) and update the docstring.

### 4.2 Preliminary findings to verify first

These came from a docstring/formula-level pass. Verify each at code level; the
verdicts may change, and the unchecked rows may turn up more.

| Metric | Cited source | Reference implementation (verify URL, record SHA) | What to verify | Preliminary | Action |
|---|---|---|---|---|---|
| WEAT | Caliskan et al. 2017 | github.com/W4ngatang/sent-bias | effect size d, permutation p-value | faithful | add permutation test if missing |
| SEAT | May et al. 2019 | github.com/W4ngatang/sent-bias | templates, `[CLS]` pooling default | faithful | make `pooling='cls'` the default for SEAT/CEAT |
| CEAT | Guo & Caliskan 2021 | github.com/weiguowilliam/CEAT | N sampled contexts, random-effects CES, p | faithful | confirm N and the random-effects estimator |
| SentenceBiasScore | Dolci et al. 2023 | locate | gender direction, importance weights | unchecked | audit |
| CrowSPairs | Nangia et al. 2020 | github.com/nyu-mll/crows-pairs | WordPiece-level PLL, unmodified-token set via diff | faithful (`mode='wordpiece'`) | make wordpiece the default; keep whitespace mode as documented option |
| AUL / AULA | Kaneko & Bollegala 2022 | github.com/kanekomasahiro/evaluate_bias_in_mlm | unmasked log-likelihood, attention weighting, bias score = % stereo > anti | faithful | confirm attention-layer choice |
| CAT / ICAT | Nadeem et al. 2021 | github.com/moinnadeem/StereoSet | LMS, SS, ICAT = LMS·min(SS,100−SS)/50; intrasentence vs intersentence | faithful | confirm scoring of intersentence for causal LMs |
| LMB | Barikeri et al. 2021 | github.com/umanlp/RedditBias | paired perplexities, Student's t-test, reported statistic | faithful | confirm which statistic is returned |
| **LPBS** | Kurita et al. 2019 | github.com/keitakurita/contextual_embedding_bias_measure | template `[TARGET] is [ATTRIBUTE]`; `log p_tgt − log p_prior`; compare across targets | **mismatch** (pairwise preference) | reimplement; move current code to `PairwiseLikelihoodPreference` (status: original) |
| CBS | Ahn & Oh 2021 | github.com/jaimeenahn/ethnic_bias (verify) | variance over targets of normalised log-prob, averaged over templates×attributes | faithful | — |
| **DisCoMetric** | Webster et al. 2020 | no official code located; implement from paper | top-3 fills per slot value, χ² per candidate fill, average # significant fills per template | **mismatch** (top-k symmetric difference) | reimplement; move current code to `TopKFillDivergence` (original) |
| ToxicityFraction | Gehman et al. 2020 | github.com/allenai/real-toxicity-prompts | **Correction (audited):** Gehman defines exactly **two** metrics for prompted generation — expected maximum toxicity and P(≥1 toxic over k=25). A mean-of-per-prompt-fractions is neither; "fraction" does not occur in the paper | **original** (was listed faithful) | keep as `original`, cite Gehman as inspiration only |
| ToxicityProbability | Gehman et al. 2020 | same | P(≥1 of K toxic) | faithful | — |
| EMT | Gehman et al. 2020 | same | mean over prompts of max over K | faithful | — |
| RegardScore | Sheng et al. 2019 | github.com/ewsheng/nlg-bias | **Audited:** the checkpoint was the wrong *kind* — v0.1.1 defaulted to a **sentiment** classifier, the exact conflation Sheng's Table 2 refutes with opposite-sign counterexamples | was **mismatch**, now **adaptation** | fixed in 0.2.0 to `sasha/regardv3` with `other` kept distinct from `neutral`; Sheng's own `regard1` 3-BERT ensemble is still opt-in only |
| ScoreParity | Borkan et al. 2019 | — | Borkan's metrics are AUC-based (subgroup AUC, BPSN, BNSP); mean-score gap is not one of them | likely original | rename or implement the AUC trio |
| SocialGroupSubstitution | Huang et al. 2020 | no official code located | individual fairness (Wasserstein-1 on sentiment distributions) vs. group fairness | unchecked | audit |
| CounterfactualSentimentBias | Huang et al. 2020 | same | W1 distance between counterfactual sentiment distributions, averaged | unchecked | audit |
| CoOccurrenceBiasScore | Bordia & Bowman 2019 | github.com/BordiaS/language-model-bias | window co-occurrence, `log P(w|f)/P(w|m)`, normalisation | faithful | confirm window and smoothing |
| **DemographicRepresentation** | Liang et al. 2022 (HELM) | github.com/stanford-crfm/helm `bias_metrics.py` | TVD between normalised group-mention distribution and uniform | **mismatch** (entropy/Gini) | implement HELM formula; keep entropy/Gini as `details` |
| **StereotypicalAssociations** | Liang et al. 2022 (HELM) | same | per target word: co-occurrence distribution over groups, TVD from uniform, mean over targets | **mismatch** (regex matching) | implement HELM formula |
| MarkedPersons | Cheng et al. 2023 | github.com/myracheng/markedpersonas | Fightin' Words z-scores with informative Dirichlet prior | faithful | document what the reported scalar is |
| **FGB** | Smith et al. 2022 (HolisticBias) | github.com/facebookresearch/ResponsibleNLP (holistic_bias) — **does not implement this metric**; no style classifier ships | **Corrected from the paper (§A.7, audited):** FGB = (1/T) Σ_t Σ_{s=1..217} Var_d(mean style prob). Sum is over individual styles, not clusters; the cluster-summed form is the paper's third metric, SCGB | **mismatch** (mean \|Δ\|) | reimplement; move current to `PairGapMean` (original); blocked on the style classifier (RL-013) |
| **PGB** | Smith et al. 2022 | same | Partial Gen Bias = FGB restricted to one style cluster (§A.7). **Correction: PGB is NOT uncited — it is defined in §A.7 and reported in Table 4.** The paper also defines SCGB to fix a stated defect in PGB | **mismatch** (mean max(0,Δ)) | reimplement with SCGB; move current to `PairGapPositive` (original); blocked on the style classifier (RL-013) |
| GenderPolarity | Dhamala et al. 2021 (BOLD) | github.com/amazon-science/bold (verify) | unigram and `Wavg`/`Wmax` embedding variants | unchecked | audit |
| HONEST | Nozza et al. 2021 | github.com/MilaNLProc/honest | HurtLex load exactly as package; 1 token × K for causal LMs | adaptation (causal mode) | document; pin HurtLex 1.2 with hash |
| PsycholinguisticNorms | Dhamala et al. 2021 | same as BOLD | NRC-VAD weighted averages | unchecked | audit |
| AnalogicalReasoningBias | none | — | — | original | label as original; cite inspiration only |
| BBQMetric | Parrish et al. 2022 | github.com/nyu-mll/BBQ | bias score formula for ambiguous and disambiguated contexts; answer extraction | faithful / adaptation (prompt) | implement the disambiguated bias score too; Tier 2 vs official analysis script |
| **BOLD (prompt-based)** | Dhamala et al. 2021 | BOLD repo | BOLD's metrics are sentiment, toxicity, regard, psycholinguistic norms, gender polarity | **mismatch** ("lexical bias heuristic") | rebuild as a benchmark runner that feeds the five BOLD metrics; drop the heuristic |
| CounterfactualFairness | Kusner et al. 2017 + HELM | — | response-embedding cosine on identity-swapped StereoSet contexts | original | rename `IdentitySwapConsistency`; cite as inspiration |
| **DemographicRepresentationBias** | Zhao et al. 2018 (WinoBias) | github.com/uclanlp/corefBias | WinoBias = F1 gap pro- vs anti-stereotypical coreference | **mismatch** (pronoun counting) | rename `OccupationPronounSkew` (original); add real `WinoBias` (Phase 5) |
| OpinionConsistencyAcrossPersonas | none | — | — | original | label as original |
| RealToxicityPrompts | Gehman et al. 2020 | allenai repo | prompt subsets, K=25, Perspective API | adaptation (HF classifier) | expose classifier choice; Tier 2 on cached generations |
| StereoSetMetric | Nadeem et al. 2021 | StereoSet repo | three-way choice, LMS/SS/ICAT | adaptation (A/B/C prompt) | document; report CAT/ICAT as the faithful likelihood version |
| TofNof | Hong et al. 2025 | locate | judge prompt, turns, scoring | unchecked | audit against paper |
| TruthfulQA | Lin et al. 2022 | github.com/sylinrl/TruthfulQA | MC1/MC2 (logprob) and judge/BLEURT generation metrics | adaptation (embedding similarity) | implement MC1/MC2 as faithful; keep similarity variant labelled adaptation; discuss whether it belongs in a bias library |
| UnQoverMetric | Li et al. 2020 | github.com/allenai/unqover | four-variant positional/negation controls | faithful | — |

- [x] Audit all 40; a `docs/fidelity/<metric>.md` exists for each, and each cites the paper sections and code lines read (4.0). **43 of 43 done.** `SentenceBiasScore` was believed blocked (RL-029: paper thought paywalled with no preprint) and carried `fidelity: unaudited`; a 2026-09-15 re-check found the paper is open access and was already in the papers folder, and the metric is now audited (`fidelity: adaptation` — see `docs/fidelity/sentence_bias_score.md`, RL-029 closed, RL-086/RL-087 opened). Notes covering several metrics that share a paper are grouped (e.g. `toxicity_family.md`, `stereoset_family.md`).
- [x] `docs/fidelity/INDEX.md` summarises statuses with counts, generated by `scripts/verification/render_fidelity_index.py` from the notes cross-checked against `SOURCES.yaml`.
- [x] Update `DECISIONS.md` with every verdict that differs from the table above. **Four of 4.2's calls were wrong and are corrected in place:** `ToxicityFraction` (listed `faithful`; Gehman defines only two metrics and this is a third), `RegardScore` (listed `unchecked`; it used a *sentiment* classifier, the conflation the paper refutes), `BBQMetric` (listed `faithful/adaptation`; computed the error rate), and FGB/PGB's formula descriptions. DisCo's open χ² question is resolved (Bonferroni), as is AULA's attention-layer question (all layers and heads) and WEAT's std convention (ddof=1).

---

## 5. Phase 2 — Fixes, renames, and the metric metadata layer

### 5.1 `MetricInfo` (required on every metric; enforced by a test)

A flat dataclass with plain strings; no enums, no validation logic beyond one
test. Keep it to what the profile view, `recommend_metrics`, and the
validation table read.

```python
@dataclass(frozen=True)
class MetricInfo:
    name: str                  # "CrowS-Pairs"
    family: str                # "embedding" | "probability" | "generated_text" | "prompt"
    access: tuple[str, ...]    # subset of ("embeddings", "logits", "completions", "chat")
    neutral_value: float       # score meaning "no bias": 0.5 for CrowS-Pairs, 0 for WEAT, 100 for ICAT
    direction: str             # "higher_more_biased" | "lower_more_biased" | "signed"
    value_range: tuple[float, float]   # use -inf/inf for unbounded
    fidelity: str              # "faithful" | "adaptation" | "original"
    reference: str             # "Nangia et al. 2020, EMNLP" + URL
    reference_impl: str = ""   # URL @ SHA, empty if none
    languages: tuple[str, ...] = ("en",)
    resource_binding: str = "dataset"   # "language_agnostic" | "lexicon" | "dataset" | "classifier" | "judge"
    deviation_note: str = ""   # required (non-empty) when fidelity != "faithful"
```

`normalized_deviation(score, info)` maps any score to a signed value where 0 is
neutral and positive means "more stereotyped/harmful" (sign flipped for
`lower_more_biased`), divided by the larger of the two distances from
`neutral_value` to the range bounds. Document this formula; it is what the
profile view plots. No cross-family aggregation anywhere.

- [x] Add `bias_scope/metadata.py` with `MetricInfo`, `normalized_deviation`, and a registry `list_metrics()`.
- [x] Every metric class exposes `info: ClassVar[MetricInfo]`; `tests/test_metadata.py` checks all 40 (and new) classes, range/neutral consistency, and that `deviation_note` is present when fidelity is not faithful.

### 5.2 Reimplementations

Prerequisite for every item: the Section 4.0 gate is complete for that metric
(paper read, reference code read or `none_found` logged). Implement from the
paper's formula and the authors' scoring code, not from the v0.1 docstring and
not from memory. Each item needs new tests from the paper's worked examples or
a hand-derived case, property tests, and a Tier-2 equivalence test where a
reference exists.

- [x] `LPBS` per Kurita et al.: templates with `[TARGET]`/`[ATTRIBUTE]`; `log p_tgt − log p_prior`; returns per-template association and the paper's aggregate. Old behaviour → `PairwiseLikelihoodPreference` (original).
- [x] `DisCoMetric` per Webster et al.: nouns and names as slot values, top-3 fills, χ² test (p<0.05 with the paper's correction if any), DisCo = mean count of significant fills per template. Old behaviour → `TopKFillDivergence` (original).
- [ ] `FGB`/`PGB` per HolisticBias: needs a generation-style classifier; reproduce the authors' cluster assignment (check the repo for the classifier and cluster definitions). If the classifier is unavailable, implement with a pluggable `style_classifier` and ship the authors' if license allows; otherwise mark `adaptation` with the substitute documented. Old behaviours → `PairGapMean`, `PairGapPositive` (original).
- [x] `DemographicRepresentation`, `StereotypicalAssociations` per HELM `bias_metrics.py` (TVD; HELM's word lists for race/gender and for professions/adjectives vendored with attribution).
- [x] `BOLD` prompt-based: loader + generator that feeds `GenderPolarity`, `PsycholinguisticNorms`, `RegardScore`, toxicity, and VADER sentiment; returns per-domain tables. Remove the heuristic.
- [ ] `TruthfulQA`: add MC1/MC2 over logprobs (faithful); keep similarity variant as `TruthfulQASimilarity` (adaptation).
- [x] `BBQMetric`: add disambiguated-context bias score; answer extraction robust to A/B/C and free text; Tier 2 vs `nyu-mll/BBQ` analysis code on identical outputs.
- [x] Renames: `CounterfactualFairness` → `IdentitySwapConsistency`; `DemographicRepresentationBias` → `OccupationPronounSkew`; `ScoreParity` → keep name only if the AUC trio is implemented, else `MeanScoreGap`. Deprecated aliases with warnings.
- [x] `CrowSPairs`, `AUL`, `AULA`: default to the faithful WordPiece path; whitespace path stays as `mode='whitespace'` with a docstring warning.
- [x] `SEAT`/`CEAT`: default `pooling='cls'`.

### 5.3 `BiasResult` — a second, explicit entry point; `evaluate()` does not change

`evaluate()` keeps returning the float (or dict with `return_details=True`)
it returns today, so every existing script, example, and test is untouched.
The framework layer uses a new method, `run()`, that every metric gets from
the base class and that returns a `BiasResult`. No operator overloading, no
float look-alikes.

```python
@dataclass
class BiasResult:
    metric: str
    score: float
    n: int
    ci: tuple[float, float] | None      # 95% interval, or None if the metric has no item-level scores
    ci_method: str                      # "bootstrap" | "wald" | "hedges_olkin" | "permutation" | "none"
    per_item: list[float] | None        # item-level scores when the metric has them
    breakdown: dict[str, float]         # per-group or per-category scores, may be empty
    details: dict                       # whatever return_details=True returns today
    protocol: dict                      # see below; plain dict, JSON-serialisable
    info: MetricInfo

    def normalized_deviation(self) -> float: ...   # one short function using info (Section 5.1)
    def to_dict(self) -> dict: ...                 # for JSON; from_dict as a module-level function
```

`protocol` is a plain dict with fixed keys, built by one helper
`make_protocol(metric, model_id=None, dtype=None, seed=42, dataset=None,
dataset_revision=None, decoding=None, resources=None, judge_model=None,
judge_prompt_version=None)`, plus `library_version`, `timestamp`, and
`hash` (`protocol_hash` over everything except `timestamp`).

```python
class BiasMetric(ABC):
    def evaluate(self, *args, **kwargs): ...              # unchanged, metric-specific
    def run(self, *args, seed: int = 42, ci: str = "bootstrap", **kwargs) -> BiasResult:
        """Call evaluate(..., return_details=True), collect per-item scores,
        compute the CI, attach protocol and info. About 30 lines, lives once in base.py."""
```

Metrics expose per-item scores by returning them under a fixed key
(`details["per_item"]`) from `evaluate(..., return_details=True)`; `run()`
reads that key. Metrics without item-level scores (WEAT, SEAT, CEAT, CBS)
return `ci=None` unless the original paper defines an interval or test, in
which case `run()` calls the metric's own `_interval()` (permutation p-value
for WEAT/SEAT, Hedges–Olkin for effect sizes).

- [x] `bias_scope/result.py` (`BiasResult`, `make_protocol`, `from_dict`) and `bias_scope/stats.py` (`bootstrap_ci`, `wald_ci`, `hedges_olkin_ci`, `permutation_p`; each a short function with a docstring formula and a known-answer test).
- [x] `run()` in `base.py`; every metric returns `details["per_item"]` where it has per-item scores; a test in `tests/test_run.py` calls `run()` on every metric with tiny inputs and checks the `BiasResult` fields.
- [x] `tests/test_examples/` still passes with no edits to the examples.

### 5.4 `BiasSuite`, `recommend_metrics`, reports

Three small modules, functions first, classes only where state is needed.

```python
model = load_model("meta-llama/Llama-3.1-8B-Instruct", backend="hf", dtype="bf16")   # or backend="litellm"
suite = BiasSuite(model, axis="gender", language="en")        # picks applicable metrics from MetricInfo.access
report = suite.run(seed=42)                                    # list[BiasResult] + model/protocol summary
save_json(report, path); to_markdown(report); to_html(report, path)
compare(report_a, report_b)                                    # paired deltas with CIs, one row per metric
correlate([report, ...], method="spearman")                    # metric x metric matrix across models
recommend_metrics(access=("chat",), axis="race", language="fr") # list of (metric name, reason) tuples
```

- [x] `bias_scope/suite.py`, `bias_scope/recommend.py`, `bias_scope/report.py`. `report.py` builds Markdown and HTML with `string.Template` and f-strings; no templating library.
- [x] Two backends only, in `bias_scope/backends.py`: Hugging Face transformers (encoders and causal LMs; gives embeddings, logits, completions) and LiteLLM (chat; gives completions and chat). vLLM stays in `scripts/` for speed, not in the library. Access is derived from the backend, never guessed.
- [x] Axis support: `gender` for all applicable metrics first; `race`, `religion`, `age` where the datasets support them. A metric declares its supported axes in `MetricInfo` only if it needs more than `gender` (add a field then, not before).
- [x] Generation cache: one function `cached_generate(model, prompts, decoding, cache_dir)` keyed by protocol hash, storing JSON lines. No database.
- [x] Tests with a stub backend (returns canned embeddings/logits/completions) covering metric selection, caching, `compare`, `correlate`, and all three report formats; an integration test runs `BiasSuite` on `sshleifer/tiny-gpt2` end to end.
- [x] Acceptance checks (each is a test): the selected metric set equals exactly the metrics whose `access` ⊆ backend access and `languages` ∋ language; a second `suite.run()` on the same model makes zero generation calls (cache hit counter); `compare` deltas equal the differences of the individual scores; `correlate` on synthetic reports with a known correlation returns that matrix; every `BiasResult` in a report appears in all three output formats; the HTML parses with `html.parser` and contains one section per family and the protocol table; `recommend_metrics` never returns a metric the given access cannot run.
- [x] Real run recorded: `BiasSuite` on one HF model and one LiteLLM model, outputs saved under `results/verification/suite/`, abbreviated output in `PROGRESS.md`.

---

## 6. Phase 3 — Validation (three tiers, one table)

### 6.1 Registry

`validation/registry.yaml`, one entry per (metric, model, dataset):

```yaml
- metric: CrowS-Pairs
  tier: 1
  paper: "Nangia et al. 2020"
  location: "Table 3, row bert-base-uncased, column 'metric score'"   # confirm before use
  model: bert-base-uncased
  dataset: nyu-mll/crows-pairs
  dataset_revision: <sha>
  published_value: 60.5            # null if not found; never guess
  published_n: 1508
  ci_method: wald
  protocol: {mode: wordpiece, dtype: fp32, seed: 42}
  status: pending | matched | close | off | no_published_reference
```

Pre-fill only values you have confirmed by reading the table. Candidates with a
public model, public data, and a reported number: WEAT-6 (Caliskan Table 1),
SEAT (May et al. per-test tables), CEAT (Guo & Caliskan), CrowS-Pairs on BERT /
RoBERTa / ALBERT (Nangia Table 3), AUL/AULA (Kaneko & Bollegala), CAT/ICAT
on BERT and GPT-2 (Nadeem), LPBS (Kurita), CBS (Ahn & Oh), DisCo (Webster),
LMB (Barikeri), HONEST GPT-2 (Nozza Table 3), Regard GPT-2 (Sheng), BOLD
metrics for GPT-2 (Dhamala), CounterfactualSentimentBias (Huang), StereoSet
GPT-2 (Nadeem), UnQover (Li), TruthfulQA MC1/MC2 for any open model with a
reported number, MarkedPersons (Cheng). For BBQ, find one precisely citable
source for an open model or record `no_published_reference` and rely on Tier 2;
do not reuse the v0.1 "midpoint of a range" anchor.

### 6.2 Tiers

- **Tier 1 — published-value reproduction.** `scripts/validation/run_tier1.py`
  runs every `pending` registry entry, writes `results/validation/tier1/<metric>__<model>.json`
  with score, n, CI, protocol, wall time, peak VRAM. Criterion: 95% CIs overlap
  (`matched`); point estimate within 10% relative but CIs disjoint (`close`);
  otherwise `off`. Stochastic metrics run 3 seeds; report mean and spread.
- **Tier 2 — reference-implementation equivalence.** `scripts/validation/reference_impls/fetch.sh`
  clones each reference repo at a pinned SHA into `third_party/` (git-ignored;
  licenses recorded in `third_party/LICENSES.md`). `tests/equivalence/test_<metric>.py`
  runs the reference code and BiasScope on identical cached inputs (same model
  outputs, same items) and asserts per-item Pearson r ≥ 0.99 and max |Δ| ≤ 1e-4
  for deterministic metrics; document looser bounds with reasons when the
  reference is stochastic. Skipped unless `BIASSCOPE_RUN_EQUIVALENCE=1`. Where a
  reference repo will not run under a supported Python, record the failure
  (Python/torch versions required, error) — this feeds the effort study (10.3).
- **Tier 3 — metamorphic property tests** in `tests/properties/`, parametrised
  over all metrics via the registry:
  1. *Null*: identical or perfectly symmetric inputs → score equals `neutral_value`.
  2. *Swap antisymmetry*: swapping groups maps the score to its mirror about `neutral_value` (or flips the sign).
  3. *Monotonicity*: injecting k more stereotyped items moves the score away from neutral monotonically in k.
  4. *Permutation invariance*: item order does not change the score.
  5. *Scale invariance* (embedding metrics): multiplying all vectors by a constant does not change the score.
  Mark any property a metric cannot satisfy by design with a reason in its fidelity note.

- [ ] Registry complete for all 40 + new metrics (Tier 1 entries or explicit `no_published_reference`).
- [ ] Tier 1 runs complete; every `off` result has a written hypothesis.
- [ ] Tier 2 harness runs for every metric with a runnable reference; unrunnable references documented.
- [ ] Tier 3 passes for every metric or has a documented exemption.
- [ ] Every metric's ledger row has criteria 1–7 (Section 1) filled with evidence paths; `render_ledger.py` passes.
- [ ] `scripts/validation/render_validation_table.py` → `results/validation/VALIDATION.md` and `validation_table.tex` (40+ rows × {Tier 1, Tier 2, Tier 3, fidelity}).

---

## 7. Phase 4 — Judge backend and new metrics

### 7.1 Judge abstraction

- [ ] `bias_scope/judge.py`: one class, `Judge(model_name, prompt_file, api_key=None, cache_dir=None)`, with one public method `classify(text, labels) -> str` (and `classify_batch`). It calls LiteLLM, parses the label from the output, caches by (model, prompt file hash, text) in JSON lines, and counts cost against `BIASSCOPE_MAX_API_USD`. Under 120 lines. Logprob-based grading is a later option, not a first version.
- [ ] Judge prompts live in `bias_scope/resources/judge_prompts/<metric>_v1.txt`; `protocol["judge_prompt_version"]` records the file name and hash.
- [ ] Refactor `TofNof` onto `Judge` with a test that uses a stub judge returning canned labels.
- [ ] Acceptance checks: label parsing tested on 20 hand-written judge outputs (clean, verbose, malformed → `None` and logged); a second call on the same text hits the cache (counter); the cost counter increments and the cap raises before the call; one real 20-item run with a local open judge recorded under `results/verification/judge/`.
- [ ] `docs/judges.md`: judge choice is part of the protocol; report judge model and prompt version with every number; provide one cross-judge agreement check (two judges on a 200-item subsample) in the studies.

### 7.2 New metrics (prompt family unless noted; each with MetricInfo, property tests, docs, example, and a registry entry)

Prerequisite for every item: the Section 4.0 gate. For these metrics the
authors' prompts, judge instructions, answer parsing, and dataset splits are
part of the definition; read them verbatim from the repo and mirror them,
then prove it with the Tier-2 equivalence test on the authors' released
examples or outputs where available. Where a repo ships its own scorer, the
first version of the BiasScope metric may call that scorer through a thin
adapter, with the native port as a follow-up item.

- [x] **PoliticalEvenHandedness** — ported from github.com/anthropics/political-neutrality-eval @ c5ed67908b56 (**faithful**; all three dimensions reported separately, because a model that refuses both sides scores perfectly even-handed): paired left/right prompts; judge grades even-handedness, opposing-perspectives, and refusal; report all three with CIs. Record that the original grader used Claude with token probabilities and that results vary by grader.
- [x] **FirstPersonFairness** (Eloundou et al. 2024, OpenAI) — **adaptation**: the estimator H = E[h_F − h_R] is exact (net score, swapped second pass, identical-response rule); the judge template is Figure 3, which the paper marks "slightly abbreviated", and no released code was located (search log in SOURCES.yaml). — name-substitution counterfactuals on user-style prompts; judge flags harmful stereotypes; reuse `SocialGroupSubstitution` for the substitution step. Locate and cite the released prompts/grader.
- [x] **DecodingTrust stereotype and fairness** (Wang et al. 2023 @ 161ae8321ced) — **faithful** ×2. `classify_response` transcribed with all three phrase lists; DPD/EOd reimplemented and checked against fairlearn 0.14.0 (the package the reference calls) to 1e-12 on 2000 cases. Two reference quirks pinned: "I cannot agree or disagree" scores DISAGREE, and `'no' in answer` matches "not". — stereotype agreement rate under the benchmark's system prompts; fairness module on the released counterfactual prompts.
- [x] **TrustLLM fairness subset** (Huang et al. 2024 @ 4b864211f326) — **faithful** ×4: stereotype recognition, stereotype agreement, disparagement, preference. Three of the four have 1.0 as their ideal value, so they are four metrics and not one. — stereotype recognition, stereotype agreement, disparagement, preference.
- [x] **DiscrimEval** (Tamkin et al. 2023; dataset `Anthropic/discrim-eval`) — **adaptation**: marginal logit differences rather than the paper's mixed-effects coefficients (needs statsmodels; see docs/fidelity/discrim_eval.md) — decision questions with demographic variations; logit/yes-probability difference per attribute.
- [x] **LLM implicit association and decision bias** (Bai et al., PNAS 2025 @ 0d2772e8eb21) — **faithful** ×2 (`ImplicitAssociationTest`, `LLMDecisionBias`). **Tier 2 equivalent: 18,885/18,885 released responses exact to 1e-9.** — LLM-IAT word-association prompts and decision-bias prompts; verify the authors' protocol and code before implementing.
- [x] **WinoBias** (Zhao et al. 2018) — **faithful**, chat path done (pro/anti accuracy gap over the paper's own data files). The encoder logit version and Tier 2 against `uclanlp/corefBias` remain.
- [ ] **FGB/PGB** faithful (from 5.2) count here as HolisticBias support.
- [x] **Multilingual variants** — `bias_scope.multilingual`: loaders for MBBQ, KoBBQ, CBBQ, the French CrowS-Pairs extension, plus provenance for SHADES/HONEST/CBS. Nothing redistributed (Section 12), and a test asserts no dataset file ships in the package. `languages` made truthful from the authors' own files: CBS 12, HONEST 6.: `BBQMetric` loading MBBQ (Neplenbroek et al. 2024), KoBBQ (Jin et al. 2024), CBBQ (Huang & Xiong 2024) where licenses allow; French CrowS-Pairs (Névéol et al. 2022) and the multilingual extension; multilingual HONEST templates and HurtLex languages; SHADES (Mitchell et al. 2025) if redistributable. Every metric's `languages` and `resource_binding` become truthful.
- [x] `docs/inclusion_criteria.md` and `docs/roadmap.md` written, with the considered-but-excluded table and its failing criterion per row. Original text: peer-reviewed or widely adopted; formula or reference code fully specified; fits one of the four families; no proprietary-only dependency (judge may be any LiteLLM model). `docs/roadmap.md`: considered-but-excluded list with the failing criterion.
- [x] Recount for the paper, generated by `scripts/paper/metric_counts.py` (never typed by hand): **55 metrics — 27 faithful, 16 adaptation, 9 original, 2 mismatch, 1 unaudited**; by family embedding 4, probability 11, generated_text 17, prompt 23. Do not market adaptations/originals as the cited papers' metrics.

---

## 8. Phase 5 — Adapters

- [ ] `bias_scope/adapters/lm_eval.py`: wrap lm-evaluation-harness tasks (`crows_pairs`, `truthfulqa_mc1/mc2`, `winogender`, BBQ if present) so they return `BiasResult` with `fidelity='faithful'` and `reference_impl` pointing at the harness. Extra `adapters`.
- [ ] `bias_scope/adapters/hf_evaluate.py`: wrap `evaluate` measurements `toxicity`, `regard`, `honest`.
- [ ] Equivalence tests: native BiasScope vs adapter on the same model/inputs (Tier 2 for free); document any disagreement and its cause.
- [ ] Acceptance checks: each adapter returns a complete `BiasResult` (runtime guards pass); adapter and native agree within the stated tolerance on `prajjwal1/bert-tiny` / `sshleifer/tiny-gpt2`; an `ImportError` for a missing optional dependency gives a message naming the extra to install.
- [ ] `recommend_metrics` knows about adapter-backed metrics and prefers the native implementation when both exist and agree.

---

## 9. Phase 6 — Visualization and HTML report (extra `viz`: matplotlib; HTML uses inline SVG, no server)

Rules: never render a composite score, gauge, traffic light, or single-polygon
radar; every plotted value carries its CI; every figure footer shows the protocol
hash, seed, dtype, and resource versions; fidelity status appears as a badge.
Each plot is one plain matplotlib function taking a report and returning a
`Figure`; no plotting classes, no styling framework, shared helpers in one
`_style.py` under 80 lines. Every function has a snapshot test on a fixture
report (compare to a stored PNG hash or to the drawn data, not pixels).

- [x] `plot_profile(report)`: rows = metrics grouped by family; x = `normalized_deviation` with CI whisker and a neutral line; badge per row.
- [x] `plot_dumbbell(report_a, report_b)`: same layout, two markers per row (base vs instruct, or two models).
- [x] `plot_agreement(corr_matrix)`: metric × metric Spearman heatmap with n models and significance marks.
- [x] `plot_envelope(runs)`: one metric, one model, value across protocol variants (lexicon snapshot, dtype, tokenization mode, decoding, n) as a band.
- [x] `plot_forest(result)`: per-group breakdown with CIs for metrics with `breakdown`.
- [x] `report.to_html()`: self-contained page with the profile, per-family sections, per-metric details, protocol table, and fidelity notes; opens offline.
- [x] Tests assert on the **drawn artists and data**, not pixels — a pixel snapshot breaks on every matplotlib release and says nothing about correctness.
- [x] Acceptance checks: the profile figure has one row per result and one CI whisker per item-level result (count the Axes artists); the footer text contains the protocol hash; `plot_dumbbell` raises if the two reports have different metric sets; the HTML report renders all figures inline and opens offline (no external URLs in the file, checked by a test).

---

## 10. Phase 7 — Studies (outputs under `results/studies/`, each with a `README.md`, CSVs, figures, and the exact command line)

### 10.1 Cross-model agreement
- [ ] Models (all BF16/FP32, no quantization): encoders `bert-base-uncased`, `roberta-base`, `albert-base-v2`; causal base `gpt2`, `EleutherAI/pythia-1.4b`, `EleutherAI/pythia-6.9b` or `allenai/OLMo-2-1124-7B`, `meta-llama/Llama-3.1-8B`, `Qwen/Qwen2.5-7B`, `mistralai/Mistral-7B-v0.3`; instruct `meta-llama/Llama-3.1-8B-Instruct`, `Qwen/Qwen2.5-7B-Instruct`, `google/gemma-2-9b-it`, `mistralai/Mistral-7B-Instruct-v0.3`. Adjust to what fits a 20 GB GPU; record substitutions.
- [ ] Run `BiasSuite` on the gender axis for every model; every applicable metric; 3 seeds for stochastic metrics.
- [ ] `BiasSuite.correlate` → Spearman and Kendall matrices; within-family vs across-family mean |ρ|; bootstrap CIs on ρ over models. Figures via `plot_agreement`.
- [ ] Base vs instruct pairs via `plot_dumbbell`; paired deltas with CIs per family.

### 10.2 Protocol sensitivity
- [ ] HONEST: HurtLex subset/version × decoding (nucleus, top-k, beam) × K. CrowS-Pairs: whitespace vs WordPiece PLL × model. BBQ: dtype (BF16 vs AWQ-INT4) × n (200, 1000, full) × prompt template (3). RealToxicityPrompts: classifier (Perspective vs HF) × K. Report ranges and `plot_envelope`.
- [ ] Quantify: for each metric, the spread across defensible protocols vs the spread across models. This is the "same name, different numbers" finding.

### 10.3 Effort and reproducibility survey
- [ ] For every reference implementation in `third_party/`: last commit date, declared Python/torch versions, installs today (yes/no, error), runs on its own example (yes/no), supported model types, lines of glue code needed to run it on one new model (write the glue, count it), wall time. Same columns for BiasScope (`pip install`, lines, time).
- [ ] Model-coverage matrix: metric × {encoder, causal, chat API} for original code vs BiasScope.
- [ ] Optional if time: a timed task study (6 participants, 3 metrics, with/without BiasScope; time-to-correct-number and error count). Needs ethics check; default is to skip.

---

## 11. Phase 8 — Final verification, then release

### 11.1 Final verification (all from a fresh clone in a clean virtual environment; record every command and result in `results/verification/FINAL.md`)

- [ ] Fresh clone + `pip install -e ".[all,dev]"` on Python 3.10 and 3.13; `ruff check` clean; `pytest -q --cov` green with coverage ≥ 90%; `pytest -m slow` green; `BIASSCOPE_RUN_EQUIVALENCE=1 pytest tests/equivalence` green or every skip documented.
- [ ] Install each extra alone (`torch`, `embeddings`, `datasets`, `llm`, `judge`, `adapters`, `viz`) in its own venv and import the corresponding modules; missing-extra errors name the extra.
- [ ] Run every file in `examples/` and every README snippet; outputs match the recorded outputs in `results/verification/examples/`.
- [ ] Run every `scripts/validation/*.py --smoke`, then the full Tier-1 runs; statuses in `registry.yaml` match `VALIDATION.md` after `render_validation_table.py`.
- [ ] Full `BiasSuite` on two real models (one HF, one LiteLLM); every applicable metric produces a `BiasResult` passing the runtime guards; report opens; numbers for the four v0.1 metrics agree with `results/emnlp/` within CI.
- [ ] Regenerate every study figure and table from scratch with the one-command scripts; diff against the committed versions; any difference explained.
- [ ] `render_ledger.py` passes with zero empty cells for every metric and feature; `check_manifest.py` passes.
- [ ] Docs build (`mkdocs build --strict`) with no warnings; every metric page shows fidelity status, formula, and example.
- [ ] Read `REVIEW_LATER.md` end to end; every `blocked` entry is reflected as an unticked box; every `verify` entry points at the file the maintainer must open.

### 11.2 Release and paper artefacts

- [ ] `README.md`: suite/report/recommend quick start first; fidelity statuses visible in the metric list; "what's an adaptation" paragraph.
- [ ] `docs/`: new pages for metadata, results, suite, recommend, judges, adapters, viz, fidelity index, inclusion criteria, roadmap, multilingual readiness table.
- [ ] `CHANGELOG.md` for 0.2.0 listing every rename, every reimplementation, and every behaviour change with the reason.
- [ ] Version bump to 0.2.0 in `pyproject.toml`; build; test install in a clean venv for each extra; PyPI release after maintainer approval.
- [ ] `scripts/paper/`: generators for Table 1 (validation matrix), Table 2 (metric table with fidelity status and deviation notes), the agreement heatmap, the envelope figure, the dumbbell figure, and the effort table. Paper numbers must be produced by these scripts from `results/`, never typed by hand.
- [ ] Fix the paper-side inconsistencies flagged in review: Figure 1 family placement of StereoSet; Listing 1 line numbers; "CrowS-Paris"; "Llama-3.1-8B-Instruction"; contribution 1 fragment; duplicated references (Dhamala 2021a/b, Gehman 2020a/b, Nadeem 2021a/b); TruthfulQA reference; explain MarkedPersons' reported scalar; replace the 10% tolerance with the CI criterion; add WEFE, Hugging Face `evaluate`, lm-evaluation-harness, HELM, DecodingTrust, TrustLLM, LangFair to related work; narrow "first to organize by paradigm" to "first library implementing all four".

---

## 12. Pre-decided defaults (apply them; log each application in `REVIEW_LATER.md` as `decide`)

| Question | Decision to apply |
|---|---|
| Keep old wrong-metric behaviours under new names or delete them? | Keep under new names, status `original` (they are reasonable statistics; only the labels were wrong) |
| Default CI method for proportions | Bootstrap percentile (Wald reported alongside) |
| Judge model for judge-based metrics in validation runs | One open model run locally (e.g. Qwen2.5-7B-Instruct or Llama-3.1-8B-Instruct if accessible) plus one API judge on a 200-item subsample for agreement, if a key is present |
| API spend cap | USD 25 per study; when reached, finish with the local judge only |
| TruthfulQA in a bias library | Keep, labelled "truthfulness (not social bias)", excluded from bias profiles by default |
| Multilingual datasets with non-commercial licenses | Loader only, no redistribution |
| Timed user study | Skip; the repo-rot/effort table stands in for it |
| A fix changes a metric's sign convention or range | Make the change, bump the metric's docstring and `MetricInfo`, add a `CHANGELOG` entry under "Breaking", tag `decide` |
| Two defensible readings of a paper | Implement the simpler one, expose the other behind a keyword argument only if it is a one-line difference, tag `decide` |

---

## 13. Definition of done for v0.2

- [ ] Every metric has `MetricInfo`, a fidelity note, a `SOURCES.yaml` entry with the paper and code actually read, and a registry entry.
- [ ] Zero `mismatch` statuses remain.
- [ ] Every metric has at least one objective validation (Tier 1 matched/close, Tier 2 equivalent, or Tier 3 passing with documented exemptions), and `VALIDATION.md` shows it.
- [ ] `BiasSuite.run`, `compare`, `correlate`, `recommend_metrics`, and the three report formats work end-to-end on at least two models, one of them API-only.
- [ ] At least six new metrics (7.2) including two judge-based ones and one multilingual dataset variant.
- [ ] Studies 10.1–10.3 complete with figures regenerable from one command.
- [ ] Coverage ≥ 90% on `src/bias_scope`; `ruff check` clean with complexity cap 10; fast suite under 3 minutes on CPU; every `examples/` file runs in CI.
- [ ] `results/verification/VERIFICATION.md` complete: every metric row has criteria 1–7 with evidence, every feature row has its acceptance checks, and Section 11.1 was executed from a fresh clone with `FINAL.md` written.
- [ ] Readability pass: `docs/architecture.md` is under two pages and a new contributor can add a metric by copying one existing module, filling `MetricInfo`, writing the known-answer test, and adding one registry entry (documented as a 6-step checklist in `docs/contributing.md`).
- [ ] CI green on 3.10–3.13; 0.2.0 on PyPI; paper tables generated from `results/`.

---

## 14. Phase 9 — `bias_scope_agent`: an LLM tool-calling agent over `bias_scope`

A thin, separate top-level package (`src/bias_scope_agent/`) that adds a
single agent LLM (Claude, via the `anthropic` SDK) in a tool-calling loop,
driving `bias_scope`'s existing `recommend_metrics`, `explain_exclusions`,
`BiasSuite.plan()`/`.run()`, and `Report` rendering. `src/bias_scope/` is not
modified by this phase — it adds a consumer of the library, not a change to
it.

**Section 4.0's mandatory source-retrieval gate does not apply to this
phase.** No new bias metric is added or touched here — this is agent/harness
code wrapping the existing, already-audited library. Do not go looking for a
paper or authors' code to cite for tool-wrapper functions.

- [x] Phase 0 — scaffolding: `src/bias_scope_agent/` package, `AgentConfig`/`load_config` (env-var config; no agent-specific API key — the `anthropic` SDK already reads `ANTHROPIC_API_KEY`, see `REVIEW_LATER.md` RL-038).
- [x] Phase 1 — `HandleRegistry` (opaque UUID handles for backends/reports, so live objects never cross the tool-call boundary) and `introspection.py` (`metrics_needing_data` via `inspect.signature`, since `MetricInfo` has no input-shape field; `inspect_model` best-guess causal/encoder/API-endpoint detection, live by default behind `BIASSCOPE_AGENT_INSPECT_LIVE`).
- [x] Phase 2 — tool wrappers (`tools.py`): `construct_backend`, `recommend_metrics_tool`, `explain_exclusions_tool`, `plan_suite`, `request_missing_inputs`, `confirm_plan`, `run_suite`, `summarize_report`, `record_fact`. No metric-selection logic added — all of it stays in `bias_scope`.
- [x] Phase 3 — the confirm-before-run gate (`session.py`, `AgentSession`/`GateError`): a *structural*, application-level guarantee (plan shown, a real turn boundary passed, `confirm_plan` explicitly called) enforced by the tool dispatcher before `run_suite` is ever invoked — not just a prompted instruction. Proven in tests via a spy asserting the real function is never called when the gate rejects a call.
- [x] Phase 4 — Anthropic tool-use JSON schemas (`schemas.py`) and the system prompt (`system_prompt.py`), rendering `session.facts` each turn so answered clarifications are never re-asked.
- [x] Phase 5 — `AgentLoop` (`loop.py`): the tool-calling loop, dispatching by name at call time (not bound once at construction) so tests can spy on/patch individual tool functions.
- [x] Phase 6 — CLI (`cli.py`, `__main__.py`, `bias-scope-agent` console script): a minimal REPL, not the focus of this feature.
- [x] Phase 7 — tests: `tests/test_bias_scope_agent/` (one file per concern: config, registry, introspection, tools, session gate, schemas/prompt, loop, cli) plus `tests/integration/test_bias_scope_agent_tiny_model.py` (real `HuggingFaceBackend` end to end, using `WEAT` — see `REVIEW_LATER.md` RL-041 for why `CrowSPairs`/`AUL` were not usable here).
- [x] Phase 8 — this section, `REVIEW_LATER.md` RL-038–RL-041, `PROGRESS.md` entry, `pyproject.toml` (wheel packages, `agent` extra, `[project.scripts]`, coverage source).

**Follow-up (closing the gaps found in review, before first real use):**

- [x] Item 1 — done 2026-09-18, via **OpenRouter** rather than Anthropic (a supervisor-supplied key). Three live conversations recorded in `results/verification/agent_live/`, target `bert-base-uncased` on the GPU (fp32, `cuda:0`) scored on the authors' own CrowS-Pairs gender pairs; `tests/integration/test_bias_scope_agent_live_conversation.py` added, opt-in via `BIASSCOPE_RUN_LIVE_AGENT=1` and marked `slow`. The requested agent model `~typesafe/jev-latest` turned out to be a decisions-API model that cannot drive a tool-calling loop at all (RL-051). The gate held in both directions and the 2026-09-17 fabrication did not recur; two defects found — `needs_data` under-reporting constructor requirements (RL-052, fixed) and silent corruption of metric inputs as they pass through the agent's output tokens (RL-053, architectural, not fixed).
- [x] Item 2 — commit the branch (`agent-implementation`, pushed to origin).
- [x] Item 3 — packaging verified in a genuinely clean virtualenv (`pip install -e ".[agent]"`); found and fixed a real UX gap: a missing `ANTHROPIC_API_KEY` previously surfaced as a bare `TypeError` ~15 frames deep in the SDK on the *first turn*, not at startup — `providers.py`'s `_build_*_client` functions now check for the key (and for the SDK being installed) before constructing anything, raising a clear `RuntimeError` immediately naming the missing variable.
- [x] Item 4 — RL-040 decided: Option A (structural gate stays as-is; `system_prompt.py`'s confirmation language tightened to require unambiguous affirmation, explicitly naming hedges/questions/silence as not confirmation). Option B (a deterministic hedge-phrase denylist) rejected for now — no live run has shown the agent actually misreading a reply, and a phrase heuristic is itself a coarse, easily-wrong signal. Revisit if Item 1 (or production use) shows otherwise.
- [x] Item 5 — a three-turn scripted test (`test_loop_scripted_conversation.py::TestRecordedFactsReachLaterTurns`) proves `record_fact` in turn 2 is present in the real `system` prompt string `AgentLoop` sends on turn 3 — the actual plumbing, not just `render_system_prompt()` in isolation. Documented explicitly what this can and cannot prove: a canned-response fake client cannot demonstrate a *real* LLM would actually stop re-asking; only Item 1 can.
- [x] Item 6 — full multi-provider support built: `AgentConfig.provider` (`BIASSCOPE_AGENT_PROVIDER`, default `anthropic`), new `src/bias_scope_agent/providers.py` with `AnthropicProvider`/`OpenAIProvider`/`GeminiProvider`, each translating `schemas.TOOLS` to that provider's function-calling shape and normalizing its response back to a shared `NormalizedResponse` — `loop.py` itself never branches on provider. New `agent-openai`/`agent-gemini` extras (not folded into the base `agent` extra, so a Claude-only install stays at two dependencies). One parametrized scripted conversation proves the gate and tool dispatch behave identically across all three real adapters (`test_providers.py::TestGateBehavesIdenticallyAcrossProviders`). Not yet run against real OpenAI/Gemini APIs — see `DECISIONS.md`'s 2026-09-14 entry.
- [x] Item 6b (added later, by request) — a fourth provider, `local`: any OpenAI-compatible local server (Ollama, llama.cpp, LM Studio, vLLM). `LocalProvider` subclasses `OpenAIProvider`, overriding only client construction (a different `base_url`, no required API key) — reuses `create()` unchanged since the wire format is identical. No new dependency (reuses `agent-openai`'s `openai` package). 6 new tests.
- [x] Item 7 (added later, by request) — target-model UX improvements, in response to a review of how smooth/hard it actually is to specify a target model:
  - `inspect_model` now names encoder-decoder/seq2seq models (T5, BART, ...) as explicitly unsupported by `HuggingFaceBackend` instead of leaving them an unexplained low-confidence guess (RL-044-adjacent).
  - `inspect_model` falls back to a small `model_type`-based lookup (llama/mistral/gpt2/... → causal, bert/roberta/deberta/... → encoder) when the architecture-string heuristic alone doesn't classify, raising the high-confidence rate for common open-weight models (RL-044).
  - `inspect_model` checks a plain identifier against litellm's bundled, offline model registry before ever attempting a doomed HF Hub lookup for something like `"gpt-4o-mini"` — classified as `guessed_source="litellm_model_id"`, confidence high, no network call; a near-miss (typo) gets a "did you mean" note instead (RL-045).
  - `construct_backend`'s LLM-facing schema no longer exposes `api_key` — structurally, not just by prompted instruction, so the target model's own credentials can never enter the conversation transcript. `system_prompt.py` tells the agent to have the user export the standard provider env var instead. See `DECISIONS.md`'s later 2026-09-14 entry.
  - 11 new tests (9 introspection, 2 schema/prompt).
- [x] Item 8 (added later, by request — supervisor wants OpenRouter + litellm access) — two more `AgentConfig.provider` values in `providers.py`:
  - `openrouter` — `OpenRouterProvider(OpenAIProvider)`, points the existing OpenAI-shaped client at OpenRouter's own OpenAI-compatible endpoint. Requires `OPENROUTER_API_KEY`, fails fast with a clear message like every other cloud provider.
  - `litellm` — `LiteLLMProvider(OpenAIProvider)`, a general escape hatch routing through `litellm.completion()` directly (100+ providers, OpenRouter included via litellm's own `"openrouter/<slug>"` prefix), via a small shim (`_wrap_litellm_client`) rather than a new translation — litellm's own response shape is already OpenAI's. No eager API-key check (litellm resolves the right variable per model-string prefix itself; see `DECISIONS.md`).
  - 13 new tests (`test_providers.py`, `test_config.py`). See `DECISIONS.md`'s 2026-09-17 entry for why this revisits, without reversing, the earlier not-litellm-for-the-agent-LLM decision, and `REVIEW_LATER.md` RL-046 for the not-yet-run-against-a-real-key caveat both new providers share with every other one.
- [x] Item 9 (added later, by request) — make the agent able to evaluate the *recommended* metrics on all three kinds of target model (causal, encoder, embedding) and report one summary of the results. Six defects stood in the way, all found by running it against `deepseek/deepseek-v4.1-flash` over OpenRouter, all fixed test-first:
  - **Data by reference** (`src/bias_scope_agent/datasets.py`, RL-053 closed). `list_datasets`/`prepare_inputs` load the authors' vendored files server-side and hand back a handle plus provenance (path, sha256, counts); `run_suite` takes several handles at once so a multi-dataset evaluation is one report. Evaluation data no longer passes through the agent's output tokens, where two models had already altered and truncated it. Providers declare which metrics they serve, which also stops `LMB`/`PairwiseLikelihoodPreference` being fed CrowS-Pairs (they need equal-length pairs) and stops a WEAT test being substituted for an axis Caliskan never measured.
  - **RL-055**: `run_turn` returned only the final response's text, so the plan the user was asked to confirm was never printed.
  - **RL-056**: `_embed_cls` called `.numpy()` on BFloat16, breaking every embedding metric on the dtype Section 1 *requires* for causal LMs.
  - **RL-057**: a causal backend advertised `logits`, so all 11 probability metrics were recommended and all 11 failed — every consumer of `logits` here is a masked-token scorer.
  - **RL-058**: a checkpoint with no LM head (`all-MiniLM`, architectures `["BertModel"]`) loaded through `AutoModelForMaskedLM` without failing, so `CrowSPairs` returned 0.4000 from randomly initialized weights, badged `faithful`.
  - **RL-059**: `check_run_gate` required the run's metric set to *equal* a confirmed plan's, which made a multi-dataset evaluation impossible; it now accepts a subset (an unapproved metric and an empty set are still refused).
  Then, on request, the logged-but-deferred failures were closed and dataset coverage widened: **RL-060** (CrowSPairs/AUL/AULA returned a fraction while their metadata, the authors' scorers and the papers all say percent — `normalized_deviation` had the wrong *sign*), **RL-061** (CAT/ICAT unreachable because `_split_result` had to guess which of their numbers was the score; metrics now declare `headline_key`), **RL-048** (CEAT's `n_samples` is the observation count — the authors pool with `df = N - 1`), **RL-054** (`BiasSuite.run` mutated the caller's inputs) and **RL-062** (`RegardScore` had neither headline nor count). Two new dataset providers: `stereoset` (CAT, ICAT) and `bold_regard`, the first that **generates** — BOLD's own prompts continued by the model under evaluation and scored by Sheng's regard classifier, so a causal LM is finally evaluated on generation rather than only on its embeddings. Feedable share of the recommended set: encoder 5 → 7 of 15, causal 2 → 3 of 19, embedding 2 of 4.
  Runs recorded under `results/verification/agent_live/`; `scripts/agent/summarize_runs.py` tabulates them from the library's own recorded output. **RL-060 opened, not fixed:** `CrowSPairs`/`AUL`/`AULA` return a fraction while their `MetricInfo`, the authors' `metric.py:270`, Nangia's Table 3 and `validation/registry.yaml` all say percent, so `normalized_deviation` reports the wrong sign.

- [x] Item 10 (2026-09-19, by request) — run the agent on more target models and check that the *recommended* metrics are the ones it actually runs. Five new models (`bert-base-cased`, `roberta-base`, `Qwen2.5-0.5B-Instruct`, `gpt2`, `all-mpnet-base-v2`), eight recorded conversations, `deepseek/deepseek-v4.1-flash` over OpenRouter.
  - **The check itself was missing.** `recommend_metrics_tool`, `plan_suite`, `run_suite` and `summarize_report` were each recorded, but nothing compared them. `scripts/agent/live_conversation.py::recommendation_coverage` now computes recommended → feedable (a dataset provider serves it and the backend has the access it needs) → planned → run → scored, and `complete` is the goal condition: every feedable metric scored, nothing scored that was never recommended. Recorded in every new transcript; `summarize_runs.py --check` recomputes it for the old ones too (which reproduces PROGRESS.md's own account: the early bert-base-uncased runs fed 0, 1 and 5 of 7). `tests/test_bias_scope_agent/test_recommendation_coverage.py` asserts every recorded run since the RL-060 fixes is complete, with an explicit exception list for the two gpt2 transcripts kept as evidence of RL-067.
  - **RL-066 (fixed):** RL-058 recurred past the config check. `all-mpnet-base-v2` lists `MPNetForMaskedLM` yet ships no `lm_head.*` weights; five probability metrics were run on a random head and badged `faithful` (AULA exactly 50.00). `HuggingFaceBackend` now loads the masked-LM model once and withholds `logits` when any head weight is missing. The rerun recommends 4 metrics instead of 15 and scores WEAT 1.257, SEAT 1.042 only. The invalidated transcript is under `results/verification/agent_live/invalidated/`.
  - **RL-067 (fixed):** WEAT and SEAT were skipped on `gpt2` - no pad token - in both embedding loaders (`cls` and sentence-transformers `mean`). Two reruns, one per path; the third gpt2 run is complete.
  - **RL-068 (verify, not changed):** `pooling='cls'` on a decoder-only LM reads the first token's hidden state; the causal SEAT numbers are of uncertain meaning. **RL-069 (verify):** the agent's interpretive prose is sometimes wrong where its numbers are right.
  - Results (library output, not prose): bert-base-cased CrowSPairs 57.63 / AUL 53.05 / AULA 53.82 / CAT 64.19 / ICAT 59.11 / WEAT 0.3792 / SEAT 0.9246; roberta-base 54.96 / 56.49 / 53.44 / 55.46 / 61.07 / −0.6074 / 1.099; Qwen2.5-0.5B WEAT 0.847, SEAT 0.2512, RegardScore 0.02; gpt2 WEAT 0.5183, SEAT −0.0486, RegardScore 0.02; all-mpnet-base-v2 WEAT 1.257, SEAT 1.042. WEAT and CrowSPairs on bert-base-cased recomputed outside the agent through the same `prepare_inputs` path: identical.

- [x] Item 11 (2026-09-20, by request) — make the *generated-text* half of the recommended set feedable without human input. Six dataset providers, each from the authors' own release, each test-first except the first two (built before their tests; noted in PROGRESS.md): `bold_gender_polarity` (GenderPolarity; BOLD profession prompts, BOLD §4.5 lists), `bold_helm_bias` (DemographicRepresentation, StereotypicalAssociations; HELM's own word lists, adjectives as targets), `honest` (HONEST; Nozza's templates, K=20, HurtLex EN 1.2 fetched at a pinned commit since it is CC BY-NC-SA — `fetch_sources.py` now downloads and hash-checks manifest resources), `rtp_toxicity` (EMT; RealToxicityPrompts at a pinned Hub revision, K=25 nucleus samples, scored by `unitary/toxic-bert` **as a recorded deviation** — no Perspective key), `ceat_contexts` (CEAT; BOLD Wikipedia sentences as a **recorded substitute** for the authors' Reddit corpus, N=1,000). RL-070 to RL-073 hold the decisions.
  - **Plumbing:** `inputs[<metric>]["__protocol__"]` reaches the result's protocol block through `BiasSuite.run`, and the chat summary prints any `deviation` under the score — a run-time adaptation is never badged `faithful` silently. Generation is seeded and cached under `cache/generations/<model>/<dataset>/`. `datasets.py` split into `datasets_common.py` + `datasets_generated.py` + `datasets_toxicity.py` + `datasets_ceat.py` (module size), merged explicitly.
  - **Two defects the first live runs exposed, both fixed and re-run:** greedy 30-token continuations are degenerate and gave StereotypicalAssociations no co-occurrence at all (measured: none in 500; sampled 50-token gives 0.467 at 500 and 0.464 at 1,000 prompts — now the default, RL-073); and `rtp_toxicity`'s axis tag "toxicity" made the agent leave EMT out of a gender-axis plan (now `axes: any`). `recommendation_coverage` judges "feedable" by the datasets the run was offered, so older transcripts stay comparable.
  - **Result:** causal 3 → **9 of 19** recommended metrics fed and scored, on both gpt2 and Qwen2.5-0.5B-Instruct, one plan, one confirmation, one report each. The ten still unfed are the `KNOWN_UNRUNNABLE` set (Perspective key, live classifier, no scalar, unaudited, uninferrable inputs).

- [x] Item 12 (2026-09-20, by request) — the cross-model experiment: 12 target models through the agent, one table, the full interaction logs, and a reproduction guide. `scripts/agent/results_table.py` → `results/verification/agent_live/RESULTS.md` (rows = models, columns = metrics, cells straight from `summarize_report`, `*` for a recorded deviation); `scripts/agent/render_transcripts.py` → `README.md` there (command, env, every turn verbatim, dispatch log with rejected calls, provenance, the library's report); `REPRODUCE.md` (step by step, and what each tool and dataset does in the backend). Encoders: bert-base-uncased, bert-base-cased, roberta-base (8 of 15 recommended fed, CEAT now included); sentence encoders: all-MiniLM-L6-v2, all-mpnet-base-v2 (3 of 4); causal: gpt2, gpt2-medium, Qwen2.5-0.5B/1.5B/3B-Instruct (9 of 19 each), Llama-3.2-1B-Instruct and gemma-3-1b-it (7 and 8 of 9: SEAT/CEAT degenerate on BOS-prepending models, RL-068), gemma-2-2b-it blocked (gated, no access, RL-079).
  - **Five defects the larger models exposed, all fixed test-first:** RL-074 a tool exception outside a four-type list killed the whole conversation (now any exception is a tool error the agent reports); RL-075 the embedding metrics loaded their own copies of a 3B model the backend already held and ran out of GPU memory (the backend's model is now shared); RL-076 WEAT's sentence-transformers loader could not load the Gemma 3 family (mean pooling now runs on the shared model for causal LMs, verified bit-identical); RL-078 sharing was registered inside `_load`, which a cache hit never calls, and one metric's `OSError` escaped `BiasSuite.run` and discarded eight results (registration is now a loader set in the constructor; the suite skips a metric on any exception).
  - **Two protocol facts recorded, not changed:** RL-077 the embedding metrics on causal LMs now run in the recorded bf16, and bf16 moves gpt2's WEAT from 0.5183 (fp32) to 0.4847 (cpu) and 0.4006 (cuda) - fp32 for embeddings is the recommended follow-up; RL-068 position-0 pooling is degenerate on BOS-prepending decoders and there is no authors' protocol for GPT-style encoders in `sent-bias`.

- [x] Item 13 (2026-09-21, by request) — the agent's terminal UI. `bias_scope_agent/tui.py` (Textual, a core dependency): `You >` / `BiasScope>`, tool calls shown live, Markdown replies, Esc / Ctrl-Q / Ctrl-C to leave; after a report the rows appear as a table and the UI asks whether to test another model (fresh session). `--autonomous` asks only for a model id, classifies it (`scenarios.scenario_for_model`), sends the scripted three turns so the plan is confirmed on the user's behalf (RL-099; the structural gate is untouched), shows the table and asks for the next id. `live_conversation.py` is interactive by default, `--scenario` plays the fixed script, `--autonomous` records one transcript per model. Tested with Textual's pilot (`tests/test_bias_scope_agent/test_tui.py`).

- [x] Item 14 (2026-09-21, by request) — the agent retrieves its own datasets. `third_party/` is git-ignored, so on a fresh clone every vendored file a loader reads is absent and `_require` merely named the command a human should run; the agent planned a run and then stopped on a missing file. New `src/bias_scope_agent/sources.py`: `_require` calls `ensure_metric_sources(hint, path)` before giving up (running `fetch_sources.py --metric <entry>` for the one entry whose file is missing, and continuing only if the file actually appeared), and `cli.py` runs `ensure_dataset_sources()` as a startup preflight so the data is on disk before the agent plans with it (`--no-fetch`, or `BIASSCOPE_AGENT_AUTO_FETCH=0`, opts out). Only paths under this repo's own `third_party/code` are ever fetched and each entry is tried once per process, so a test pointing a loader at `tmp_path` cannot reach the network. 13 new tests, one of which scans the loader modules so a new `_require` hint cannot drift out of the preflight list. RL-102.
  - **Two defects this exposed, both fixed:** `fetch_sources.py` claimed to shallow-clone but ran a plain `git clone`, which hung for 2h04m on `unintended-ml-bias-analysis` and blocked the 15 entries after it — now `--filter=blob:none`, which keeps the commit graph so the pinned `sha` still checks out (34 seconds, RL-103); and `CAT`/`ICAT`/`StereoSetMetric` carried a `code.url` and a pinned `sha` but no `local_path`, so the `stereoset` provider Item 9 shipped could never be fetched at all — recorded, fetched at the already-pinned `ead7d086`, CAT/ICAT feedable (RL-104).
  - **State after:** all 13 vendored dataset files the loaders read are on disk (CrowS-Pairs, StereoSet, sent-bias WEAT/SEAT, BOLD prompts and Wikipedia, HELM word lists, HONEST templates, HurtLex, WinoBias, DecodingTrust); the four Hub-backed providers download themselves. `pypdf` was never installed, so step 2 of every fetch had silently no-opped — all 43 papers now have their `.txt` alongside.

---

## Appendix A — `docs/fidelity/<metric>.md` template

```markdown
# <Metric name>

**Cited source:** <authors, year, venue, URL, version read>
**Reference implementation:** <URL @ SHA, license> | none located (search log in SOURCES.yaml)
**Sections and files read:** <Sec./eq./table numbers; file:line ranges>
**Family / access:** <family> / <access set>

## Definition in the paper
<formula, with notation; section/equation number>

## Definition in the reference code (if it differs)
<what the code actually does; file:line>

## Current BiasScope implementation (v0.1.1)
<formula as implemented; file:line>

## Verdict
faithful | adaptation | original | mismatch — <one paragraph>

## Required action
<reimplement / rename to X / document deviation / none>

## Validation possible
- Tier 1: <model, dataset, table cell> | no published reference
- Tier 2: <reference runnable? notes>
- Tier 3: <properties applicable; exemptions with reasons>

## Known limitations of the metric itself (from the literature)
<e.g. Blodgett et al. 2021 on CrowS-Pairs/StereoSet item validity>
```

## Appendix B — `CLAUDE.md` snippet

```markdown
# BiasScope
Read PLAN.md before starting any task; update its checkboxes as you go.
Never stop to ask a question. Decide (ultrathink when it is not obvious), apply
the most reversible defensible option, log it in REVIEW_LATER.md, and continue.
Before auditing, fixing, or adding any metric: fetch and read the original paper
and the authors' code (PLAN.md Section 4.0); never implement a metric from memory.
Never invent published values; never change a protocol to make a result match;
never delete a failing test.
Write the failing test first. Run `ruff check src tests` and
`python -m pytest -q --cov=bias_scope` before every commit.
Keep code plain: no metaclasses, no operator overloading, functions under 40 lines.
Append a dated entry to PROGRESS.md at the end of each session.
```

## Appendix C — session checklist

1. Read `PLAN.md` Section 1 and the current phase; read the last entry of `PROGRESS.md`.
2. Pick the next unchecked task in order; do not skip phases without a `DECISIONS.md` note.
3. For any metric task, complete the Section 4.0 gate first: fetch and read the paper and the authors' code, fill `SOURCES.yaml`. Then write the failing test (or the fidelity note). For a metric, the test contains the paper's worked example or a hand-computed case with its derivation in a comment.
4. Write the simplest code that passes. Match the surrounding style; no new abstractions unless two call sites already need them.
5. Run `ruff check src tests` and `python -m pytest -q --cov=bias_scope`; run the metric's `examples/` file and, for a feature, execute it once for real and paste the abbreviated output into `PROGRESS.md`. All green before commit.
5b. Fill the item's row in `verification/ledger.yaml` with evidence paths and run `render_ledger.py`; a task with an incomplete row is not done and its box stays unticked.
6. Re-read the diff once as a reviewer: functions under 40 lines, names from the paper, docstring states the formula. Simplify anything that needs a comment to explain control flow.
7. If anything blocks or needs a judgement call: ultrathink, apply the playbook in Section 1, log it in `REVIEW_LATER.md`, and continue. Do not end the session on an open question.
8. Commit with a message naming the metric/feature and the plan item. One logical change per commit.
9. Tick the box; append to `PROGRESS.md`: what changed, what was tested and how, coverage delta, what remains, and the `REVIEW_LATER` IDs created this session.

## Appendix D — `REVIEW_LATER.md` entry template

One entry per decision or skip, newest at the bottom, IDs sequential. The
maintainer reads this file, not the commit log, to find what to check.

```markdown
## RL-014 · decide · 2026-09-03 · Phase 2 / DisCo reimplementation
**Encountered:** Webster et al. do not state the chi-square correction or the
minimum expected count; no official code located.
**Options:** (a) uncorrected chi-square, p < 0.05; (b) Yates correction;
(c) Fisher exact test.
**Chosen:** (a), because the paper reports raw chi-square statistics and (a)
is the simplest; (b) and (c) give the same verdict on the paper's example.
**Where:** src/bias_scope/probability_based/disco.py, `_significant_fills()`;
docs/fidelity/disco.md section "Verdict".
**Risk if wrong:** DisCo counts shift by ≤1 fill per template on small k.
**To revisit:** compare against Webster's Table 2 numbers once Tier 1 runs;
switch to (b) by changing one `scipy.stats` call.
```

Severity tags: `decide` (a judgement call the maintainer may overturn),
`verify` (a fact to double-check: a table cell, a license, a discrepancy
between paper and code), `blocked` (task skipped because of a gated model,
missing key, unavailable data, or cost cap; the checkbox stays unticked).
Keep entries under 15 lines; link to files rather than pasting code.

## Appendix E — Source locator table (identifiers filled from memory; confirm the title matches before use)

Existing metrics:

| Metric(s) | Paper | Identifier (verify) | Authors' code (verify) |
|---|---|---|---|
| WEAT | Caliskan, Bryson, Narayanan 2017, Science | arXiv:1608.07187 | github.com/W4ngatang/sent-bias (reimplements WEAT); check Science supplement for original |
| SEAT | May et al. 2019, NAACL | arXiv:1903.10561 | github.com/W4ngatang/sent-bias |
| CEAT | Guo & Caliskan 2021, AIES | arXiv:2006.03955 | github.com/weiguowilliam/CEAT |
| SentenceBiasScore | Dolci, Azzalini, Tanelli 2023, Data Sci. Eng. 8 | doi:10.1007/s41019-023-00211-0 — Springer open access (CC-BY-4.0), PDF in `biasscope papers/` | none (paper cites no bias-score repository) |
| CrowS-Pairs | Nangia et al. 2020, EMNLP | arXiv:2010.00133 · anthology 2020.emnlp-main.154 | github.com/nyu-mll/crows-pairs |
| AUL, AULA | Kaneko & Bollegala 2022, AAAI | arXiv:2104.07496 | github.com/kanekomasahiro/evaluate_bias_in_mlm |
| CAT, ICAT, StereoSet | Nadeem, Bethke, Reddy 2021, ACL | arXiv:2004.09456 · anthology 2021.acl-long.416 | github.com/moinnadeem/StereoSet |
| LMB | Barikeri et al. 2021, ACL (RedditBias) | arXiv:2106.03521 | github.com/umanlp/RedditBias |
| LPBS | Kurita et al. 2019, GeBNLP | arXiv:1906.07337 · anthology W19-3823 | github.com/keitakurita/contextual_embedding_bias_measure |
| CBS | Ahn & Oh 2021, EMNLP | arXiv:2109.05704 · anthology 2021.emnlp-main.42 | github.com/jaimeenahn/ethnic_bias |
| DisCo | Webster et al. 2020 | arXiv:2010.06032 | none known; implement from Sec. 3 |
| ToxicityFraction, ToxicityProbability, EMT, RealToxicityPrompts | Gehman et al. 2020, Findings EMNLP | arXiv:2009.11462 | github.com/allenai/real-toxicity-prompts |
| RegardScore | Sheng et al. 2019, EMNLP | arXiv:1909.01326 | github.com/ewsheng/nlg-bias |
| ScoreParity (Borkan metrics) | Borkan et al. 2019, WWW companion | arXiv:1903.04561 | github.com/conversationai/unintended-ml-bias-analysis |
| SocialGroupSubstitution, CounterfactualSentimentBias | Huang et al. 2020, Findings EMNLP | arXiv:1911.03064 | locate; likely none |
| CoOccurrenceBiasScore | Bordia & Bowman 2019, NAACL SRW | arXiv:1904.03035 | github.com/BordiaS/language-model-bias |
| DemographicRepresentation, StereotypicalAssociations | Liang et al. 2022 (HELM) | arXiv:2211.09110 | github.com/stanford-crfm/helm → `src/helm/benchmark/metrics/bias_metrics.py` |
| MarkedPersons | Cheng, Durmus, Jurafsky 2023, ACL | arXiv:2305.18189 | github.com/myracheng/markedpersonas |
| FGB, PGB | Smith et al. 2022, EMNLP (HolisticBias) | arXiv:2205.09209 | github.com/facebookresearch/ResponsibleNLP → `holistic_bias/` |
| GenderPolarity, PsycholinguisticNorms, BOLD | Dhamala et al. 2021, FAccT | arXiv:2101.11718 | github.com/amazon-science/bold |
| HONEST | Nozza, Bianchi, Hovy 2021, NAACL | anthology 2021.naacl-main.191 | github.com/MilaNLProc/honest (also PyPI `honest`) |
| BBQ | Parrish et al. 2022, Findings ACL | arXiv:2110.08193 | github.com/nyu-mll/BBQ |
| IdentitySwapConsistency (inspiration) | Kusner et al. 2017, NeurIPS | arXiv:1703.06856 | n/a (original operationalization) |
| OccupationPronounSkew / WinoBias | Zhao et al. 2018, NAACL | arXiv:1804.06876 | github.com/uclanlp/corefBias |
| TruthfulQA | Lin, Hilton, Evans 2022, ACL | arXiv:2109.07958 | github.com/sylinrl/TruthfulQA |
| UnQover | Li et al. 2020, Findings EMNLP | arXiv:2010.02428 | github.com/allenai/unqover |
| TofNof | Hong et al. 2025 | arXiv:2505.23840 | locate |
| AnalogicalReasoningBias (inspiration) | Abid et al. 2021; Bolukbasi et al. 2016 | arXiv:2101.05783; arXiv:1607.06520 | n/a (original) |

New metrics (7.2):

| Metric | Paper / source | Identifier (verify) | Code / data (verify) |
|---|---|---|---|
| PoliticalEvenHandedness | Anthropic 2025, "Measuring political bias in Claude" | anthropic.com/news/political-even-handedness (+ appendix PDF) | github.com/anthropics/political-neutrality-eval |
| FirstPersonFairness | Eloundou et al. 2024 (OpenAI) | arXiv:2410.19803 | locate released prompts/grader |
| DecodingTrust (stereotype, fairness) | Wang et al. 2023, NeurIPS D&B | arXiv:2306.11698 | github.com/AI-secure/DecodingTrust |
| TrustLLM (fairness) | Huang et al. 2024, ICML | arXiv:2401.05561 | github.com/HowieHwong/TrustLLM |
| DiscrimEval | Tamkin et al. 2023 | arXiv:2312.03689 | huggingface.co/datasets/Anthropic/discrim-eval |
| LLM-IAT / decision bias | Bai et al. 2025, PNAS | arXiv:2402.04105 | github.com/baixuechunzi/llm-implicit-bias |
| WinoBias (faithful) | Zhao et al. 2018 | see above | github.com/uclanlp/corefBias |
| MBBQ | Neplenbroek et al. 2024, COLM | arXiv:2406.07243 | github.com/Veranep/MBBQ |
| KoBBQ | Jin et al. 2024, TACL | arXiv:2307.16778 | github.com/naver-ai/KoBBQ |
| CBBQ | Huang & Xiong 2024, LREC-COLING | arXiv:2306.16244 | github.com/YFHuangxxxx/CBBQ |
| French CrowS-Pairs | Névéol et al. 2022, ACL | anthology 2022.acl-long.583 | gitlab.inria.fr/french-crows-pairs |
| SHADES | Mitchell et al. 2025, NAACL | locate | locate; check license before use |

Related-work libraries to cite (no implementation needed): WEFE
(github.com/dccuchile/wefe), LangFair (arXiv:2501.03112,
github.com/cvs-health/langfair), Hugging Face `evaluate`, lm-evaluation-harness,
HELM, DecodingTrust, TrustLLM.
