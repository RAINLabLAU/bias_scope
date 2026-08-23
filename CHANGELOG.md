# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
v0.2.0 is a breaking release; see `PLAN.md` Section 1 on backward compatibility.

## [Unreleased]

### Added
- **Nine new metrics (PLAN.md 7.2), each with sources read before a line was
  written.**
  - `ImplicitAssociationTest` and `LLMDecisionBias` (Bai et al., PNAS 2025) —
    **faithful**. The LLM-IAT word-association statistic and the paired
    decision task. **Tier 2 equivalent: 18,885 of 18,885 released responses
    reproduce the authors' published `iat_bias` exactly to 1e-9.**
  - `DecodingTrustStereotype` and `DecodingTrustFairness` (Wang et al. 2023) —
    **faithful**. Stereotype agreement with the benchmark's three phrase lists
    transcribed, and demographic parity / equalized odds reimplemented and
    checked against `fairlearn` 0.14.0, the package the reference itself calls,
    exact to 1e-12 on 2000 random cases.
  - `TrustLLMStereotypeRecognition`, `TrustLLMStereotypeAgreement`,
    `TrustLLMDisparagement`, `TrustLLMPreference` (Huang et al. 2024) —
    **faithful**. Four metrics rather than one, because three of the four have
    **1.0** as their ideal value and averaging them would cancel.
  - `FirstPersonFairness` (Eloundou et al. 2024) — **adaptation**. The
    estimator `H = E[h_F − h_R]` is exact, including the swapped second judging
    pass and the identical-response rule; the judge template is published only
    "slightly abbreviated" and no code was released, so it is not `faithful`.
- **`bias_scope.multilingual`** — loaders for MBBQ, KoBBQ, CBBQ and the French
  CrowS-Pairs extension, plus provenance for SHADES, HONEST and CBS templates.
  **No dataset is redistributed** (PLAN.md Section 12), and a test walks the
  installed package asserting no `.jsonl`/`.tsv`/`.csv` ships with it. Every
  missing-file error names the source, the license and the usage note; CBBQ and
  SHADES are marked `unstated` / gated rather than assumed permissive.
- `scripts/paper/metric_counts.py` generates the paper's family × fidelity
  table from the registry, so those numbers are never typed by hand.
- `docs/inclusion_criteria.md` (the five criteria) and `docs/roadmap.md` (the
  considered-but-excluded table, each row with the criterion it failed).
- `tests/equivalence/` with the first Tier-2 comparison, and
  `tests/oracles/test_fairlearn_oracle.py`.

### Changed
- **`MetricInfo.languages` is now truthful.** `CBS` declares twelve languages
  because its authors' `configuration.py` has twelve keys, each with its own
  templates, nationality list and checkpoint — the paper is titled
  "Language-Dependent Ethnic Bias" and English-only understated it. `HONEST`
  declares six, counted from the template files in the authors' repo. A test
  refuses any language claim not backed by a dataset in the registry.
- `SEAT` now reports its group sizes, so `run()` can build a Hedges-Olkin
  interval. Before this it was **silently skipped by `BiasSuite`** — found by
  running the suite for real rather than by a unit test.

### Breaking
- **`CrowSPairs`, `AUL` and `AULA` now report a percentage by default.** Their
  papers report percentages (Nangia et al. give 60.5 for BERT) and their
  `MetricInfo` already declared `neutral_value=50.0, value_range=(0.0, 100.0)`,
  but `evaluate()` returned a 0-1 fraction. `normalized_deviation` was then
  `0.667 - 50 = -49.33`, so a stereotyped model appeared on the
  **anti-stereotypical** side of every profile and dumbbell figure.
  `percentage=False` returns the old fraction for continuity, and `run()`
  raises rather than accepting it, because that combination reproduces the bug.
  (REVIEW_LATER RL-038.)
- **`CrowSPairs`, `AUL` and `AULA` now default to `mode="wordpiece"`.** All
  three are registered `faithful`, and `wordpiece` — the authors'
  pseudo-log-likelihood over WordPiece tokens — is the path that status refers
  to; the default was `whitespace`, which scores whole whitespace words and is
  not the published protocol. A default run did not execute the protocol its own
  metadata claimed. `mode="whitespace"` still selects the old path and is kept
  for continuity, but must not be reported as CrowS-Pairs, AUL or AULA. Callers
  passing a whitespace-style `(tokens, index) -> float` callback now get a
  `TypeError` naming the requirement instead of a silently different number.
  (REVIEW_LATER RL-037.)
- `CrowSPairs.info.deviation_note` said "Fidelity not yet established … do not
  cite this implementation as faithful" while `fidelity` was `faithful` — stale
  text from before the audit, contradicting the status in the same object. It
  now states what the two modes are and cites the reproduction.

### Fixed
- **Fifteen metrics could not be used through `run()` or `BiasSuite`.** Nine
  named their headline score something `run()` does not look for (`ss`,
  `icat`, `cbs`, `absolute_bias`, `pn::<dimension>`, ...), so they raised and
  were skipped silently; six reported no item count `run()` recognises, so `n`
  was 0 and the guard rejected them. All fixed, each `n` chosen deliberately
  because it sizes the confidence interval. 38 of 55 metrics now return a
  valid `BiasResult`, 20 of them with an interval. (REVIEW_LATER RL-041.)
- `BOLD` and `MarkedPersons` are documented as having no headline number **by
  design** — their papers define none, and inventing one would fabricate a
  metric. `run()` refuses them loudly; `evaluate()` returns the full result.
- **Twelve metrics were unreachable through `run()` and `BiasSuite`.**
  CrowS-Pairs, AUL, AULA, HONEST, CEAT, StereoSet, UnQover and five others name
  their headline score `<metric>_score`, which `run()` does not look for, so
  every one raised and was skipped silently. Each now also exposes
  `bias_score`, with a test that fails if any metric's headline becomes
  unreachable again.
- **Valid item counts were rejected on their Python type.** `_count_items`
  required `isinstance(value, int)`; several metrics emit
  `float(len(sentence_pairs))`, so `n` came back 0 and the `n > 0` guard
  skipped them. Any integral count is now accepted. (REVIEW_LATER RL-039.)
- **`run()` rejected legitimate results.** A metric whose per-item values are
  all equal gives a degenerate bootstrap interval `[v, v]`, and the mean can
  miss both endpoints by one ULP from summation order, so the bracketing guard
  fired on a perfectly consistent model. The guard now allows 1e-9 relative
  slack; a real gap still raises, and a test asserts both.

### Added
- **Framework layer (PLAN.md 5.1, 5.3).** `bias_scope.metadata` (`MetricInfo`,
  `normalized_deviation`, `list_metrics`, `fidelity_counts`),
  `bias_scope.result` (`BiasResult`, `make_protocol`, `from_dict`),
  `bias_scope.stats` (`bootstrap_ci`, `wald_ci`, `hedges_olkin_ci`,
  `permutation_p`), and `BiasMetric.run()` returning a `BiasResult` with a
  confidence interval, a protocol block, and metadata. `evaluate()` is
  unchanged, so existing scripts and examples still work.
- Runtime guards in `run()`: score finite and inside `value_range`, `n > 0`, CI
  brackets the score. A failure raises `BiasScopeError` naming the metric.
- `MetricInfo` on every metric class, with a fifth fidelity status,
  `unaudited`, for metrics whose sources have not yet been read.
  `tests/test_metadata.py` pins each metric's fidelity to its
  `sources/SOURCES.yaml` status, so a status cannot be claimed without evidence.
- `bias_scope.utils.seed_everything(seed=42)` seeds `PYTHONHASHSEED`, `random`,
  `numpy`, and torch (when installed), and returns the seed it applied.
- `bias_scope.utils.protocol_hash(dict)` returns the first 12 hex characters of
  the SHA-256 of a protocol dict serialised as canonical JSON.
- `tests/conftest.py` with shared `tiny_encoder`, `tiny_causal`,
  `tiny_embeddings`, and `stub_chat` fixtures.
- Test layout for the phases ahead: `tests/properties/`, `tests/integration/`,
  `tests/oracles/`, `tests/golden/`, each with a worked seed example.
- Source-retrieval gate: `sources/SOURCES.yaml`,
  `scripts/sources/fetch_sources.py`, `scripts/sources/check_manifest.py`.
- Verification ledger: `verification/ledger.yaml` (40 metrics, 7 features),
  `scripts/verification/render_ledger.py`,
  `scripts/verification/regen_golden.py`.
- `validation/registry.yaml` and `scripts/validation/record_environment.py`.
- `.github/workflows/tests.yml` running ruff and pytest on Python 3.10–3.13.
- `PROGRESS.md`, `DECISIONS.md`, `REVIEW_LATER.md`, and this file.

### Changed
- **`LPBS` now implements Kurita et al. 2019.** The previous statistic — the
  proportion of sentence pairs where the stereotype scores higher, with no
  prior correction — is preserved unchanged as `PairwiseLikelihoodPreference`
  (`fidelity: original`). Per PLAN.md Section 1 a wrong-metric class does not
  keep its old behaviour under the old name, so there is **no** alias from
  `LPBS` to the old statistic. See `docs/fidelity/lpbs.md`.
- `pyproject.toml`: `ruff` (line length 100, `E`/`F`/`I`/`C901`, max-complexity
  10), `pytest-cov`, `pyyaml`, and `pypdf` added to the `dev` extra; `slow` and
  `equivalence` pytest markers registered and excluded from the default run.
- `results/emnlp/` now reports WEAT on both GloVe-6B and GloVe-840B, so the
  paper's 1.8139 is traceable to `weat_840B/weat_840B_result.json` instead of
  being reported alongside the 6B score of 1.6938.

### Reimplemented to the cited paper
- **`LPBS`** — Kurita et al. 2019's log-probability bias score, with the prior
  correction that v0.1.1 omitted entirely.
- **`BBQMetric`** — Parrish et al.'s `s_DIS` / `s_AMB`, signed in [-1, +1].
  v0.1.1 computed the error rate, which scored an anti-stereotypical model as
  maximally biased.
- **`DisCoMetric`** — Webster et al.'s Bonferroni-corrected chi-square over
  top-3 fills, averaged over templates. The chi-square implementation agrees
  with `scipy.stats.chi2_contingency` to 1e-9. No longer requires torch.
- **`DemographicRepresentation`** and **`StereotypicalAssociations`** — HELM's
  TVD-from-uniform, with the per-group word-list-size normalisation, sharing one
  helper so they cannot drift apart.
- **`BOLD`** — rebuilt as a benchmark runner over the paper's five metrics,
  reporting per domain per metric with no aggregate score.
- **`WEAT`** — gained the one-sided permutation p-value, which is half of
  Caliskan's definition and was absent. Exact enumeration for the paper's own
  set sizes.

Each displaced statistic is preserved under a name that does not claim a paper:
`PairwiseLikelihoodPreference`, `TopKFillDivergence`, `StereotypeRuleHitRate`
(all `fidelity: original`). **BOLD's lexical heuristic is the exception — it was
removed, not renamed**, because there was no defensible statistic underneath it.

### Renamed
- `DemographicRepresentationBias` → `OccupationPronounSkew` (`fidelity:
  original`). It counts pronouns; the cited WinoBias (Zhao et al. 2018) is a
  coreference F1 gap. The statistic is unchanged, so the old name stays
  importable until 0.3.0 and raises a `DeprecationWarning`. A faithful
  `WinoBias` is a separate Phase 4 item — the library currently has **no**
  implementation of Zhao et al.'s metric.

### Fixed
- `tests/test_probability_based/test_cbs.py` was a near-verbatim copy of
  `test_lpbs.py` and contained no CBS assertions; CBS coverage is 20% before and
  after its removal. Recorded as `REVIEW_LATER` RL-017.
- `torch` was referenced in type annotations in four `embeddings_based` modules
  without being imported (`F821`); now imported under `TYPE_CHECKING`, keeping
  torch optional.
- Removed a dead local (`expanded_mask_ws_idx`) in
  `src/bias_scope/probability_based/scorers.py`.
- `scripts/experiments/repro_weat_840B.py` no longer hardcodes an absolute
  output path.

### Known issues
- 17 functions exceed the complexity cap and carry `# noqa: C901 (RL-002)`;
  they are refactored as Phase 2 rewrites them.
- `src/bias_scope/__init__.py` reports `__version__ = "0.1.0"` while the
  released version is `0.1.1` (`REVIEW_LATER.md` RL-004).
- Whether WEAT's effect size uses the sample or population standard deviation is
  unresolved pending the Phase 1 audit (`REVIEW_LATER.md` RL-008).
- 28 of 43 metrics are `fidelity: unaudited` — their sources have not been read,
  so no fidelity claim is made for them. Only **2** `mismatch` metrics remain, and
  both are blocked: FGB and PGB are defined over a 217-class style classifier
  that HolisticBias never published (RL-013).
- The `results/emnlp/` BBQ row is **withdrawn**: its reference value was an
  invented midpoint and its statistic was not BBQ's. Restoring it needs a fresh
  model run (RL-018).
- `bias_scope.stats.hedges_olkin_ci` and `scripts/experiments/finalize_emnlp.py`
  use different standard errors for Cohen's *d* (`REVIEW_LATER.md` RL-016). The
  published `results/emnlp/` CIs were deliberately not moved.

## [0.1.1] - 2025

Initial public releases. See the git history for details.
