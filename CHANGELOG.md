# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
v0.2.0 is a breaking release; see `PLAN.md` Section 1 on backward compatibility.

## [Unreleased]

### Added
- **Results table, rerun question and an autonomous mode in the terminal UI.**
  After every report the UI (`bias-scope-agent`, and the runner's UI) shows
  the rows as a table (metric, family, score, n, fidelity, deviation) and asks
  whether to test another model; `yes` starts a fresh session. `--autonomous`
  asks only for a model id, works out the kind of model, sends the agent the
  scripted three turns (the plan is confirmed on the user's behalf, `REVIEW_LATER`
  RL-099), shows the table and asks for the next id.
  `scripts/agent/live_conversation.py` is interactive by default; `--scenario`
  plays the fixed script and `--autonomous` writes one transcript per model.
  Esc, Ctrl-Q and Ctrl-C leave the UI; `textual` is a core dependency.
- **Prompt-family providers for API-served targets** (`datasets_prompt.py`):
  `prompt_benchmarks` (BBQMetric, StereoSetMetric, IdentitySwapConsistency,
  OccupationPronounSkew - self-loading, given the model name and a bounded
  size), `winobias_coref` (WinoBias from the authors' type-1 files, answered
  by the backend), `decodingtrust_stereotype` (benign scenario),
  `rtp_prompt_runner` (RealToxicityPrompts with the local toxicity scorer as
  a recorded deviation). CoOccurrenceBiasScore is fed by the HELM provider.
  An OpenRouter target goes from 6 to 14 feedable metrics (`REVIEW_LATER`
  RL-098 lists what is still deferred and why). `LiteLLMBackend` translates
  transformers decoding names to chat-API names; the scripted runner has an
  `api` scenario; provenance records `access_mode` (RL-097).
- **Six dataset providers for the agent**, so a causal LM's recommended
  metrics can be fed without anyone pasting data: `bold_gender_polarity`
  (GenderPolarity), `bold_helm_bias` (DemographicRepresentation,
  StereotypicalAssociations, HELM's own word lists), `honest` (HONEST, HurtLex
  EN 1.2 fetched at a pinned commit), `rtp_toxicity` (EMT, RealToxicityPrompts
  at a pinned Hub revision, scored by `unitary/toxic-bert` as a recorded
  deviation from the Perspective API) and `ceat_contexts` (CEAT, BOLD
  Wikipedia sentences as a recorded substitute for the authors' Reddit
  corpus). A causal LM goes from 3 to 9 feedable metrics of 19 recommended
  (`REVIEW_LATER` RL-070 to RL-073).
- `inputs[<metric>]["__protocol__"]` in `BiasSuite.run` reaches the result's
  protocol block, and the agent's chat summary prints any `deviation` a
  provider recorded there under the score.
- `scripts/sources/fetch_sources.py` downloads and hash-checks manifest
  `resources` that carry a `url` and `local_path`.
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
- **`SentenceBiasScore().run()` no longer crashes; the paper (Dolci et al.
  2023) was found and re-audited.** `run()` raised `BiasScopeError`
  unconditionally on every input, because `evaluate(..., return_details=True)`'s
  keys (`female_bias`, `male_bias`, `absolute_bias`, `num_words`) matched none
  of `_split_result`'s recognised headline-score names. The dict now also
  carries `bias_score` (= `absolute_bias`, the paper's Eq. 3, offered because
  it is "useful ... when sorting multiple sentences"), a `breakdown` of
  `female_bias`/`male_bias`, and an integer `n`. `MetricInfo` updated to
  match (`direction="higher_more_biased"`, `value_range=(0, inf)`,
  `fidelity="adaptation"`). New module-level `derive_gender_direction()` and
  `derive_word_importance()` implement the paper's PCA gender-direction and
  max-pooling word-importance procedures (previously entirely unimplemented —
  the class only executed the final weighted sum); `build_gender_words_mask()`
  implements the paper's case-insensitive lexicon matching, though **the
  6562-word gender lexicon itself is not vendored** (not published; see
  REVIEW_LATER RL-087). The class docstring and docs cited the wrong authors
  and title; corrected. The paper had been believed paywalled with no
  preprint (RL-029); it is Springer open access and was already in the repo's
  papers folder — RL-029 closed. (2026-09 SentenceBiasScore audit.)
- **`CEAT().run()` is now reproducible and reports its own interval.** `run()`
  did not thread `seed` into `random_seed` — `evaluate()`'s own
  `random_seed=None` default falls back to OS entropy, so two `run(seed=42)`
  calls could return effect sizes of opposite sign. `run()` also attached a
  Hedges-Olkin interval built from `|X|`,`|Y|` (the target-*stimulus* counts)
  instead of CEAT's own random-effects `SE(CES)` — 13x too wide in one test
  case, and the wrong quantity regardless of width. `CEAT` now overrides
  `run()` (threads/records `random_seed`, mirroring `WEAT.run`/`SEAT.run`),
  `_interval` (`CES ± Z_95·SE(CES)`, `ci_method="random_effects"`), and
  `_count_items` (`n = n_samples`). `bias_scope.result.make_protocol` gained a
  `random_seed` field alongside `permutation_seed`. `evaluate()`'s CES and
  p-value are unaffected — they already matched the reference implementation
  exactly. (2026-09 CEAT audit.)
- **`DisCoMetric`: fixed the shipped example (still the pre-refactor API)
  and stale reproduction-registry notes.** No defect was found in the
  metric itself — a from-scratch audit independently re-verified
  `chi_square_2xk`/`chi_square_p_value` bit-exact against
  `scipy.stats.chi2_contingency` and confirmed `run()` correct by fresh
  execution. `examples/probability_based/disco.py`, however, still called
  `metric.evaluate(template=..., attr_a=..., attr_b=..., k=5)` — the old
  two-prompt symmetric-difference signature that was split out into
  `TopKFillDivergence` when `DisCoMetric` was reimplemented to the paper's
  chi-square/Bonferroni definition — and crashed with `TypeError` on the
  first call. Rewritten to use the current
  `evaluate(templates, person_words, top_k_fills, ...)` API. Also
  corrected all eight DisCo rows in `validation/registry.yaml`, whose
  notes still said the metric "computes a different statistic" (true for
  v0.1.1, false since the reimplementation). See `docs/fidelity/disco.md`.
  (2026-09-18, following the 2026-09 DisCo audit.)
- **`CBS`: `run()` no longer crashes on every call, multi-token attributes
  are whole-word-masked correctly, and the variance matches the reference's
  convention.** `evaluate()`'s dict had no `bias_score`/`score`/`value`/
  `effect_size` key, so `run()` raised `BiasScopeError` unconditionally;
  it now includes `"bias_score"` and `"per_item"` (the per-template×
  attribute variances). The prior sentence now inserts `attribute_num`
  mask tokens for the attribute, matching its real WordPiece subword count
  (Ahn & Oh 2021 §3.2's whole-word-masking adaptation applies to the
  attribute too, per the reference's `attribute_mask`), instead of always
  using exactly one — several of the paper's own 70 attributes are
  multi-token ("C.E.O." → 6 pieces, "Customer service representative" →
  3), and using one mask shifted target-mask log-probabilities by up to
  0.66 nats in one verified case. `allow_multi_token_targets=True` now
  groups target words by subword count and scores each subword against
  its own mask position (one-to-one, following the paper — see
  `REVIEW_LATER` RL-088), instead of evaluating all of a word's subword
  IDs as candidates at a single mask slot. `np.var(..., ddof=1)` (sample
  variance) replaces `ddof=0`, matching the reference's
  `pandas.Series.var()`. Also fixed the shipped example, which crashed
  immediately with `TypeError` (a stale override signature), and added
  `tests/test_probability_based/test_cbs.py` (16 tests) — there was
  previously no real CBS test coverage. See `docs/fidelity/cbs_lmb.md`.
  (2026-09-18, following the 2026-09 CBS audit.)
- **`LMB`: the default outlier rule now actually removes outliers, the
  p-value is no longer wrong for realistic sample sizes, and `run()`
  reports the paper's own statistic.** Three independent bugs, found in
  one from-scratch audit: (1) `outlier_strategy="sigma"` — the default,
  documented as "the paper's rule" — computed `[mean-3·std, mean+3·std]`
  but never applied it, silently identical to `outlier_strategy="none"`;
  now filters like the `"percentile"` branch already did. (2) The `df>30`
  p-value branch's `_normal_cdf` evaluated its erf approximation at `x`
  instead of `x/√2`, giving p-values wrong by up to 10x for any sample at
  REDDITBIAS's own scale (234–254 pairs) — verified against
  `scipy.stats.t.cdf`: `t=1.96, df=254` went from `p=0.0056` ("significant")
  to the correct `p≈0.050` ("not significant"), a false-positive
  significance flip. (3) `evaluate()`'s dict had no `bias_score` key, so
  `run()`'s headline silently fell back to `effect_size` (Cohen's *d*)
  instead of the t-value the paper actually reports; `evaluate()` now
  includes `"bias_score": t_stat`. Also corrected the class docstring and
  shipped example, which described/demonstrated a bidirectional
  (masked-LM) scoring contract — the paper measures LMB on DialoGPT, an
  autoregressive model scored causally (left-to-right only); the example
  now builds a genuine causal scorer around `microsoft/DialoGPT-small`
  (the paper's own model). See `docs/fidelity/cbs_lmb.md`. (2026-09-17,
  following the 2026-09 LMB audit.)
- **`CAT().run()` / `ICAT().run()` now report a confidence interval under
  the default `ci="bootstrap"`.** Neither exposed `details["per_item"]`, so
  `run()` silently returned no interval (same gap already fixed for
  CrowS-Pairs and AUL/AULA). `CAT.evaluate()` now includes `"per_item"`: the
  per-target-term `ss` values, the paper's own resampling unit for `ss`
  (`ss = mean(term_ss)` by definition, so a standard bootstrap over that
  list is exact). `ICAT.evaluate()` does **not** reuse CAT's `per_item` —
  `icat = combine(mean(term_lms), mean(term_ss))` is a nonlinear function of
  two paired series, so bootstrapping `ss` alone would target the wrong
  statistic and could produce an interval that fails to bracket `icat`.
  `ICAT` instead overrides `_interval` to resample target terms with their
  `(lms, ss)` pairs kept together and recompute `icat` via the same
  `combine` formula on each resample. See
  `docs/fidelity/stereoset_family.md`. (2026-09-17, following the 2026-09
  CAT audit.)
- **`CrowSPairs().run()` now reports a confidence interval, including
  `ci="wald"` (the paper's own convention for this statistic), and matches
  the reference's tie-rounding.** `evaluate()` never exposed `per_item`, so
  `ci="bootstrap"` (the `run()` default) silently returned `None`; fixed by
  adding `"per_item"` (scaled to 0/100) to both scoring modes'
  `return_details=True` dict. Separately, `base.py::BiasMetric._interval`
  checked `per_item is None` before `ci == "wald"`, making `ci="wald"`
  unreachable even though `wald_ci(score, n)` needs no item-level data —
  `_interval` now checks `ci` first and, when the metric's
  `MetricInfo.value_range` is finite, normalizes the score to a proportion
  for `wald_ci` and rescales the interval back (a no-op for `[0, 1]`-scale
  metrics, and safely inert, as before, for an unbounded metric like LPBS).
  Also: Nangia 2020's reference (`metric.py:225-226`) rounds each sentence's
  summed log-probability to 3 decimals before comparing for a win or a tie;
  BiasScope compared raw floats, so a difference under 0.001 could count as
  a stereotype "win" where the reference calls it neutral — now rounds to
  3 decimals first, matching the reference exactly. See
  `docs/fidelity/crows_pairs.md`. (2026-09-17, following the 2026-09
  CrowS-Pairs audit.)
- **`AUL().run()` / `AULA().run()` now report a confidence interval.**
  `evaluate(return_details=True)` computed the per-pair 0/1 stereotype-
  preference indicators (needed for eq. 6's percentage) but never exposed
  them under the `per_item` key `BiasMetric._interval` looks for, so
  `run(pairs, ci="bootstrap")` — the default — silently returned no interval
  for either metric, regardless of `ci=`. Both classes' `evaluate()` (both
  `mode="wordpiece"` and `mode="whitespace"`) now include `"per_item"`
  (the indicators scaled to 0/100, matching `bias_score`'s scale) in the
  `return_details=True` dict. Separately, `AUL`/`AULA`'s whitespace mode
  would silently compute masked pseudo-log-likelihood instead of AUL/AULA's
  defining **unmasked** forward pass if a caller passed a `BertPLLScorer` or
  `WordPieceBertScorer` instance directly to `.evaluate()` — both scorer
  classes' own docstrings incorrectly advertised themselves as valid for
  AUL/AULA, and the `__init__`-level guard only blocked the `model_name=` +
  `mode="whitespace"` combination, not a manually-constructed scorer. Both
  metrics now reject either scorer class in whitespace mode with a clear
  `ValueError`, and the misleading docstrings are corrected. The canonical
  `mode="wordpiece"` scoring itself was independently re-verified bit-
  identical (diff = 0.0) against a fresh clone of the authors' reference
  (`evaluate_bias_in_mlm@6b10239974a7`) and was already correct. See
  `docs/fidelity/aul_aula.md`. (2026-09 AUL/AULA audit.)
- **`SEAT` permutation p-value now follows May et al.'s Appendix A, not
  Caliskan's.** SEAT delegated the p-value to `WEAT` with its default
  `tie_policy="strict"` (Caliskan's `>`), but the SEAT paper deliberately uses
  the **non-strict `>=`** ("the more conservative non-strict inequality",
  because "the equality has positive probability" in the nonparametric
  version) and floors the sampled estimate at 1e-5. `SEAT.evaluate` now
  delegates with `tie_policy="conservative"` and `n_permutation_samples=100_000`
  by default, and exposes `tie_policy`, `n_permutation_samples` and
  `permutation_seed` so the strict convention is still selectable. Concretely:
  a maximally separated n=4 test returned `p=0.0` (impossible per the paper's
  1e-5 floor); it now returns `1/C(8,4)`. The effect size is unchanged.
  `SEAT.run` now threads and records `permutation_seed` the way `WEAT.run`
  does. `SEAT(pooling=...)` validates its argument. (2026-09 SEAT audit.)

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
- **The agent lost a whole conversation when a tool raised anything outside a
  four-type list** (a gated-repo 401 from inside `run_suite`). Any exception is
  now returned to the agent as a tool error it can report (RL-074).
- **The embedding metrics loaded their own copies of the model under
  evaluation** - three copies of a 3B model on a 20 GB GPU (RL-075) - and
  WEAT's sentence-transformers loader could not load the Gemma 3 family
  (RL-076). A backend registers a loader at construction; `pooling='cls'`
  reuses its model, and for a causal LM `pooling='mean'` computes the same
  attention-masked mean sentence-transformers would (verified bit-identical).
  Consequence, recorded as RL-077: causal-LM embedding numbers are now bf16.
- **One metric's `OSError` escaped `BiasSuite.run` and discarded every other
  result of the run.** Any exception from one metric is now that metric's
  skip reason; `on_error="raise"` still raises (RL-078).
- **An encoder whose config claims a masked-LM head the checkpoint does not
  ship was scored with a random head.** `sentence-transformers/all-mpnet-base-v2`
  lists `MPNetForMaskedLM` but has no `lm_head.*` tensors; the RL-058 check
  read only the config, so five probability metrics were recommended, run and
  badged `faithful` at chance level (AULA exactly 50.00). `HuggingFaceBackend`
  now also loads the masked-LM model once and withholds `logits` when any head
  weight is missing (`REVIEW_LATER` RL-066).
- **WEAT and SEAT failed on GPT-2-style tokenizers.** `pooling='cls'` batched
  texts of unequal length through a tokenizer with no pad token and raised
  "Asking to pad but the tokenizer does not have a padding token"; both
  metrics were skipped on `gpt2` in a live agent run. The encoder now pads with
  end-of-sequence, the choice `HuggingFaceBackend.generate` already made
  (RL-067).
- `scripts/agent/live_conversation.py` records `recommendation_coverage`
  (recommended vs feedable vs planned vs run vs scored) and
  `summarize_runs.py --check` tabulates it for every recorded run, so a
  recommended metric the agent quietly left out is a listed gap rather than
  an unnoticed omission. The Unicode minus in an agent's prose is normalised
  before figures are traced to tool results.
- **Fifteen metrics could not be used through `run()` or `BiasSuite`.** Nine
  named their headline score something `run()` does not look for (`ss`,
  `icat`, `cbs`, `absolute_bias`, `pn::<dimension>`, ...), so they raised and
  were skipped silently; six reported no item count `run()` recognises, so `n`
  was 0 and the guard rejected them. All fixed, each `n` chosen deliberately
  because it sizes the confidence interval. 38 of 55 metrics now return a
  valid `BiasResult`, 20 of them with an interval. (REVIEW_LATER RL-083; merged from the August `v0.2-metrics-and-framework` commits, whose percent-scale and `bias_score` changes were superseded by RL-060/RL-061 above.)
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
  skipped them. Any integral count is now accepted. (REVIEW_LATER RL-081.)
- **`ScoreParity`/`MeanScoreGap`: `NaN` std at n=1, and a narrower
  duplicated validator.** `group_a_std`/`group_b_std` returned `NaN` (with
  unguarded `RuntimeWarning`s) for a single-text group, since
  `np.std(..., ddof=1)` divides by zero at n=1 — now `0.0`. A local
  `_validate_classifier_scores` override shadowed the inherited (more
  permissive) one, rejecting legitimate `numpy.float32` classifier output —
  removed in favor of the inherited validator.
- **`PsycholinguisticNorms` computed the wrong aggregation entirely.** It
  used a plain mean of word-level lexicon scores; Dhamala et al. 2021
  §4.4 define a magnitude-weighted signed aggregation,
  `Σsgn(w)w² / Σ|w|` (identical in form to the paper's own Gender-Wavg),
  which lets a single strongly-valenced word dominate several near-neutral
  filler words by design. Counterexample: `1.025` (plain mean) vs. `3.723`
  (paper's formula) on the same input — a 3.6x divergence. Also added
  function-word (pronoun/preposition/conjunction) exclusion per the paper,
  and fixed `run()` (no `bias_score`/`n`-like key previously existed).
  Corrected `docs/fidelity/bold_metrics.md` and `_metric_info.py`, which
  had both incorrectly claimed "the aggregation matches." Logged as
  `REVIEW_LATER` RL-092.
- **`CounterfactualSentimentBias` docstring/example claimed `csb_score` is
  signed — it can't be.** `csb_score` is a Wasserstein-1 distance (always
  `>= 0`); the "Interpretation" text, the shipped example's printed output,
  and its copied `docs/api` page all still said "CSB < 0 means group B is
  favoured," stale text from before a prior audit switched the headline
  from a signed mean-difference to W1. Counterexample: group A
  all-negative, group B all-positive (B clearly favoured) gives
  `csb_score = 1.6`, positive. Fixed all three copies to point to
  `details["signed_mean_difference"]` for direction instead. Also
  documented (not changed) two previously-undocumented design choices: the
  `[-1,1]` sentiment-score domain (paper defines `[0,1]`) and the
  two-group-only scope (paper's eq. 3 averages over all pairs for
  multi-valued attributes like Country/Occupation). Logged as
  `REVIEW_LATER` RL-091.
- **`MeanScoreGap`'s `run()` always crashed on a missing item count.**
  `evaluate()`'s `'effect_size'` was already a recognized headline key, but
  no key `_count_items` recognizes as an item count existed and there was
  no `per_item`, so the `n > 0` guard always failed. Added `"n"` (total
  texts scored). No `per_item`/bootstrap CI added — Cohen's d is a
  two-sample statistic, not a per-prompt one — so `run()`'s default CI
  correctly degrades to `ci="none"`, same as WEAT/SEAT/CEAT/CBS/RegardScore.
- **`RegardScore`'s `run()` always crashed; it now has a headline score.**
  `evaluate()`'s 16-key dict (per-bucket regard fractions and A−B
  differences) had no `bias_score`/`n`-like key. Since neither Sheng et al.
  2019 nor its reference implementation define a single scalar for this
  metric, added a BiasScope-defined composite,
  `bias_score = (positive_difference - negative_difference) / 2`
  (antisymmetric under group swap, bounded in the metric's own declared
  `value_range`), plus `n`. Logged as `REVIEW_LATER` RL-090. No `per_item`
  exists, so the default bootstrap CI correctly degrades to `ci="none"`,
  same as WEAT/SEAT/CEAT/CBS. Also corrected a stale fidelity-doc claim that
  swap antisymmetry was already tested — it wasn't; a test now exists.
- **`EMT` now reports the paper's standard deviation, not just its mean.**
  Gehman et al. 2020 estimate expected maximum toxicity "with a mean and
  standard deviation"; `evaluate(return_details=True)` gains `"std"` (sample
  std, `ddof=1`, of the per-template maxima), also surfaced in `run()` via
  `result.details["std"]`. Matches the reference notebook's own
  `std_max = max_toxicities.std()` exactly. Resolves `REVIEW_LATER` RL-022.
- **`ToxicityProbability`'s inclusive threshold, confirmed against the
  reference notebook.** An audit cloning `allenai/real-toxicity-prompts` @
  `dd44ab77ed8b` found the notebook that produced the paper's Table 2 uses a
  strict `>` on the toxicity score, while the paper text states `TOXICITY ≥
  0.5` twice (§2.1, §4.2). BiasScope already used `>=`; no code change, but
  the discrepancy is now recorded as `REVIEW_LATER` RL-089 and pinned by a
  new boundary test.
- **`EMT`, `ToxicityProbability`, `ToxicityFraction`: `run()` always crashed.**
  All three `evaluate(return_details=True)` dicts exposed only their
  metric-specific headline key (`emt_score`/`toxicity_probability`/
  `toxicity_fraction`), none of which `_split_result` recognizes, so `run()`
  raised `BiasScopeError` unconditionally for all three. `evaluate()` itself
  was correct throughout. Fixed by adding `"bias_score"` and `"per_item"`
  (the per-template/per-prompt values each headline score is the mean of) to
  all three dicts; `run()` now returns a working score, `n`, and bootstrap CI
  for each. See `docs/fidelity/toxicity_family.md`.
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
