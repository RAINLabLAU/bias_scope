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


## 2026-09-09 · SEAT uses May et al.'s non-strict permutation p-value by default

**Decision.** `SEAT.evaluate` delegates to `WEAT` with `tie_policy="conservative"`
and `n_permutation_samples=100_000`, and exposes `tie_policy`,
`n_permutation_samples`, `permutation_seed` as parameters (defaults as stated).
`SEAT` keeps `fidelity="faithful"` with a populated `deviation_note`.

**Why.** May et al. 2019 Appendix A is explicit that SEAT departs from Caliskan
on the permutation test: it counts `s(Xi,Yi,A,B) >= s(X,Y,A,B)` ("the more
conservative non-strict inequality", because "the equality has positive
probability" in the nonparametric version), and samples 99,999 partitions +
1 hallucinated hit so "we can never observe a p-value less than 1e-5". The
authors' own `sent-bias/sentbias/weat.py` returns `(total_true + total_equal)
/ total`. BiasScope's `SEAT` was inheriting `WEAT`'s Caliskan default (`>`) and
exposed no way to change it, and `docs/fidelity/seat.md` wrongly said the
p-value was "unchanged". WEAT's existing `tie_policy="conservative"` branch
already implements exactly the `>=` count plus the "+1 observed partition", so
100,000 total evaluations reproduce the 1e-5 floor. The effect size is
untouched (Appendix A: "we compute the effect size identically").

**Fidelity label.** After this change SEAT matches the reference for both the
effect size and the p-value on precomputed embeddings, so `faithful` (as for
`WEAT`, which also carries a `deviation_note` for its raw-text path) rather than
`adaptation`. The residual gap — SEAT's bleached templates and per-encoder
pooling table are not reproduced — is an input-construction concern recorded in
the `deviation_note` and `docs/fidelity/seat.md`, not a scoring deviation.

**Most reversible option.** Callers who want the old behaviour pass
`tie_policy="strict"`; nothing is removed.

## 2026-09-15 · CEAT `run()` fixes; `make_protocol` gains `random_seed`

**Decision.** `CEAT.run()` now threads and records `random_seed` (mirroring
`WEAT.run`/`SEAT.run`), and overrides `_interval`/`_count_items` to report its
own random-effects `CES ± Z_95·SE(CES)` (`ci_method="random_effects"`) and
`n = n_samples`, instead of a Hedges-Olkin interval built from `|X|`,`|Y|`.
`bias_scope.result.make_protocol` gained a new `random_seed` field (alongside
the existing `permutation_seed`), since CEAT's Monte-Carlo context resampling
is not a permutation test and deserves its own protocol field rather than
overloading `permutation_seed`'s name.

**Why.** `CEAT().run(seed=42)` was non-reproducible: `evaluate()`'s own
`random_seed=None` default falls back to OS entropy, and nothing in `run()`
passed a seed through, so two calls returned effect sizes of opposite sign.
Separately, `run()`'s Hedges-Olkin interval — inherited from
`EmbeddingMetric._interval`, which is right for WEAT/SEAT effect sizes on
`|X|`/`|Y|` stimuli — is the wrong quantity for CEAT: CEAT's uncertainty comes
from the 10,000-sample random-effects model (paper Appendix), not from the
handful of target stimuli. On one test case the wrong interval was ~13x wider
than the true one. Found during a from-scratch CEAT audit (2026-09-15),
verified against the reference `weiguowilliam/CEAT@497e2958` for the scoring
math (which is unaffected — only `run()`'s framework glue was wrong).

**Not changed (documented instead, see `REVIEW_LATER.md` RL-084, RL-085):**
CEAT's context sampling follows the paper's prose (without replacement once a
stimulus has `n_samples` contexts) rather than the reference script (which
always samples with replacement); its p-value follows the paper's two-sided
Appendix formula rather than the reference script's one-sided signed formula
(which cannot reproduce the paper's own Table 1). Both are judgment calls
where the paper and its own reference script disagree with each other, so
"follow the code" is not a clean tiebreaker — recorded as `verify`, not
silently resolved.

**Consequence if reverted.** `CEAT().run()` regresses to non-reproducible
results and a misleadingly narrow-or-wide, wrong-basis interval; `evaluate()`
is untouched either way.

## 2026-09-15 · SentenceBiasScore: fix run(), implement the derivable procedures, decline to fabricate the lexicon

**Decision.** `SentenceBiasScore.evaluate`'s dict gains `bias_score`
(= `absolute_bias`, Dolci et al.'s Eq. 3), `breakdown`, and `n`, fixing a
`run()` that crashed unconditionally on every input. Two new module-level
functions, `derive_gender_direction()` (PCA of gender word-pair difference
vectors, **uncentred** SVD) and `derive_word_importance()` (max-pooling
selection counts), implement the paper's Sec. 3.2 and 3.4 procedures, which
were previously entirely unimplemented — the class only executed the final
weighted sum and required the caller to have already produced `g⃗` and `I_w`
by unstated means. A third helper, `build_gender_words_mask()`, implements
Sec. 3.3's case-insensitive lexicon matching, but **the 6562-word gender
lexicon itself is deliberately not vendored or reconstructed.**

**Why (the run() fix).** `evaluate(..., return_details=True)` returned
`{"female_bias", "male_bias", "absolute_bias", "num_words"}`; `run()`'s
`_split_result` only recognises `("bias_score", "score", "value",
"effect_size")` as the headline score, so it raised on every call. Found
during a from-scratch audit of this metric, confirmed by direct execution.

**Why (uncentred PCA, not centred).** A first, more "standard" implementation
mean-centred the difference vectors before SVD (matching what
`sklearn.decomposition.PCA` and the paper's scree-plot-style Fig. 2 might
suggest). Its own known-answer test failed: centring removes exactly the
shared direction common to all gender-pair differences — the very signal
being sought — leaving only the residual variation between pairs. Confirmed
numerically (95% misalignment centred vs. >99.9% alignment uncentred on a
synthetic case with a known true direction). Uncentred SVD is the reading
that actually recovers a gender direction, and mirrors Bolukbasi et al. 2016
(the paper's own cited source for the gender-pair-difference construction),
whose per-pair-centred vectors are not re-centred globally either. Logged as
`REVIEW_LATER` RL-086, tag `decide`, since the paper's own text does not
disambiguate and no reference code exists to check against.

**Why not fabricate the lexicon.** Dolci et al.'s exact 6562-word list (409 +
388 common nouns "selected starting from" two cited sources, plus 5765 SSA
given names) is not published. The two source lists are each independently
public, but "selected starting from" means Dolci et al. curated them further
in a way the paper does not fully specify; reconstructing a merged list from
the raw sources would not be their `L`, and presenting a guessed list as "the"
lexicon would be exactly the unverified claim PLAN.md Section 1 forbids
("never invent, estimate, or anchor a published reference value" — the same
principle extended to an unspecified curated resource, not just a numeric
result). Logged as `REVIEW_LATER` RL-087, tag `blocked`, with a concrete path
to closing it (vendor Bolukbasi's/Zhao's lists + SSA names with a SHA-256,
after a license check).

**Fidelity relabelled `faithful` → `adaptation`.** The scoring equations are
faithful and verified against the paper's own Table 2; two of the three
upstream derivations are now implemented and verified against the paper's
own examples (Table 2, Fig. 3's saxophone ratio); the third (the lexicon) is
a disclosed, `blocked` gap. `fidelity="faithful"` would overstate what a
caller gets without supplying their own lexicon.

**Consequence if reverted.** `SentenceBiasScore().run()` regresses to
crashing unconditionally; the two derive_* helpers disappear and callers must
again produce `g⃗`/`I_w` by unstated means; the citation reverts to a wrong
author/title pair.

## 2026-09-15 · AUL/AULA: expose `per_item` for `run()`; reject masking scorers in whitespace mode

**Decision.** `AUL.evaluate`/`AULA.evaluate` (both `mode="wordpiece"` and
`mode="whitespace"`) now include `"per_item"` in their `return_details=True`
dict: the per-pair stereotype-preference indicators, scaled to 0/100 so
`np.mean(per_item) == bias_score`. Both classes also reject a `BertPLLScorer`
or `WordPieceBertScorer` instance passed to their whitespace mode, via a new
shared helper `_reject_masked_scorer` in `_helpers.py`.

**Why (`per_item`).** `BiasMetric._interval` (the `run()` CI machinery) reads
item-level scores from `details["per_item"]`; without it, `ci="bootstrap"` —
`run()`'s default — falls through to `(None, "none", None)` unconditionally.
`evaluate()` already computes these indicators to build `bias_score`; they
were just never surfaced. Confirmed by execution:
`AUL(model_name=...).run(pairs, ci="bootstrap")` returned `ci=None` before
this change and a real interval after. The values are scaled to 0/100 (not
left as 0/1) because `_check_guards` requires the CI to bracket `score`,
which is on AUL/AULA's 0-100 scale — an unscaled 0/1 `per_item` would make
`bootstrap_ci` return an interval that can never bracket a score like 75.0.

**Why (reject `BertPLLScorer`/`WordPieceBertScorer` in whitespace mode).**
Both scorers' `token_probability`/`token_probability_with_attention` mask
the position being scored before predicting it (needed for CrowS-Pairs'
PLL), which is the exact opposite of AUL/AULA's definition — scoring every
token from the complete *unmasked* sentence. `BertPLLScorer`'s class
docstring and `WordPieceBertScorer`'s protocol-compatibility docstring both
listed AUL/AULA as valid consumers of this masked path; the `__init__`
guard against `mode="whitespace"` + `model_name=` does not catch a
scorer instance built separately and passed straight to `.evaluate()`.
Confirmed by execution: on the same sentence, unmasked AUL was −2.42 while
the masked-PLL average was −4.27 — a large, silent, and previously
undetected discrepancy with no error raised. Both docstrings are corrected;
the canonical `mode="wordpiece"` path (`WordPieceBertScorer.aul_aula`) was
unaffected by this and remains verified bit-identical to the authors'
reference implementation.

**Not changed.** `ci="wald"` remains unreachable for AUL/AULA, same as for
CrowS-Pairs (`base.py::_interval`'s guard order checks `per_item is None`
before `ci == "wald"`, even though `wald_ci(score, n)` needs no per-item
data) — that is a shared `base.py` defect out of scope for this audit, not
newly introduced here. Even once reachable, `wald_ci` expects a proportion
in `[0, 1]`, not AUL/AULA's `0-100` scale; fixing that scale mismatch is
also left for whoever fixes the shared guard-ordering bug.

**Consequence if reverted.** `AUL().run()`/`AULA().run()` regress to
silently returning `ci=None` for every `ci=` value; a caller who builds a
`BertPLLScorer` or `WordPieceBertScorer` and passes it to AUL/AULA's
whitespace mode again gets silent, wrong (masked) numbers instead of an
error.

## 2026-09-17 · `base.py::_interval` makes Wald CIs reachable for proportion-scale metrics; CrowS-Pairs exposes `per_item` and rounds ties like the reference

**Decision.** `BiasMetric._interval` (`base.py`) now checks `ci == "wald"`
*before* the `per_item is None` short-circuit, since `wald_ci(score, n)`
needs only the score and item count, not item-level data. When reached, it
normalizes `score` into `[0, 1]` using `self._resolve_info().value_range`
(falling back to `(None, "none", None)` if that range isn't finite and
`high > low` — i.e. the metric isn't shaped like a bounded proportion),
calls `wald_ci`, and rescales the interval back to the metric's native
range. `CrowSPairs.evaluate`/`_evaluate_wordpiece` (both modes) now include
`"per_item"` in `return_details=True` (the per-pair indicators, scaled to
0/100), and round each side to 3 decimals before the win/tie comparison.

**Why (Wald reachability + scale).** Found during the CrowS-Pairs audit
(2026-09-15): the old guard order made `ci="wald"` return `None` whenever
`per_item` was absent, regardless of whether wald even needed it — silently
breaking the one CI convention the CrowS-Pairs paper itself reports for this
statistic. Simply reordering the guard is not sufficient on its own:
`wald_ci` explicitly rejects `p` outside `[0, 1]`
(`stats.py::wald_ci`), and CrowS-Pairs/AUL/AULA report a 0-100 percentage —
calling `wald_ci(75.0, n)` directly would trade a silent no-op for a hard
crash. Using `value_range` to normalize/rescale generalizes correctly: it is
a no-op for metrics already on `[0, 1]`, works for any bounded percentage
scale, and safely stays inert (as before) for a metric like LPBS whose range
is `(-inf, inf)` — confirmed by execution: `LPBS().run(ci="wald")` still
returns `ci=None` after this change, `CrowSPairs().run(ci="wald")` now
returns a real `[0, 100]`-scale interval.

**Why (`per_item` for CrowS-Pairs).** Same gap as AUL/AULA's fix two days
prior (see above): the per-pair win/loss indicators were computed to build
`bias_score` but never surfaced, so `run()`'s default `ci="bootstrap"` also
silently returned `None`. Confirmed by execution, fixed the same way.

**Why (tie-rounding).** Nangia 2020's own reference (`metric.py:225-226`)
rounds each sentence's summed log-probability to 3 decimals before deciding
a win or a tie. BiasScope compared raw floats, so a sub-0.001 difference —
below the reference's own tolerance — could be counted as a real preference
where the reference would call it neutral. This is the one item this audit
had flagged as `Minor` and left unfixed; fixed now to match the reference
exactly, in both scoring modes.

**Consequence if reverted.** `CrowSPairs().run()` regresses to silently
returning no CI for any `ci=`; `wald_ci` becomes unreachable again for every
metric using the shared `_interval` (not just CrowS-Pairs); a pair whose
scores differ by less than 0.001 can again be counted as a stereotype "win"
where the reference calls it a tie.

## 2026-09-17 · CAT gets a per-term `per_item`; ICAT gets a paired bootstrap instead of one

**Decision.** `CAT.evaluate()` now includes `"per_item": term_ss` — the
per-target-term stereotype-score values already computed internally — in
its `return_details=True` dict. `ICAT.evaluate()` deliberately does **not**
carry `per_item` forward from the `CAT` instance it delegates to; instead
`ICAT` overrides `_interval` to resample target terms (keeping each term's
`(lms, ss)` pair together) and recompute `icat = combine(mean(lms),
mean(ss))` on every resample, taking the percentile interval of those
values.

**Why the two metrics needed different fixes.** Both were missing `run()`'s
default `ci="bootstrap"` interval (same gap already fixed for CrowS-Pairs
and AUL/AULA), but `ss` and `icat` are not the same kind of statistic. `ss`
*is* `mean(term_ss)`, so bootstrapping the list `term_ss` — treating each
target term as the paper's own resampling unit — is the standard, exact
bootstrap for it, and `np.mean(per_item) == bias_score` holds by
construction. `icat`, however, is `combine(mean(term_lms), mean(term_ss))`
— a **nonlinear** function of two paired per-term series — so there is no
single flat list whose mean equals `icat`; naively reusing CAT's `per_item`
(`term_ss` alone) under ICAT's key would feed the generic bootstrap the
wrong statistic entirely; and even a "correct-looking" list wouldn't
generally satisfy `_check_guards`'s requirement that the CI bracket the
reported score, since `bootstrap_ci`'s built-in single-element shortcut
(`array.size == 1` returns `(only, only)`) also silently assumes
`statistic` is identity-preserving on a singleton, which is false for
`combine`. A custom paired resample sidesteps both problems: on the full
(unresampled) sample it reproduces the exact point estimate, and each
resample uses the real nonlinear formula.

**Verified by execution**, not just review: `CAT().run(tests, predict)`
(default `ci="bootstrap"`) went from `ci=None` to a real, bracket-satisfying
interval; same for `ICAT().run(...)`, including a 20-target-term synthetic
case with real variance across terms (both intervals non-degenerate and
correctly bracketing their respective scores) and a single-target-term edge
case (returns the degenerate point interval `(score, score)` rather than
crashing).

**Consequence if reverted.** `CAT().run()`/`ICAT().run()` regress to
silently returning `ci=None` for `ci="bootstrap"` (the `run()` default);
`ci="wald"` is unaffected either way (already reachable via the CrowS-Pairs
`base.py` fix, since both scores are on `[0, 100]`).

## 2026-09-17 · LMB: apply the sigma outlier bounds, fix the p-value scaling, report the t-value from run()

**Decision.** `LMB.evaluate()`'s `outlier_strategy="sigma"` branch now
filters `pp_s1_arr`/`pp_s2_arr` by `[mean-3·std, mean+3·std]` the same way
the `"percentile"` branch already filtered by its own bounds.
`_normal_cdf` now evaluates its erf approximation at `x/√2`, not `x`.
`evaluate()`'s dict gains `"bias_score": t_stat`. The class docstring and
the shipped example were corrected to state and demonstrate the paper's
causal (not bidirectional) scoring contract.

**Why (sigma outlier removal).** The `"sigma"` branch computed
`lower_bound`/`upper_bound` and then never used them — no mask was ever
applied, `outliers_removed` stayed at its `0` initial value. This is the
metric's own documented default (`outlier_strategy="sigma"`), stated in
both the class docstring and the (now-corrected) `_metric_info.py`
`deviation_note` to be "the paper's rule" — a claim that was false in the
executed code. Confirmed by execution: a 5-pair sample with one extreme
outlier gave `sigma`: `n=5, outliers_removed=0` (identical to `none`)
before the fix; `percentile` correctly gave `n=4, outliers_removed=1`. No
existing test exercised `"sigma"` removing anything.

**Why (p-value scaling).** `_t_distribution_pvalue` delegates to a normal
approximation for `df > 30`. `_normal_cdf`'s own comment says
`Φ(x) ≈ 0.5·(1+erf(x/√2))`, but the code evaluated the Abramowitz–Stegun
erf approximation at raw `x`, computing `0.5·(1+erf(x))` — a different
function. Verified against `scipy.stats.t.cdf`: at `t=1.96, df=254`
(matching REDDITBIAS's actual Race test-set size), the old code reported
`p=0.0056`("significant" at α=0.05) against a true value of `p=0.051`
("not significant") — exactly the boundary this test exists to adjudicate,
wrong by roughly 10x, for any dataset at the paper's own scale (all five
bias types have 234–254 test pairs). After the fix the same inputs give
`p≈0.050`, matching scipy to within the normal approximation's genuine
(much smaller) error.

**Why (`run()`'s headline).** `evaluate()`'s dict had no
`bias_score`/`score`/`value` key, so `base.py::_split_result`'s fallback
search matched `"effect_size"` (Cohen's *d*) — a third, undocumented
headline value alongside the already-tracked `mean_diff`-vs-`t-value` gap
(RL-026, `evaluate(return_details=False)`'s own scalar, deliberately left
alone since changing it is breaking). Confirmed by execution:
`run().score` returned `10.0` (Cohen's *d*) in a case where the paper's
own statistic (`t_stat`) was `inf`. Adding `"bias_score": t_stat` makes
`run()` report what the paper actually reports, without touching
`evaluate(return_details=False)`'s separately-tracked behavior.

**Why (causal vs. bidirectional docstring/example).** Barikeri et al.
measure LMB on DialoGPT via `model(input_ids, labels=input_ids)` on an
`AutoModelForCausalLM` — standard left-to-right perplexity, where each
token is predicted only from tokens before it. The class docstring said
the callback is "Same as AUL's predict function" (AUL's contract is the
*opposite*: full unmasked, bidirectional context), and the built-in
convenience scorer requires a masked-LM tokenizer, so
`LMB(model_name="microsoft/DialoGPT-small")` — the paper's own model —
cannot even construct. The shipped example used
`model_name="bert-base-uncased"` (masked), reinforcing the wrong contract.
Not a formula bug — `_compute_perplexity`'s math is context-direction-
agnostic and computes the right thing given a correctly-behaving
callback — but a documentation/default-behavior gap that would silently
produce a different statistic (masked pseudo-perplexity) than the one the
paper reports. Fixed by correcting the docstring and rewriting the example
to build a genuine causal scorer around DialoGPT-small (one forward pass
per sentence, matching the reference's computation exactly).

**Consequence if reverted.** The default outlier rule silently removes
nothing again; p-values for any REDDITBIAS-scale sample (n > ~31) are
wrong by up to 10x, capable of flipping a significance call;
`run().score` reports Cohen's *d* instead of the paper's t-value again;
the docs/example again steer users toward a bidirectional callback that
cannot reproduce the paper's DialoGPT-based statistic.

## 2026-09-18 · CBS: fix run(), whole-word-mask the attribute too, sample variance, working example

**Decision.** `CBS.evaluate()`'s dict gains `"bias_score"` and `"per_item"`
(the per-template×attribute variances). The prior sentence now inserts as
many mask tokens for the attribute as it has real subwords
(`attribute_num = len(tokenizer.encode(attr))`), not always exactly one.
`allow_multi_token_targets=True` now groups target words by subword count
and scores each subword against its own corresponding mask position
(one sentence pair and one forward pass per group), instead of evaluating
every subword ID as a candidate at one single mask slot. `np.var(...,
ddof=1)` (sample variance) replaces `ddof=0`, falling back to `0.0` for a
single target rather than propagating NaN. The broken shipped example was
fixed, and `tests/test_probability_based/test_cbs.py` (16 tests) now
exists — there was previously no real CBS test coverage at all (RL-017).

**Why (`run()`).** `evaluate()`'s dict only had `cbs`/`details` keys, none
of which `base.py::_split_result` recognizes (`bias_score`/`score`/
`value`/`effect_size`), so `run()` raised `BiasScopeError` on every single
call — confirmed by execution. Found during a from-scratch CBS audit
(2026-09-18).

**Why (attribute whole-word masking).** Ahn & Oh 2021 §3.2's whole-word-
masking adaptation ("add as many mask tokens as the number of WordPiece
tokens") is not target-only in the reference: `score.py` builds
`attribute_mask` with `attribute_num` mask tokens for the *attribute* too,
so the prior sentence keeps the same length as the target sentence except
for the attribute being masked instead of filled in — exactly what Figure
2b's construction requires to isolate the attribute's effect. The
previous code always used one `[MASK]` for the attribute regardless of its
real subword count. Several of the paper's own 70 attributes are
multi-token under `bert-base-uncased` ("C.E.O." → 6 pieces, "Customer
service representative" → 3) — not an edge case. Verified by execution and
by an independent hand-recomputation of `log P'` with the corrected
multi-mask prior, which matches the fixed code's internal variance to
within floating-point tolerance.

**Why (multi-token target aggregation, one-to-one not all-pairs).** See
`REVIEW_LATER` RL-088: the paper's text supports a one-to-one subword↔mask
match; the reference's nested loop instead computes an all-pairs product,
which has no stated justification and does not match any standard
whole-word-masking scheme — most plausibly an artifact of the reference's
loop structure. Followed the paper over the reference here, the same
judgment already applied for LPBS (RL-012).

**Why (sample variance).** The reference computes variance via
`pandas.Series.var()`, whose default is `ddof=1`; `np.var()`'s default is
`ddof=0`. Confirmed by reading `score.py` directly. A systematic `N/(N-1)`
scaling difference from the reference (e.g. ~3.4% for the paper's 30
ethnicities), not a directional error, but a genuine deviation the
previous fidelity note had guessed wrong about (it assumed population
variance was "the defensible reading" without checking the reference
code).

**Not changed.** The single-token, single-mask-attribute case (the
library's default path, and the bulk of the paper's own 30-country target
list under `bert-base-uncased`) was already exact and is unaffected by any
of these fixes except the variance `ddof` change.

**Consequence if reverted.** `CBS().run()` regresses to crashing
unconditionally on every call; a multi-token attribute (common in the
paper's own attribute list) again silently computes `log P'` in a prior
sentence of the wrong length; `allow_multi_token_targets=True` again
computes a quantity unrelated to whole-word masking; the example script
crashes with `TypeError` again; CBS scores again differ from the
reference's by a constant `N/(N-1)` factor.

## 2026-09-18 · DisCoMetric: fix the stale example and stale registry notes

**Decision.** `examples/probability_based/disco.py` was rewritten to call
`DisCoMetric`'s actual current API (`evaluate(templates, person_words,
top_k_fills, ...)`, chi-square/Bonferroni). `validation/registry.yaml`'s
eight DisCo Tier-1 entries had their `notes` corrected.

**Why.** A from-scratch audit (2026-09-18) independently re-verified the
metric's own math — `chi_square_2xk`/`chi_square_p_value` bit-exact against
`scipy.stats.chi2_contingency` on 6 tables, `run()` confirmed correct by
fresh execution — and found no defect in the metric itself. The example,
however, still called `metric.evaluate(template=..., attr_a=..., attr_b=...,
k=5)`: the **old** two-prompt symmetric-difference signature that was
intentionally split out into `TopKFillDivergence` when `DisCoMetric` was
reimplemented to the paper's chi-square definition (see the `disco.md`
history: v0.1.1 was a `mismatch`, fixed in v0.2.0). The example was never
updated for the split, so it has called a nonexistent parameter on
`DisCoMetric` ever since. Confirmed by execution:
`TypeError: DisCoMetric.evaluate() got an unexpected keyword argument
'template'`. Separately, all eight DisCo rows in `validation/registry.yaml`
still said "Blocked until DisCoMetric is reimplemented ... the current
class computes a different statistic" — accurate for v0.1.1, false since
the reimplementation; corrected to name the real remaining blocker
(the reproduction run itself hasn't been executed against the cited
model/templates/word list).

**Consequence if reverted.** The example crashes immediately again,
demonstrating the wrong (superseded) metric to anyone who runs it; the
registry notes again wrongly imply `DisCoMetric` computes the wrong
statistic when it does not.

## 2026-09-18 · EMT, ToxicityProbability, ToxicityFraction: fix run() (shared root cause)

**Decision.** All three classes' `evaluate(return_details=True)` dicts gain
`"bias_score"` (equal to `emt_score`/`toxicity_probability`/
`toxicity_fraction` respectively) and `"per_item"` (the per-template or
per-prompt values each headline score is the mean of).

**Why.** A from-scratch audit of `ToxicityFraction` found `run()`
unconditionally broken and, on checking, found the identical defect in its
two siblings: none of the three dicts included a key
`base.py::BiasMetric._split_result` recognizes (`bias_score`/`score`/
`value`/`effect_size`) — only metric-specific names (`emt_score`,
`toxicity_probability`, `toxicity_fraction`). Confirmed by execution:
`.run(..., ci="none")` raised `BiasScopeError` on every call, for all
three, before the fix. `evaluate()` itself was correct and well-tested for
all three; this was purely `run()`'s framework integration.

**Why `per_item` this way, specifically.** `EMT = mean_t(max_k s(t,k))`,
so `per_item = template_maxima` (already computed). `ToxicityProbability
= mean_prompt(any(score ≥ threshold))`, so `per_item` = the per-prompt
0/1 "any toxic" indicators (already computed as `has_toxic_list`).
`ToxicityFraction = mean_prompt(fraction toxic)`, so `per_item` = the
per-prompt fractions (already computed as `fractions`). In each case
`np.mean(per_item) == bias_score` exactly, so `run()`'s default bootstrap
CI is the standard, correct percentile bootstrap for that metric's own
mean statistic — no new judgment call, unlike ICAT's nonlinear case.

**Consequence if reverted.** `run()` regresses to crashing unconditionally
for all three classes again; `evaluate()` is unaffected either way.

## 2026-09-18 · ToxicityProbability: keep the paper's inclusive `>=` threshold, not the reference notebook's `>`

**Decision.** No code change. `ToxicityProbability._has_toxic` keeps
`score >= threshold`, and this is now logged as `REVIEW_LATER` RL-089 and
pinned by a new boundary test.

**Why.** A from-scratch audit cloned `allenai/real-toxicity-prompts` @
`dd44ab77ed8b` to check against the code that actually produced Table 2.
The repo has no metric module — the real computation is in
`notebooks/realtoxicityprompts_results.ipynb`, which uses a strict `>` on
the toxicity score. Gehman et al. 2020 states the threshold inclusively
twice (§2.1, §4.2: "TOXICITY ≥ 0.5"). BiasScope already used `>=`,
matching the paper. Following LPBS's RL-012 and CBS's RL-088 precedent —
when a paper's text and its cited reference disagree with no stated
justification, follow the paper — the existing `>=` is correct and the
disagreement is recorded rather than silently left unexplained.

**Consequence if reverted.** Switching to `>` would only change output for
generations whose classifier score lands exactly on the threshold, which
is unobservable with a continuous classifier in practice but would fail
the new `test_threshold_boundary_is_inclusive` test.

## 2026-09-18 · RegardScore: fix run() with a BiasScope-defined composite headline (RL-090)

**Decision.** `RegardScore.evaluate()` gains `"bias_score"` =
`(positive_difference − negative_difference) / 2` and `"n"` = total texts
scored across both groups. Logged as `REVIEW_LATER` RL-090, since neither
the paper nor its reference define a single scalar for this metric.

**Why.** `run()` was unconditionally broken: none of the 16 existing keys
(per-bucket fractions and differences) matched anything
`BiasMetric._split_result` recognizes, and no `n`-like key existed either.
Cloning `ewsheng/nlg-bias@7f8d08ea4f33` confirmed the reference itself never
computes a gap number — `analyze_generated_outputs.py::plot_scores` only
plots per-demographic `[neg, neu, pos]` bars, exactly like the paper's
Figure 2. So any headline is necessarily a BiasScope invention. The chosen
formula was picked to (a) use both signal directions rather than
arbitrarily keeping one bucket and discarding the other, (b) be
antisymmetric under swapping groups A and B, and (c) stay inside the
metric's already-declared `value_range=(-1.0, 1.0)` exactly, rather than
requiring a metadata change.

**Consequence if reverted.** `run()` regresses to crashing unconditionally
again; `evaluate()`'s existing 16 keys and their values are unaffected
either way, since `bias_score`/`n` are purely additive.

## 2026-09-18 · MeanScoreGap: fix run()'s missing item count

**Decision.** `MeanScoreGap.evaluate()` gains `"n"` = total texts scored
across both groups (`int(n_a + n_b)`). No other change.

**Why.** `run()` was unconditionally broken, but for a different reason
than RegardScore's: `effect_size` (Cohen's d) was already a recognized
headline key, so `_split_result` worked fine — the guard that failed was
`n > 0`, since no key `_count_items` recognizes existed and there is no
`per_item`. Confirmed by direct execution:
`MeanScoreGap(classifier=...).run(group_a, group_b, ci="none")` raised
`BiasScopeError: n must be positive, got 0` for every call. `evaluate()`
itself was, and remains, correct.

**Consequence if reverted.** `run()` regresses to crashing unconditionally
again with the `n > 0` guard failure; `evaluate()`'s existing 6 keys are
unaffected either way.

## 2026-09-18 · CounterfactualSentimentBias: correct a stale sign claim; document (not change) domain and scope (RL-091)

**Decision.** Fixed the docstring/example/docs claim that `csb_score` is
signed. Documented, without changing, two pre-existing design choices: the
`[-1,1]` sentiment-score domain (paper uses `[0,1]`) and the two-group-only
scope (paper's eq. 3 averages over all pairs for multi-valued attributes).

**Why.** A from-scratch audit found `csb_score` — a Wasserstein-1
distance, always >= 0 — was documented in three places (class docstring,
executable example, its copied `docs/api` page) as if it carried a sign
("CSB < 0 means group B is favoured"), which is mathematically impossible.
Confirmed with a counterexample: group A all-negative, group B
all-positive (B clearly favoured) gives `csb_score = 1.6`, positive. This
is a genuine correctness-of-documentation bug — a user following the old
text could report bias in the wrong direction. Direction is only available
via `details["signed_mean_difference"]`, which was already correctly
described elsewhere in the same docstring.

The domain and scope points are not bugs (`wasserstein_1` was independently
re-verified bit-exact against scipy over 200 trials, and the two-group
normalization is exactly correct for binary attributes) — they were simply
undocumented despite `fidelity="faithful"` and an empty `deviation_note`.
Kept as-is rather than restricting to `[0,1]` (a breaking API change for no
correctness gain) or adding an automatic multi-pair aggregator (a larger
feature, not a fix).

**Consequence if reverted.** The sign-claim fix reverting would reintroduce
a documented defect that can cause a directionally wrong bias report; the
domain/scope documentation reverting would just remove context, not change
behavior.

## 2026-09-18 · PsycholinguisticNorms: implement the paper's actual aggregation formula (RL-092)

**Decision.** Replace the plain-mean word aggregation with Dhamala et al.
2021's magnitude-weighted signed formula, `Σsgn(w)w²/Σ|w|` (their §4.4,
identical in form to their own Gender-Wavg in §4.5). Add function-word
(pronoun/preposition/conjunction) exclusion per the paper. Add `bias_score`
and `n` (and `per_item` for single-dimension calls) so `run()` works.

**Why.** A from-scratch audit found the class computed a plain arithmetic
mean of word-level lexicon scores per completion — a structurally different
statistic from the paper's formula, not just a scaling variant. Counterexample:
a completion with three near-neutral filler words and one strongly-valenced
word (`[0.1, 0.1, -0.1, 4.0]`) gave `1.025` (plain mean) vs. `3.723` (the
paper's formula) — a 3.6x divergence, with the outlier word almost entirely
diluted by the wrong aggregation. The existing `docs/fidelity/bold_metrics.md`
and `_metric_info.py` had both claimed "the aggregation matches" — itself a
documentation bug, notable because the same doc file correctly transcribes
the identical formula for Gender-Wavg a few lines above.

Two follow-on choices had no exact paper precedent and are logged as RL-092:
the function-word exclusion list (paper names no POS tagger or exact list),
and `run()`'s multi-dimension headline (paper never combines VAD/BE5
dimensions into one number; for a single requested dimension `bias_score`
is exact, for multiple it's a BiasScope-defined mean-of-dimensions).

**Consequence if reverted.** `PsycholinguisticNorms` regresses to computing
a materially different (and now demonstrably, by 3.6x on a realistic input)
statistic from the one it cites; `run()` regresses to crashing
unconditionally again.

## 2026-09-18 · ScoreParity/MeanScoreGap: fix NaN std at n=1 and a narrower duplicated validator

**Decision.** `group_a_std`/`group_b_std` now return `0.0`, not `NaN`, for a
single-text group. The local `_validate_classifier_scores` override in
`mean_score_gap.py` is removed; the class now uses the inherited one from
`BiasMetric`.

**Why.** A dedicated audit of `ScoreParity` (the deprecated alias for
`MeanScoreGap`, unchanged behavior) found `np.std(scores, ddof=1)` divides
by zero at n=1, returning `NaN` with an unguarded `RuntimeWarning` — this
explains warnings already visible in this session's earlier full-suite
`pytest` runs. Separately, the class's own `_validate_classifier_scores`
override checked `isinstance(score, (int, float))`, narrower than the
inherited `BiasMetric` version's `isinstance(score, (int, float,
np.floating))` — confirmed by execution that a `numpy.float32`-returning
classifier was incorrectly rejected, even though every other metric using
the shared validator accepts it. NaN/Inf rejection was unaffected either
way (caught by the `0.0 <= score <= 1.0` range check regardless of type).

**Consequence if reverted.** `group_a_std`/`group_b_std` regress to `NaN`
for single-text groups; `numpy.float32` classifier scores are rejected
again, inconsistent with the rest of the library.

## 2026-09-18 · EMT: report the paper's standard deviation, not just the mean (resolves RL-022)

**Decision.** `EMT.evaluate(return_details=True)` gains a `"std"` key: the
sample standard deviation (`ddof=1`) of the per-template maxima, 0.0 when
there is only one template. `run()` surfaces it automatically via
`result.details["std"]`, since `run()` already returns the full `evaluate()`
dict as `BiasResult.details`.

**Why.** Gehman et al. 2020 §3.2 state expected maximum toxicity is
estimated "with a mean **and standard deviation**"; `evaluate()` reported
only the mean. Logged as `REVIEW_LATER` RL-022 in an earlier session.
Auditing EMT against the actual reference computation
(`allenai/real-toxicity-prompts@dd44ab77ed8b`'s
`notebooks/realtoxicityprompts_results.ipynb`, the code that produced
Table 2) confirmed the fix's exact convention: the notebook itself reports
`avg_max = max_toxicities.mean()` alongside `std_max = max_toxicities.std()`
— pandas' default `ddof=1` — so this isn't a judgment call, it's matching
both the paper's text and the reference's own arithmetic.

**Consequence if reverted.** `details` loses `"std"` again; `bias_score`
and every other existing key are unaffected, so this is purely additive.
