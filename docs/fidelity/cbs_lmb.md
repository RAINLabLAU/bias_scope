# CBS and LMB

Two probability-family metrics, audited together for brevity; separate papers.

## CBS — Categorical Bias score

**Cited source:** Ahn & Oh 2021, *Mitigating Language-Dependent Ethnic Bias in
BERT*, EMNLP 2021. [arXiv:2109.05704](https://arxiv.org/abs/2109.05704).
**Reference implementation:**
[jaimeenahn/ethnic_bias](https://github.com/jaimeenahn/ethnic_bias) @
`a115eb7c3af7`, MIT.
**Sections and files read:** §3 "Measuring Ethnic Bias" (bias as "the degree of
variance of the probability of a country name given an attribute"); §3.1
"Normalized probability" (`P' = p_tgt / p_prior`, citing Kurita et al. 2019);
Figure 2b (variance for three or more target groups); repo layout.

### Definition

`P'(n) = p_tgt(n) / p_prior(n)` — the same prior-corrected construction as
[LPBS](lpbs.md), and the paper cites Kurita et al. for it. Then, per Figure 2b:

> when there are two target groups, the score is the difference of the
> normalized probabilities, and when there are more than two target groups, the
> score is the **variance** of the normalized probabilities.

CBS is therefore the **variance across target words of `log P'`**, averaged over
templates × attributes.

### Verdict: **faithful**

`cbs.py` computes `logP'(n) = logP_target(n) − logP_prior(n)`, takes the
sample variance over the targets, then the mean over templates × attributes.
That is the definition.

The implementation derives which mask occurrence in the prior sentence is the
**target's** from the template text (placeholder vs. mask-token position),
consistent with the decision made for LPBS (RL-012).

**Fixed in the 2026-09-18 audit** (independently re-cloned the reference and
re-derived from scratch — every claim below was re-verified, not carried
over from the previous version of this note):

- **`run()` was completely broken.** `evaluate()`'s dict had no
  `bias_score`/`score`/`value`/`effect_size` key (only `cbs`), so
  `base.py::_split_result` raised on *every* call to `run()` — confirmed by
  execution, on the first invocation. Fixed by adding `"bias_score"` (and
  `"per_item"`, the per-template×attribute variances, so `run()`'s default
  bootstrap CI works too — `cbs` is literally their mean).
- **The prior sentence didn't whole-word-mask the attribute.** §3.2's
  adaptation ("add as many mask tokens as the number of WordPiece tokens")
  applies to the attribute as well as the target — confirmed in the
  reference's `attribute_mask = ' '.join(['[MASK]']*attribute_num)`. The
  previous code always inserted exactly one `[MASK]` for the attribute
  regardless of its real subword count, breaking the target/prior
  structural parity Figure 2b relies on. Several of the paper's own 70
  attributes are multi-token under `bert-base-uncased` ("C.E.O." → 6
  pieces, "Customer service representative" → 3). Verified by execution:
  for `"customer service representative"`, using 1 mask vs. the correct 3
  shifted target-mask log-probabilities by up to 0.66 nats. Fixed: the
  prior sentence now inserts `attribute_num` mask tokens.
- **Multi-token targets (`allow_multi_token_targets=True`) computed a
  different, non-equivalent quantity.** The old code evaluated all of a
  target word's subword IDs as competing candidates at the *same single*
  mask slot and summed those log-probabilities — not whole-word masking
  (which needs one mask token per subword) at all. Fixed: target words are
  now grouped by subword count, each group gets its own sentence pair with
  that many target mask tokens (one forward pass per group, shared across
  every target word in it), and each subword is scored at its own
  corresponding mask position. This is deliberately **position-matched**
  (subword *i* of the target word scored against mask position *i*), not
  the reference's apparent nested-loop construction (which multiplies
  every subword's probability against *every* mask position — an
  all-pairs product rather than a one-to-one match, most likely an
  unintentional artifact of the reference's own loop structure rather
  than a deliberate design). Recorded as `REVIEW_LATER` RL-088 — following
  the paper's stated one-mask-per-subword design over a reference quirk
  that contradicts it, the same judgment already applied for LPBS's
  RL-012.
- **Variance convention now matches the reference.** `np.var(..., ddof=0)`
  (population variance) is now `ddof=1` (sample variance), matching the
  reference's `pandas.Series.var()` default — confirmed by reading
  `score.py` directly rather than assuming. Falls back to `0.0` (not NaN)
  when only one target word is supplied, since sample variance is
  undefined for a single observation.
- **The shipped example was broken.** `examples/probability_based/cbs.py`'s
  offline subclass had a stale override signature and crashed with
  `TypeError` on its very first call. Fixed to match the current
  `_log_normalized_target_scores` signature.

**Caveat that mattered more than the formula, now addressed:** `cbs.py` had
**no test file at all** (RL-017; a prior `test_cbs.py` was a near-verbatim
copy of `test_lpbs.py` with zero real CBS assertions and was removed). A new
`tests/test_probability_based/test_cbs.py` (16 tests) now covers the formula,
`run()`, both whole-word-masking paths (with independently recomputed
expected values, not just "doesn't crash"), the variance convention, and
input validation.

## LMB — Language Model Bias

**Cited source:** Barikeri, Lauscher, Vulić & Glavaš 2021, *RedditBias*, ACL
2021. [arXiv:2106.03521](https://arxiv.org/abs/2106.03521).
**Reference implementation:**
[umanlp/RedditBias](https://github.com/umanlp/RedditBias) @ `61f9ae9458e2`, MIT.
**Sections and files read:** §on the perplexity-based bias measure (outlier
removal, the t-value, the sign convention); repo layout.

### Definition

1. Compute perplexities for biased sentences and their counterfactual pairs.
2. **Remove outlier pairs**: drop a pair when either perplexity falls outside
   `[x̄ − 3s, x̄ + 3s]`, where `x̄` and `s` are the sample mean and standard
   deviation. The paper motivates this by citing Pollet & van der Meij 2017 on
   outliers distorting significance.
3. > we quantify and report the bias effect as the **t-value of the Student's
   > two-tailed test** between two ordered sets of corresponding perplexity
   > scores

4. "a negative `t` value indicates the presence of a (negative) stereotypical
   bias".

### Verdict: **adaptation**

**Fixed in the 2026-09-17 audit** (found by independently re-cloning the
reference and re-deriving the formulas from scratch — the claims below,
previously recorded in this file, turned out not to match the actual code):

- **The "sigma" outlier rule — claimed fixed in 0.2.0, actually never ran.**
  `evaluate()`'s `outlier_strategy="sigma"` branch (the *default*) computed
  `[mean-3·std, mean+3·std]` but never applied it: no mask, no array
  filtering, `outliers_removed` never left its `0` initial value. It was
  byte-for-byte identical to `outlier_strategy="none"`, and no test exercised
  it removing anything. Confirmed by execution with a 5-pair sample
  containing one extreme outlier: `sigma` reported `n=5, outliers_removed=0`
  (same as `none`); `percentile` correctly reported `n=4, outliers_removed=1`.
  Fixed by applying the same masking the `percentile` branch already did.
- **The p-value was wrong by up to 10x for any sample with `df > 30`.**
  `_normal_cdf` computed `0.5·(1+erf(x))` instead of `0.5·(1+erf(x/√2))` — the
  wrong function, missing the standard normal CDF's `/√2` scaling (its own
  comment already stated the correct formula; the code didn't match it). At
  `t=1.96, df=254` (REDDITBIAS's actual Race test-set size), this reported
  `p=0.0056` ("significant" at α=0.05) where the true value is `p=0.051`
  (**not** significant) — a false-positive significance flip at exactly the
  boundary the test exists to adjudicate, for any REDDITBIAS-scale dataset
  (all five bias types have 234–254 test pairs). Fixed by scaling the erf
  argument by `1/√2`.
- **`run()`'s headline score was Cohen's *d*, a third undocumented
  divergence.** `evaluate()`'s dict had no `bias_score`/`score`/`value` key,
  so `base.py::_split_result`'s fallback search matched `effect_size` first.
  Confirmed by execution: `run().score` returned Cohen's *d* (`10.0` in a
  test case), not the t-value (`inf` in the same case) nor even `evaluate
  (return_details=False)`'s own `mean_diff`. Fixed by adding
  `"bias_score": t_stat` to the dict, so `run()` reports the paper's own
  statistic.
- **Headline statistic for `evaluate(return_details=False)` — still open.**
  The paper reports the **t-value**; `evaluate(return_details=False)` still
  returns `mean_diff`. This one was already known (tracked as
  `REVIEW_LATER` RL-026) and is left as-is: changing it is breaking, and
  `run()` now correctly reports the t-value regardless.
- **Documented/default scoring contract was bidirectional; the paper's
  model is causal.** The class docstring said the callback is "Same as
  AUL's predict function" (full unmasked, bidirectional context) and the
  built-in convenience scorer (`model_name=`) requires a masked-LM
  tokenizer. Barikeri et al. measure LMB on **DialoGPT**
  (`AutoModelForCausalLM`, `model(input_ids, labels=input_ids)` — standard
  causal/left-to-right perplexity) — a model with no mask token, so
  `LMB(model_name="microsoft/DialoGPT-small")` cannot even construct. The
  shipped example used `model_name="bert-base-uncased"` (masked),
  reinforcing exactly the wrong contract. Fixed: the docstring now states
  the causal-conditioning requirement explicitly, and the example was
  rewritten to build a genuine causal scorer around
  `microsoft/DialoGPT-small` (the paper's own model), matching the
  reference's `model(input_ids, labels=input_ids)` computation via one
  forward pass per sentence.

Status stays `adaptation`: the mean_diff-vs-t-value default-scalar gap
(RL-026) is the one remaining, deliberate deviation.

## Validation possible

- **Tier 1:** Ahn & Oh report CB scores per language and model; Barikeri et al.
  report LMB t-values per bias type (Figure 1). Both are reproducible targets.
- **Tier 2:** both repos are MIT and modern-Python friendly — two of the easier
  equivalence cases in the library.
- **Tier 3:** for CBS, null (identical target probabilities → variance 0)
  applies; swap antisymmetry does not, since a variance is symmetric under
  relabelling — document as an exemption. For LMB, null and swap antisymmetry
  both apply (swapping the pair order negates the t-value).

## Known limitations

- CBS's variance is scale-dependent on how many target words are supplied, so
  two CBS numbers over different country lists are not comparable.
- LMB's t-value grows with sample size, so it conflates effect size with
  statistical power — a large corpus produces a large `|t|` for a small bias.
  The paper uses it as a significance indicator, not an effect size.
