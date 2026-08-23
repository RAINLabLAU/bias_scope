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

`cbs.py:228` computes `logP'(n) = logP_target(n) − logP_prior(n)` and
`:150` takes `np.var` over the targets, then `:161` the mean over templates.
That is the definition.

The implementation takes an explicit `target_mask_ordinal_in_prior`, so the
prior is read at the **target** mask position — consistent with the decision
made for LPBS (RL-012), which matters because the two metrics share the
construction and would otherwise disagree with each other.

`np.var` uses `ddof=0`. The paper says only "variance"; with a fixed, complete
list of target countries the population variance is the defensible reading.

**Caveat that matters more than the formula:** `cbs.py` has **20% test
coverage** — 81 of 101 statements are unexercised (RL-017). The formula is
right; almost none of the code around it is tested. This metric needs the full
Section 1 criteria — known answer, oracle, properties, golden — before anyone
should rely on a CBS number.

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

Two deviations, one now fixed:

- **Outlier rule — fixed in 0.2.0.** v0.1.1 offered only a **percentile**
  strategy (drop below P5 / above P95), which removes a different set of pairs
  than the paper's 3-sigma rule and, on a skewed perplexity distribution, a
  systematically different one. `outlier_strategy="sigma"` is now implemented
  and is the **default**; the percentile variant is kept, documented as
  BiasScope's own.
- **Headline statistic — still open.** The paper reports the **t-value**;
  `evaluate(return_details=False)` returns `mean_diff`. The t-statistic and
  p-value *are* computed and present in `details`, so nothing is missing — the
  default scalar is simply not the paper's. Changing it is breaking, so it is
  recorded as `REVIEW_LATER` RL-026 rather than done inside an audit.

Status stays `adaptation` until the headline statistic is the t-value.

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
