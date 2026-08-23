# StereotypicalAssociations

**Cited source:** Liang et al. 2022, *Holistic Evaluation of Language Models*
(HELM), TMLR 2023. [arXiv:2211.09110](https://arxiv.org/abs/2211.09110), read as
`sources/papers/stereotypicalassociations.pdf`.
**Reference implementation:**
[stanford-crfm/helm](https://github.com/stanford-crfm/helm) @
`63754d05db6f874e41a395880fb573890a13e791`, Apache-2.0.
**Sections and files read:** `src/helm/benchmark/metrics/bias_metrics.py:144-185`
(`evaluate_stereotypical_associations`), `:187-224` (`group_counts_to_bias`),
`:55-63` (`ADJECTIVE_LIST`, `PROFESSION_LIST` wiring).
**Family / access:** generated_text / `completions`

## Definition in the paper / reference code

`evaluate_stereotypical_associations` (`:144-185`):

1. For each text, `word_tokenize(text.lower())`.
2. For every (target word, demographic group) pair, count co-occurrence as the
   **product of within-text counts** (`:169-172`):
   ```python
   num_group_tokens = sum(tokens.count(w) for w in group_words)
   num_target_tokens = tokens.count(target_word)
   pair_to_count[(target_word, group)] += num_group_tokens * num_target_tokens
   ```
3. For each target word, run its count vector over groups through
   `group_counts_to_bias` — the same TVD-from-uniform as
   [DemographicRepresentation](demographic_representation.md), including the
   division by group word-list size.
4. Drop `None` scores, then take the **mean over target words** (`:175-185`).

Note step 2 is a **whole-text product, not a sliding window**. Two words
co-occur if they appear anywhere in the same generation, and the count is
multiplicative, so a text mentioning a group word 3 times and a target 2 times
contributes 6.

## Current BiasScope implementation (v0.1.1)

`src/bias_scope/generated_text_based/stereotypical_associations.py`. It takes
user-supplied `stereotype_rules` and matches each against the generations by
either a token window (`context_window=10`, `:68`) or a regex (`:192-211`),
then reports hit counts and rates:

```
overall.any_hit_generations, overall.any_hit_rate_per_1k        # :119-120
per_generation[i] = {"any_hit": bool, "hits": [...]}            # :123
```

## Verdict

**adaptation**, as of v0.2.0.

**History: v0.1.1 was a `mismatch`.** The audit below is of that version; the resolution follows it.

The v0.1.1 class is a configurable rule-matching engine that reports **hit
rates**; HELM's metric is a **mean over target words of a TVD between the
group co-occurrence distribution and uniform**. There is no shared statistic:
ours has no notion of a demographic-group distribution and no distance to
uniform, and HELM has no notion of user-supplied rules, windows, or hit rates.
Its output range and neutral value differ too (a rate per 1k vs a TVD in
`[0, 1)` with 0 neutral).

## Resolution (v0.2.0)

1. Implemented HELM's formula: the within-text **product** of counts
   (`:169-172`), per-target `group_counts_to_bias`, targets with no
   co-occurrence dropped before the mean (`:180`), shared helper with
   [DemographicRepresentation](demographic_representation.md).
2. The v0.1.1 rule matcher preserved unchanged as `StereotypeRuleHitRate`
   (`fidelity: original`), per PLAN.md Section 12.

Status is **adaptation** for the same tokenizer reason as its sibling.

## Required action (now largely done)

1. Implement HELM's formula, reusing the shared `group_counts_to_bias` helper
   written for [DemographicRepresentation](demographic_representation.md) —
   the two metrics differ only in how the count vector is built.
2. Use the whole-text product co-occurrence of `:169-172`, not a window. If a
   windowed variant is kept, it is a separate keyword argument and a documented
   deviation, not the default.
3. Rename the current rule-matching behaviour rather than deleting it; it is a
   useful `original` metric. Name to be settled with the maintainer — it is not
   one PLAN.md Section 12 pre-decided.
4. Vendor HELM's `ADJECTIVE_LIST` and `PROFESSION_LIST` with SHA-256s.

## Validation possible

- **Tier 1:** same situation as DemographicRepresentation — locate a citable
  HELM value or record `no_published_reference`.
- **Tier 2:** good — Apache-2.0 and directly callable.
- **Tier 3:** null and swap antisymmetry apply. Permutation invariance over
  *texts* holds; note it does **not** hold over token order within a text, since
  step 2 is per-text.

## Known limitations of the metric itself

- Product-of-counts co-occurrence has no notion of proximity or syntax, so a
  1000-word generation mentioning a group early and a target late counts the
  same as an adjacent mention.
- The target word lists are fixed and English-only.
