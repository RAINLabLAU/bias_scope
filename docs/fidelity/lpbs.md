# LPBS — Log Probability Bias Score

**Cited source:** Kurita, Vyas, Pareek, Black & Tsvetkov 2019, *Measuring Bias in
Contextualized Word Representations*, GeBNLP@ACL 2019.
[arXiv:1906.07337](https://arxiv.org/abs/1906.07337), v1, read as
`sources/papers/lpbs.pdf`.
**Reference implementation:**
[keitakurita/contextual_embedding_bias_measure](https://github.com/keitakurita/contextual_embedding_bias_measure)
@ `18044f87e2ff640fab43941120f14dcf482df6ca` — **no license file** (see
`third_party/LICENSES.md`; its code may be read and run locally but not
vendored).
**Sections and files read:** paper §2 (the four-step procedure and the
definitions of `p_tgt`, `p_prior`, the increased log probability score, and the
log probability bias score); `lib/bias_calculator.py:32-74` (`BiasScorer.bias_score`),
`lib/bias_calculator.py:22-30` (`get_mask_fill_logits`), `lib/bert_utils.py:51-55`
(`tokenize`), `lib/bert_utils.py:72-90` (`get_index`),
`notebooks/construct_bias_score.py:106-140` (the driver, same logic).
**Family / access:** probability / `logits`

## Definition in the paper

For a template such as `[TARGET] is a [ATTRIBUTE]`, §2 gives four steps:

1. Prepare the template.
2. Mask `[TARGET]` only; compute `p_tgt = P([MASK] = [TARGET] | sentence)`.
3. Mask **both** `[TARGET]` and `[ATTRIBUTE]`; compute
   `p_prior = P([MASK] = [TARGET] | sentence)`.
4. The **increased log probability score** is `log(p_tgt / p_prior)`.

> "the difference between the increased log probability scores for two targets
> (e.g. he/she) as log probability bias score which we use as measure of bias"

So, for two targets `t₁`, `t₂`:

```
LPBS(t₁, t₂, attribute) = [log p_tgt(t₁) − log p_prior(t₁)]
                        − [log p_tgt(t₂) − log p_prior(t₂)]
```

Signed and unbounded; **0 means no bias**. `p_prior` exists to cancel the
model's unconditional preference for one target word over the other.

## Definition in the reference code

`bias_calculator.py:51-73` computes the same difference-of-differences, with one
addition the paper does not state: targets are **word sets**, and probabilities
are summed within a set *before* the log.

```python
subject_fill_bias = log(Σ_{m∈male} p(m)) − log(Σ_{f∈female} p(f))          # :51-52
subject_fill_bias_prior_correction = log(Σ_{m} p_prior(m)) − log(Σ_{f} p_prior(f))  # :63-65
return {"bias_prior_corrected": subject_fill_bias − subject_fill_bias_prior_correction}  # :73
```

`get_mask_fill_logits` (`:22-30`) applies a softmax over the vocabulary at one
mask position, so these are probabilities, not logits, despite the name.

### Two discrepancies between the code and the paper

Both concern **which position the prior is read at**, and they compound.

1. **Mask selection.** `bias_score` passes `use_last_mask=not gender_comes_first`
   for the target sentence but `use_last_mask=gender_comes_first` for the
   both-masked sentence (`:49` vs `:61`). With the documented
   `"GGG is XXX"`/`gender_comes_first=True`, the both-masked sentence
   `"[MASK] is [MASK]"` is therefore read at the **last** mask — the attribute
   slot — while the paper's step 3 asks for the probability of the *target* at
   the target slot.

2. **Off-by-one.** `get_index` (`bert_utils.py:82-90`) returns `i + 1` in the
   `not last` branch, "to take the [CLS] token into account", but
   `len(toks) - 1 - i` in the `last` branch, with no such offset.
   `tokenize` (`:51-55`) emits bare wordpieces and `PretrainedBertIndexer`
   prepends `[CLS]` later, so the `last` branch is short by one.

For `"[MASK] is [MASK]"` (`toks = ["[MASK]", "is", "[MASK]"]`) the `last` branch
returns `3 - 1 - 0 = 2`, which in the model input `[CLS] [MASK] is [MASK]` is
the token `is`. The prior is read at a position that holds neither mask.

**Status: `verify`, not settled.** This is a static reading of the code. It has
not been confirmed by execution, because the repo pins `pytorch_pretrained_bert`
and AllenNLP and does not install on a supported Python (Tier 2, Phase 3, is
where that gets attempted; see PLAN.md 6.2 and the obstacle playbook). Until it
runs, treat the above as "the code as written appears to do this", not as an
established fact about the paper's published numbers.

## Current BiasScope implementation (v0.1.1)

`src/bias_scope/probability_based/lpbs.py`. `evaluate` takes
`sentence_pairs: List[Tuple[List[str], List[str]]]` and a `logprob_fn` returning
one score per **whole sentence** (`:143-144`), then:

```python
if score_stereo > score_anti:   # :149
    ...                          # count the pair as stereotype-preferred
bias_score = float(np.mean(pair_scores))   # :157
```

That is the **fraction of pairs where the stereotype sentence scores higher**,
in `[0, 1]`, with **0.5** as the neutral value (docstring `:44-51`).

## Verdict

**faithful**, as of v0.2.0.

**History: v0.1.1 was a `mismatch`,** and this is the defect a reviewer
spot-checked. The audit below is of that version; the resolution is at the end.

The v0.1.1 statistic was not the cited one, on every axis that matters:

| | Kurita et al. | BiasScope v0.1.1 |
|---|---|---|
| Unit scored | one masked token in a template | a whole sentence |
| Uses `p_prior` | yes — it is the paper's contribution | **no prior correction at all** |
| Comparison | two targets in one template | two sentences in a pair |
| Statistic | signed log-ratio difference | proportion of wins |
| Range / neutral | unbounded / 0 | [0, 1] / 0.5 |

Dropping `p_prior` is the substantive part. The paper's whole argument is that
raw `p_tgt` is confounded by the model's unconditional preference for one target
word, and that dividing by `p_prior` removes it. A metric without that term is
not a prior-corrected measure under any reading.

The v0.1.1 statistic is nonetheless a reasonable one — it is essentially the
CrowS-Pairs/StereoSet family's preference rate — so PLAN.md Section 12 keeps it
under a new name rather than deleting it.

## Resolution (v0.2.0)

1. `LPBS` reimplemented to the §2 formula in
   `src/bias_scope/probability_based/lpbs.py`, with target word sets summed
   before the log as the reference code does.
2. The v0.1.1 behaviour preserved unchanged as `PairwiseLikelihoodPreference`
   (`fidelity: original`), per PLAN.md Section 12. It is **not** reachable under
   the `LPBS` name — Section 1 forbids a wrong-metric class keeping its old
   behaviour under the old name.
3. The prior-position discrepancy resolved in favour of the paper; see RL-012.
   `tests/.../test_lpbs_faithful.py::test_prior_is_read_at_the_target_slot`
   pins the choice with a fixture that returns different values at each mask.
4. Known-answer tests derive `2·log 2` by hand; swap antisymmetry and the null
   property are tested directly.

Still open: the Tier-1 reproduction of Table 2, and the Tier-2 comparison that
would settle RL-012 empirically.

## Required action (from the audit; items 1-2 are now done)

1. **Reimplement `LPBS`** to the paper's formula: template-based, per-template
   `log p_tgt − log p_prior` per target, and the difference between two targets.
   Support target word *sets* with sum-then-log, per the reference code.
2. **Move the current behaviour** to `PairwiseLikelihoodPreference`, status
   `original`, with a deprecating alias per PLAN.md Section 1.
3. On the prior-position discrepancy: **implement the paper's step 3** (prior at
   the target slot, both masked) rather than reproducing the reference's
   position. PLAN.md's "the code wins" rule is for cases where the paper is
   ambiguous and the code reveals the authors' intent; here the paper is
   explicit and the code's behaviour looks like an unintended off-by-one that
   the authors would not defend. Expose the reference's position behind a
   keyword argument only if Tier 2 shows it is needed to reproduce Table 2.
   Logged as `REVIEW_LATER` RL-012.

## Validation possible

- **Tier 1:** Kurita et al. Table 2 reports log probability bias scores for
  `bert-base-uncased` on the WEAT-derived attribute sets. Registry entry to be
  filled once the exact cell is read; **do not** anchor to a range.
- **Tier 2:** blocked as written — the repo requires `pytorch_pretrained_bert`
  and AllenNLP. Playbook step 3 (copy the self-contained scoring function into
  `tests/equivalence/reference_snippets/`) is **not available**: the repo states
  no license. Reimplementing `bias_score` from the reading above and comparing
  is possible, but that is an oracle, not an equivalence test.
- **Tier 3:** all five properties apply. Null and swap antisymmetry are
  especially informative here — the paper's LPBS is exactly antisymmetric under
  swapping `t₁` and `t₂`, which the v0.1.1 statistic is not.

## Known limitations of the metric itself

- Template-bound: the paper concedes (§2) that its templates are "merely simple
  sentences", and Section 10.2 of PLAN.md will measure template sensitivity.
- Requires each target to be a single wordpiece, or a policy for multi-piece
  targets that the paper does not give.
- Blodgett et al. 2021's critique of stereotype-pair benchmarks applies to the
  attribute word lists, which are inherited from WEAT.
