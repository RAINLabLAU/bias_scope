# SEAT — Sentence Encoder Association Test

**Cited source:** May, Wang, Bordia, Bowman & Rudinger 2019, *On Measuring
Social Biases in Sentence Encoders*, NAACL 2019.
[arXiv:1903.10561](https://arxiv.org/abs/1903.10561).
**Reference implementation:**
[W4ngatang/sent-bias](https://github.com/W4ngatang/sent-bias) @
`e3559fb669ca4832743b42fee715994c15c7f1af`, CC-BY-4.0 — the authors' own repo.
**Sections and files read:** §"The Sentence Encoder Association Test" (SEAT as
WEAT applied to sentence vectors, the pooling requirement, the bleached
templates); **Appendix A** (p-value and effect-size computation);
Appendix C / Table 3 (per-encoder pooling); `sentbias/weat.py:178-192` (shared
effect size), `sentbias/weat.py::p_val_permutation_test` (`>=` convention,
99,999 + 1 sampling); `sentbias/encoders/bert.py:15-27` (`encode`).
**Family / access:** embedding / `embeddings`

## Definition in the paper

> SEAT compares sets of sentences, rather than sets of words, by applying WEAT
> to the vector representation of a sentence. Because SEAT operates on
> fixed-sized vectors and some encoders produce variable-length vector
> sequences, **we use pooling as needed** to aggregate outputs into a
> fixed-sized vector. We can view WEAT as a special case of SEAT in which the
> sentence is a single word.

Words are slotted into "semantically bleached" templates — `"This is <word>."`,
`"<word> is here."`, `"This will <word>."`, `"<word> are things."` — chosen to
"convey little specific meaning beyond that of the terms inserted into them".
Each encoder in Table 3 has its own pooling (CBoW mean, InferSent max, GenSen
last, USE native, ELMo mean+layer-sum, GPT last token, **BERT `[CLS]`**).

The **effect size** is WEAT's, unchanged: same `s(w, A, B)`, same standardised
mean difference with an unbiased (`ddof=1`) standard deviation. Appendix A:
"we compute the effect size identically".

The **permutation p-value differs from Caliskan's**. Appendix A, verbatim:

> in our nonparametric version, the equality has positive probability, so we
> implement the more conservative non-strict inequality:
> `Pr[s(Xi, Yi, A, B) ≥ s(X, Y, A, B)]`.

and, for the sampled case, "we sample 99,999 partitions uniformly with
replacement and hallucinate that one more partition satisfied the inequality
… when sampling, we can never observe a p-value less than 1e-5".

## Definition in the reference code

`encoders/bert.py:15-27`:

```python
tokenized = tokenizer.tokenize(text)
indexed   = tokenizer.convert_tokens_to_ids(tokenized)
enc, _    = model(tokens_tensor, segments_tensor, output_all_encoded_layers=False)
enc = enc[:, 0, :]  # extract the last rep of the first input
```

So the reference pools by taking the **position-0** representation.

**A detail that matters for Tier 2:** `tokenizer.tokenize` in
`pytorch_pretrained_bert` does **not** add `[CLS]`/`[SEP]`, and the code never
adds them, so position 0 is the first *content* wordpiece — not a `[CLS]` token.
BiasScope's `pooling="cls"` uses the HF tokenizer, which does add `[CLS]`, so
its position 0 is the real `[CLS]`. Both are "position 0", of different
sequences. Any Tier-2 comparison must account for this or it will report a
spurious disagreement.

## Current BiasScope implementation

`src/bias_scope/embeddings_based/seat.py` delegates the mathematics to `WEAT`,
so it inherits WEAT's `s(w, A, B)`, effect size, and `ddof=1`.

**Permutation convention (fixed after the 2026-09 audit):** `SEAT.evaluate` now
delegates with `tie_policy="conservative"` (WEAT's name for the non-strict
`>=`) and `n_permutation_samples=100_000` by default, reproducing Appendix A:

- exact enumeration when there are ≤ 100,000 equal-size partitions;
- otherwise the conservative branch draws 99,999 partitions and counts the
  observed one, so the p-value is floored at 1e-5;
- `tie_policy="strict"` is still selectable for Caliskan's `>` convention.

`SEAT.run` threads and records `permutation_seed` the same way `WEAT.run` does.

**Pooling:** the default is `pooling="cls"` (matches the BERT row of Table 3).
This is a BiasScope convenience for raw Hugging Face encoders — the reference's
position 0 is its first content wordpiece (`pytorch_pretrained_bert` adds no
special tokens), while BiasScope's is the real `[CLS]`. BiasScope does **not**
reproduce the full per-encoder pooling table, and the default model
(`all-MiniLM-L6-v2`, a mean-pooling model) is a poor match for `cls`. Exact
reproductions should pass precomputed sentence embeddings from the intended
encoder protocol.

## Verdict

**faithful for the scoring statistic (effect size and p-value); the bleached
templates and per-encoder pooling table are not reproduced.**

The effect size matches WEAT/Caliskan exactly; the p-value now matches May et
al.'s Appendix A convention. What SEAT does not do is build the sentence
stimuli or match each encoder's pooling — the caller supplies sentence
embeddings, and `MetricInfo.deviation_note` says so.

## Required action

None outstanding. Note for Phase 3: the bleached templates are not vendored, so
a caller currently supplies their own sentences. Vendoring May et al.'s
templates with a SHA-256 would make the Tier-1 reproduction one call.

## Validation possible

- **Tier 1:** May et al. report per-test effect sizes and p-values for ELMo,
  BERT, GPT and others. Good targets, though the encoders are dated.
- **Tier 2:** the repo is CC-BY-4.0 but pins `pytorch_pretrained_bert`; expect
  the obstacle playbook. The scoring function is shared with WEAT and is the
  easy part — the encoder is where the versions bite.
- **Tier 3:** as WEAT — all five properties apply.

## Known limitations of the metric itself

- May et al. themselves conclude SEAT can confirm the presence of bias but is
  a weak instrument for its absence; the paper says so directly.
- Bleached templates still carry some meaning, and the choice of template moves
  the result — a Section 10.2 sensitivity target.
