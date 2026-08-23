# SEAT — Sentence Encoder Association Test

**Cited source:** May, Wang, Bordia, Bowman & Rudinger 2019, *On Measuring
Social Biases in Sentence Encoders*, NAACL 2019.
[arXiv:1903.10561](https://arxiv.org/abs/1903.10561).
**Reference implementation:**
[W4ngatang/sent-bias](https://github.com/W4ngatang/sent-bias) @
`e3559fb669ca4832743b42fee715994c15c7f1af`, CC-BY-4.0 — the authors' own repo.
**Sections and files read:** §"The Sentence Encoder Association Test" (SEAT as
WEAT applied to sentence vectors, the pooling requirement, the bleached
templates); `sentbias/weat.py:178-192` (shared with WEAT);
`sentbias/encoders/bert.py:15-27` (`encode`).
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

The statistic is WEAT's, unchanged: same `s(w, A, B)`, same effect size, same
one-sided permutation p-value. Only the inputs differ.

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

`src/bias_scope/embeddings_based/seat.py` delegates to `WEAT` with the same
effect size, so it inherits WEAT's `ddof=1` and its permutation test.

**Changed in 0.2.0:** the default is now `pooling="cls"`, matching the
reference's position-0 protocol. It was `"mean"`, which is not what May et al.
did. PLAN.md 4.2 asks for exactly this.

## Verdict

**faithful.**

The statistic is WEAT's and matches. The pooling default now matches the
reference's protocol, and the `[CLS]`-vs-first-token subtlety is documented
rather than silently absorbed.

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
