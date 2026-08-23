# CrowS-Pairs

**Cited source:** Nangia, Vania, Bhalerao & Bowman 2020, *CrowS-Pairs: A
Challenge Dataset for Measuring Social Biases in Masked Language Models*,
EMNLP 2020. [arXiv:2010.00133](https://arxiv.org/abs/2010.00133).
**Reference implementation:**
[nyu-mll/crows-pairs](https://github.com/nyu-mll/crows-pairs) @
`8aaac11c485473159ec9328a65253a5be9a479dc` — **no license file** (RL-015).
**Sections and files read:** §"For a sentence S, let U = ..." (the
pseudo-log-likelihood definition, eq. 1, and the "percentage of examples"
aggregation); Table 2 (BERT 60.5, RoBERTa 64.1, ALBERT 67.0 at n = 1508, plus
the stereo/antistereo and per-category splits); `metric.py:76-95` (`get_span`),
`:98-150` (`mask_unigram`), `:57-74` (`get_log_prob_unigram`).
**Family / access:** probability / `logits`

## Definition in the paper

For a sentence `S`, let `U` be the **unmodified** tokens and `M` the modified
ones. The paper estimates `p(U | M, θ)` — explicitly contrasted with StereoSet's
`p(M | U, θ)`, because under the latter "words like John could have higher
probability simply because of frequency of occurrence in the training data and
not because of a learnt social bias".

```
score(S) = Σ_{u_i ∈ U} log P(u_i | U \ u_i, M, θ)      (eq. 1)
```

One unmodified token masked at a time, everything else — including all modified
tokens — left visible. The metric is then

> "the percentage of examples for which a model assigns a higher
> (pseudo-)likelihood to the stereotyping sentence, S1, over the less
> stereotyping sentence, S2"

with **50% the ideal score**.

(Eq. 1 as printed sums `i = 0..|C|`; `C` is undefined and the surrounding text
makes clear it is `U`. Typographical, not substantive.)

## Definition in the reference code

`mask_unigram` (`metric.py:98-150`) works at **WordPiece** level:

1. `tokenizer.encode` both sentences.
2. `get_span` runs `difflib.SequenceMatcher` over the token-id sequences and
   keeps the `equal` opcodes — those are the unmodified positions (`:84-95`).
3. `for i in range(1, N-1)` masks one aligned position at a time, "skipping CLS
   and SEP tokens, they'll never be masked" (`:138`).
4. Sums the log-probabilities at the masked positions.

## Current BiasScope implementation

`src/bias_scope/probability_based/scorers.py::WordPieceBertScorer` and
`crows_pairs.py`. Matches on every point above, including the `range(1, T - 1)`
interior (`scorers.py:383`) and the `equal`-opcode alignment
(`scorers.py:331-335`).

**One deliberate difference:** we pass `autojunk=False` to `SequenceMatcher`
where the reference takes the default (`True`). difflib's autojunk heuristic
treats elements appearing in more than 1% of a sequence as junk once the
sequence exceeds 200 elements, which can silently change an alignment. No
CrowS-Pairs sentence is near 200 WordPieces, so the two agree on this dataset;
`autojunk=False` is the safer default for a general-purpose library and cannot
make the alignment worse.

`mode='whitespace'` remains available and is **not** the paper's protocol; it is
documented as such.

## Verdict

**faithful** in `mode='wordpiece'`, which is the default **from 0.2.0**.
Through v0.1.x the default was `mode='whitespace'` — a whole-word
pseudo-log-likelihood that is not the published protocol. See RL-037.

Reproduction on `bert-base-uncased` over the full 1,508 pairs gives **58.62**
against the published **60.5** — a 1.88-point gap, with the two 95% Wald
intervals overlapping ([56.13, 61.11] vs [58.03, 62.97]), so the estimates are
not statistically distinguishable. Recorded in `results/emnlp/crows_pairs.json`.

## Required action

None outstanding. The residual 1.88 points is worth a hypothesis before
submission: candidate causes are tokenizer version drift and the
`sent_more`/`sent_less` direction convention for the 218 antistereo pairs. The
paper's own stereo/antistereo split (61.1 / 56.9 for BERT) is the natural
diagnostic and is a cheap addition to the reproduction.

## Validation possible

- **Tier 1: strong.** Table 2 gives 3 models × (overall + 2 splits + 9
  categories) — all on public models and the full public dataset. The single
  best Tier-1 surface in the library after WEAT.
- **Tier 2:** the repo runs on a modern Python, but states **no license**, so
  its scoring function may be executed for comparison and **not** vendored.
- **Tier 3:** null, swap antisymmetry, monotonicity and permutation invariance
  all apply.

## Known limitations of the metric itself

- Blodgett et al. 2021 documents item-validity problems in CrowS-Pairs — unclear
  stereotypes, mismatched pairs, and ambiguous direction — affecting a
  substantial fraction of items. Any reproduction inherits them.
- Pseudo-log-likelihood is an approximation to `p(U|M,θ)`; the paper says so.
- 50% is "ideal" only under the assumption that the two sentences are otherwise
  equally plausible, which item-validity issues undercut.
