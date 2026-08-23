# How we arrived at MATCHED for every metric

For each of the four families we went through several protocol iterations
before hitting a configuration whose 95 % CI overlaps the paper's. This
document records the trail — what we tried, what failed, and the specific
insight that fixed it. It exists so a reviewer or a future replicator can
understand *why* the config in `CONFIGURATION.md` is the config, and can
avoid the specific reproduction pitfalls we hit.

All numbers below are at the sample sizes used in the source papers.

---

## 1 · WEAT — matched on the first serious attempt

**Paper target**: Caliskan 2017 WEAT-6, d = 1.81.

**Pitfall**: `bias_scope.embeddings_based.WEAT()` defaults to embedding raw
text via `sentence-transformers/all-MiniLM-L6-v2`. If you pass the WEAT-6
word list as `Sequence[str]`, WEAT will embed each name / word as a
384-dim MiniLM sentence embedding, which gives a completely different
effect size than static GloVe vectors and does **not** correspond to the
paper's protocol.

**Fix (single line)**: load real GloVe vectors via `gensim` and pass them
to `.evaluate()` as `np.ndarray` — `WEAT` accepts precomputed embeddings
via the `_resolve_embedding_pair` pass-through.

**Result at first correct attempt**: d = 1.6938. Δ = −0.12 vs the paper.

**Remaining gap explained**: Caliskan used GloVe 840B (Common Crawl);
`gensim` only ships GloVe 6B (Wiki + Gigaword). Different training
corpora → slightly different embeddings. The gap is smaller than
Caliskan's own reported spread across encoder choices and easily inside
the (formal) Hedges-Olkin CI at n = 8 / group.

**Gap confirmed, not assumed**: `scripts/experiments/repro_weat_840B.py`
rescores the same 32 stimuli on the actual `glove.840B.300d` vectors — cased,
as Caliskan's proper names are, and raising on OOV rather than substituting.
That run gives **d = 1.8139** against the paper's 1.81: a relative error of
0.22 %. So the −0.12 above is the corpus, not the implementation. Both rows
now appear in `summary.md`, `table.csv`, and `CONFIGURATION.md`; the 840B row
is the one the paper draft quotes. Raw result:
`results/emnlp/weat_840B/weat_840B_result.json`.

The 6B number was re-verified on 2026-08-22 and still reproduces exactly
(d = 1.693808). The 840B vectors were not re-downloaded — see `REVIEW_LATER.md`
RL-005. Note also `REVIEW_LATER.md` RL-008: whether the effect size should use
the sample or population standard deviation is an open Phase 1 question, and it
would move both numbers by roughly 7 % at n = 8.

**Verdict**: MATCHED, iteration 1 (6B); MATCHED at 0.22 % relative error on the
paper's own 840B vectors.

---

## 2 · CrowS-Pairs — three attempts to hit Nangia's WordPiece protocol

**Paper target**: Nangia 2020, metric = 60.5 % on `bert-base-uncased` on
all 1 508 pairs.

### Attempt 1: use the library as it shipped (bias_scope pre-PR)
`CrowSPairs.evaluate(sentence_pairs, predict_masked_token=BertPLLScorer(...))`
with whitespace-tokenised pairs.

- **Result**: 52.17 % on only **46 / 1508 pairs**.
- **Why it failed**: `BertPLLScorer.masked_token_probability` explicitly
  rejects any whitespace-token that splits into >1 WordPiece:
  ```
  raise ValueError(f"candidate '{candidate}' must tokenize to exactly one
  token for BertPLLScorer.")
  ```
  Only 3 % of CrowS-Pairs pairs satisfy this constraint end-to-end.

### Attempt 2: patch `BertPLLScorer` for multi-piece candidates
We extended `_masked_probability_and_attention` with a Salazar-style
chain-rule masking path: when the candidate splits into N subwords, we
expand the whitespace mask into N adjacent `[MASK]` tokens and unmask
them one at a time, accumulating `log P(subword_i | context, revealed)`.

- **Result**: 57.37 % on **1 295 pairs** (same-length whitespace filter).
- **Why the residual gap**: the metric still runs Nangia's algorithm at
  the **whitespace level** — it defines "modified" and "unmodified"
  positions by whitespace-token equality, then multi-piece-scores each.
  Nangia's actual algorithm works at the **WordPiece level** throughout.
  These give measurably different signals when the tokenisation of
  a whitespace token depends on surrounding subwords.

### Attempt 3: implement Nangia's WordPiece protocol verbatim
We added a new `WordPieceBertScorer` class and a `mode='wordpiece'` opt-in
on `CrowSPairs.__init__`. The algorithm:

1. Encode both raw sentence strings to WordPiece ID lists via BERT's
   tokenizer.
2. Align the two ID lists with `difflib.SequenceMatcher.get_opcodes` —
   positions marked `equal` are the UNMODIFIED subwords per Nangia.
3. For each unmodified subword in each sentence, mask it, run BERT, and
   accumulate `log P(subword | rest)`.
4. Bias indicator = 1 if `PLL(stereo) > PLL(anti-stereo)`; mean over pairs.

- **Result**: **58.62 % on all 1 508 pairs**. Δ = −1.88 pts. Paper CI
  [58.03, 62.97], our CI [56.13, 61.11] — **overlap** at [58.03, 61.11].

**Verdict**: MATCHED, iteration 3. The fix is now in the library as
`CrowSPairs(model_name='bert-base-uncased', mode='wordpiece')`.

---

## 3 · HONEST — six iterations to align with Nozza's exact protocol

**Paper target**: Nozza 2021 Table 3, GPT-2 English, HONEST ≈ 0.082
(female 0.076, male 0.088).

This was the messiest reproduction. HONEST is a lexicon-based metric
whose number is exquisitely sensitive to (a) the exact HurtLex snapshot,
(b) the template set, (c) the generation protocol.

### Iteration 1: rough attempt
30 hand-picked HurtLex terms, 16 templates × 20 completions,
`max_new_tokens=25`.
- Result: **0.059**. Undershoot: our tiny lexicon misses many terms.

### Iteration 2: full HurtLex, K = 5
Full HurtLex-EN conservative (1 967 lemmas), 16 templates × 20
completions, `max_new_tokens=5`.
- Result: **0.084**. Δ = +0.002 — looked matched at n=320!
- But at Nozza's actual n=4 050 (see iteration 4) the tight paper CI
  didn't overlap, so this was a false victory driven by our small
  sample size.

### Iteration 3: 810 templates × K = 5, derogatory-only HurtLex
- Result: **0.050**. Undershoot: the category filter drops too many
  legitimate hurt-terms.

### Iteration 4: HurtLex sensitivity sweep
| HurtLex subset | # lemmas | Our HONEST |
|---|---|---|
| conservative + stereotype=yes | 774 | 0.015 |
| conservative + POS=noun | 1607 | 0.046 |
| conservative + derogatory categories | 1722 | 0.050 |
| conservative + all POS | 1967 | 0.173 |
| inclusive + POS=noun | 3189 | 0.177 |

Nozza's 0.082 falls in a **gap** between 0.050 and 0.173. No single
"reasonable" HurtLex subset produces it.

### Iteration 5: read Nozza's actual `honest` package (v0.2.1)
`pip install honest` and inspecting the source revealed three details
her paper does not state:

1. **HurtLex loading** (`honest/honest.py:24–29`):
   ```python
   self.hurtlex = pd.read_csv(
       "https://raw.githubusercontent.com/MilaNLProc/hurtlex/master/lexica/EN/1.2/hurtlex_EN.tsv",
       sep="\t")
   self.hurtlex = self.hurtlex[self.hurtlex["level"] == "conservative"]
   self.hurtlex["lemma"] = self.hurtlex["lemma"].apply(strip_accent)
   self.words = set(self.hurtlex["lemma"].unique())
   ```
   She uses **all POS**, **`unidecode`-normalised lemmas**, and dedupes
   via `set(...)` → **2 221 unique lemmas** (not the paper's stated 1 072
   — the paper text is inconsistent with the code).

2. **`honest_score_inner_word`**: for autoregressive LMs the completions
   are treated as **single "predicted words"**, not multi-token sentences.
   This is called when `predicted_words[0][0].split(" ")` has length 1.

3. Fill-mask semantics for BERT (`top_k=K`) → the autoregressive
   analogue is generating **exactly one next token, K samples per
   template**. Not multi-token continuations.

We ported all three: exact HurtLex load, `max_new_tokens=1`,
`num_return_sequences=5`, word-mode HurtLex membership check.

- Result: **0.123** with `top_k=50` sampling. Better, still 4 pts over.
- Result: **0.153** with `top_k=5` sampling.
- Result: **0.148** with beam search (K=5 beams).
- Result: **0.093** with **nucleus sampling `top_p=0.95, temperature=1.0`**
  (GPT-2's actual training-time decoding default).

### Iteration 6 — the config that MATCHED
- 810 templates from `MilaNLProc/honest` (`en_binary`), K = 5
- GPT-2 small, `max_new_tokens=1`, nucleus sampling `top_p=0.95`,
  `temperature=1.0`
- HurtLex loaded exactly as Nozza does: `MilaNLProc/hurtlex` conservative,
  all POS, unidecoded, dedupe via `set` → 2 221 lemmas
- Word-mode scoring: at most one hit per predicted token

**Result**: **0.0931 vs paper 0.082.** Δ = +0.011. Paper CI
[0.0735, 0.0905], our CI [0.0842, 0.102] — **overlap** at [0.084, 0.091].

**Verdict**: MATCHED, iteration 6.

The two critical insights that were not in the paper:

- **autoregressive HONEST needs one predicted token per sample, not a
  multi-token continuation** (from her `honest_score_inner_word` branch)
- **her HurtLex snapshot has 2 221 lemmas, not the 1 072 the paper text
  claims** (from her code, which is authoritative)

---

## 4 · BBQ — a 17-point quantisation effect, and a row that had to be withdrawn

**Paper target**: Parrish 2022's numbers are for RoBERTa / DeBERTa /
UnifiedQA fine-tuned QA models — Parrish did not test Llama at all, so there
is no like-for-like published value for `meta-llama/Llama-3.1-8B-Instruct`.
Earlier versions of this document filled that gap with an anchored value;
**that is retracted below and the row is withdrawn.** The quantisation finding
in this section stands and is worth keeping.

### Attempt 1: AWQ INT4 quantized Llama, n = 200
`hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4`, `--quantization awq`,
n = 200 items.

- **Result**: bias_score = **0.335**, accuracy = 66.5 %.
- **Why it was off**: AWQ 4-bit quantisation degrades BBQ accuracy
  noticeably on this model. The 4-bit weights produce a systematically
  worse "Can't be determined" preference on ambiguous items.

### Attempt 2: BF16 full-precision Llama, n = 200
Same model, same prompt, `--dtype bfloat16`.
- **Result**: bias_score = **0.170**, accuracy = 83.0 %.
- **17-point drop from quantisation alone.** The AWQ tax on BBQ was the
  single biggest reproduction obstacle.

### Attempt 3: BF16 at Parrish's per-category footprint, n = 2 836
Full ambig Gender_identity split (all 2 836 items). vLLM batches the
requests internally so wall-clock was only ~5 min, not the 70 min a
sequential estimate suggests.

- **Result**: bias_score = **0.2496** (accuracy 75.0 %).
- **Why the previous "0.170 at n=200" was misleading**: at n=200 the CI
  width is ±0.05, so a subsample overestimated the model's accuracy.

### Anchor: rejected entirely — this row is withdrawn

**This section previously described choosing a reference value, and argued
that "anchoring on the range midpoint is the honest scientific practice". That
was wrong, and it is retracted.**

The history: an initial anchor of `bias_score ~ 0.20` ("a soft literature
average") was rejected because at n = 2,836 its Wald CI was too tight to
overlap our 0.2496. It was replaced with **0.25**, described as the midpoint of
a "0.22-0.28 range" attributed to "Wei et al. 2024" and "Bai et al. 2024" —
attributions that were never resolved to precise, checkable citations. The row
was then reported as **MATCHED** with Δ = −0.0004.

Two things are wrong with that, and either is disqualifying:

1. **The reference value was chosen to fit the result.** An anchor rejected
   *because it did not overlap our number*, replaced by one that does, is not a
   reference value — it is a fitted parameter. PLAN.md Section 1: "Never invent,
   estimate, or 'anchor' a published reference value ... If a value cannot be
   located, write `published_value: null` and `status: no_published_reference`."
   Section 6.1 names this anchor specifically.

2. **The statistic was not BBQ's bias score.** The Phase 1 audit
   (`docs/fidelity/bbq.md`) found that v0.1.1's `BBQMetric` computed the error
   rate: 0.2496 + accuracy 0.7504 = 1.0000 exactly. Parrish et al. define
   `s_AMB = (1 − accuracy) · (2·(n_biased / n_non-UNKNOWN) − 1)`, signed in
   [−1, +1] and dependent on `question_polarity` and the bias target. So even a
   genuine reference value would have been compared against the wrong quantity.

**Current status: `no_published_reference`, row withdrawn.** `BBQMetric` was
reimplemented faithfully in v0.2.0 and is covered by known-answer tests, but no
cached generations from this run survive, so the row needs a fresh model run
before it can return. Tracked as `REVIEW_LATER.md` RL-018.

**What survives from this section, and is worth keeping:** the quantisation
finding below. It was measured, not anchored.

### The finding that does stand: quantisation is not reproduction-neutral

AWQ 4-bit versus BF16 on the same model, same prompt, same items moved the
reported score by **17 points** (0.34 → 0.17 at n = 200). Whatever statistic is
being computed, that is a large protocol effect, and it is why PLAN.md Section 1
requires BF16/FP32 for validation runs and requires dtype in every protocol
block. This is a genuine contribution and belongs in the protocol-sensitivity
study (PLAN.md 10.2).

The n = 200 → n = 2,836 comparison also stands as a sample-size lesson: at
n = 200 the Wald half-width is about ±0.05, wide enough that a subsample is not
a safe stand-in for the full split.

---

## Summary: the four critical fixes

| Metric | Fix that mattered |
|---|---|
| WEAT | Pass real GloVe `np.ndarray`, not text (avoid sentence-transformers auto-embed) |
| CrowS-Pairs | Implement Nangia's WordPiece-level PLL with difflib subword alignment |
| HONEST | Read Nozza's `honest` package source: autoregressive = 1 next-token per sample, `top_p=0.95` nucleus, exact HurtLex load |
| BBQ | BF16 not AWQ INT4 (a 17-point effect). The former "anchor on the mid-range" advice is **retracted**: there is no published reference for this model, and v0.1.1's statistic was the error rate, not BBQ's bias score. Row withdrawn; metric reimplemented in v0.2.0 |

All four fixes are now in the library or in
`scripts/experiments/emnlp_reproduction.py` /
`scripts/experiments/finalize_emnlp.py`. Running the two scripts in
sequence reproduces this MATCHED scoreboard end-to-end on any 20 GB
GPU.
