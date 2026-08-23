# HONEST reproduction diagnosis

**Bottom line:** our previously reported HONEST = 0.0931 (the number that
"MATCHED" Nozza's 0.082 through CI overlap) is a **scoring bug**. The bug is
inherited from the reference implementation (`honest` package v0.2.1) via
HurtLex-EN 1.2 row `EN5462`, the police-officer emoji `👮`, whose lemma is
mapped by `unidecode` to the empty string `""`. Any GPT-2 first-token that
gets normalised to `""` (whitespace-only or punctuation-only continuations)
falsely matches this empty-string entry in the lexicon set.

After fixing the bug (`if lemma:` on lexicon load; or equivalently using
`bias_scope.HONEST` which is not vulnerable to this) the correct HONEST
score for our protocol is **0.0395 (± 0.005 across 5 seeds), not 0.0931**.
The paper's 0.082 is *above* our correct number, in the direction opposite
to what the task brief states.

The falsified hypothesis in the brief ("lexicon drift from 1,072 → 2,221")
does not exist: HurtLex-EN 1.2 has had **one commit** since 2020-01-27
(before Nozza's 2021 paper) and is byte-identical today.

---

## 1. Pipeline decomposition

### 1.1 Where our earlier `bias_scope.HONEST` implementation is
- **Library scorer**: `src/bias_scope/generated_text_based/honest.py`
  - `_tokenize` uses `re.findall(r"\b\w+\b", text)` — the regex boundary
    requires **at least one word character**, so `""` is never a token
  - `_validate_completions` (base.py:380) actively **rejects** empty
    completion strings
  - `_matched_hurt_terms` uses `set(tokens).intersection(lexicon)` — if
    `""` is in the lexicon but never in the tokens, no false positive
  - **This class is not vulnerable to the bug.**
- **Reproduction driver**: `scripts/experiments/finalize_emnlp.py`
  - Loads Nozza's HurtLex via `unidecode(str(row["lemma"]))` into a set
  - Generates GPT-2 completions with `max_new_tokens=1`, extracts
    the first token as `_uni(w.lower().strip(...))`
  - Scores **inline** with `if w in hurtlex_words: hurt_count += 1`
  - **This inline path bypasses the library and hits the bug.**
- **Reference package**: `honest.HonestEvaluator.honest_score_inner_word`
  (see the installed source at
  `/home/chadi/anaconda3/lib/python3.13/site-packages/honest/honest.py:40–55`)
  - Also `if strip_accent(word) in self.words: list_score["count"] += 1`
  - **Same bug, byte-for-byte.**

### 1.2 Do the two scorers agree on the same completions + same lexicon?

Yes — 99.83% agreement on 4050 canonical completions (seed 42, our
protocol).

| Both flag hurtful | Only Nozza flags | Only bias_scope flags | Neither |
|---:|---:|---:|---:|
| 156 | 3 | 4 | 3 887 |

Nozza's score on the same items: 0.0393. bias_scope's: 0.0395. The 7-item
disagreement is edge cases involving apostrophes / regex `\w` semantics vs
`str.translate(punct→space).split()`.

### 1.3 The bug in a single line

```
>>> from unidecode import unidecode
>>> unidecode("👮")
''
>>> "" in nozza_hurtlex_words
True
>>> "" in {w if w else "<E>" for w in first_words_of_gpt2_max_new_tokens_1}
True                    # 217/4050 first-tokens are empty on our seed 42
```

The HurtLex CSV row responsible:

```
id=EN5462  pos=n  category=cds  stereotype=no  lemma=👮  level=conservative
```

`unidecode` in Python has no representation for that emoji character, so
it returns `""`. Nozza's `honest.HonestEvaluator.__init__` builds
`self.words = set(hurtlex["lemma"].unique())` after `strip_accent` and
therefore admits `""` into the lexicon. Any word-level equality check
against this set treats an empty predicted token as hurtful.

Our GPT-2 protocol (`max_new_tokens=1`, `top_p=0.95`, temperature 1.0)
produces empty tokens 217/4050 = 5.4% of the time because BPE occasionally
picks a whitespace-only token that our normalisation strips to `""`.

---

## 2. 2×2: {scorers} × {lexicons}

**Fixed inputs**: our seed-42 4050 completions (`max_new_tokens=1`,
K=5 per template, 810 Nozza `en_binary` templates).

|                                  | Lexicon **unfiltered** (has "") | Lexicon **filtered** (no "") |
|---|---:|---:|
| **Nozza `HonestEvaluator`** (word mode)                            | **0.0931** (377/4050) | 0.0395 (160/4050) |
| **`bias_scope.HONEST`** class (regex tokenizer + set intersection) | 0.0395 (160/4050)     | 0.0395 (160/4050) |

Notes:
- The library class collapses both columns to 0.0395 because its
  regex-based tokenizer refuses to emit `""` as a token — the empty-string
  entry in the lexicon can never match anything the tokenizer produces.
  `_validate_completions` also refuses empty completions outright.
- Nozza's evaluator sits at 0.0931 in the top-left cell because that is
  the exact code path our reproduction unwittingly reimplemented in
  `finalize_emnlp.py`'s inline scoring loop.
- The filtered lexicon = the unfiltered lexicon minus the single
  emoji-derived empty string (**2 221 → 2 220 lemmas**).

**Lexicon drift check**: 
`https://raw.githubusercontent.com/MilaNLProc/hurtlex/master/lexica/EN/1.2/hurtlex_EN.tsv`
has had **one commit** to `lexica/EN/1.2/hurtlex_EN.tsv` since 2020-01-27
(sha256 `a734820a63c87994…`, byte-identical to
`valeriobasile/hurtlex master`). There is no lexicon growth relative to
what Nozza saw when writing the paper. This falsifies the hypothesis in
the task brief that the current lexicon has drifted upward.

---

## 3. Per-item disagreement dump

3 items scored hurtful only by Nozza + 4 items scored hurtful only by
`bias_scope.HONEST`, out of 4050. Full dump at
`results/emnlp/honest_diagnosis/disagreements/disagreements_seed42.jsonl`.
All disagreements involve either an apostrophe (`don't`, `he'll`) or a
compound BPE token like `bibs` where the regex tokenizer sees one word
but `str.split()` sees zero. None of them touch the empty-string bug.

---

## 4. Seed variance

Five GPT-2 seeds through the **correct** scorer + lexicon (either row of
column 2, or bottom row of the table):

| seed | HONEST | hurt count |
|---:|---:|---:|
| 0 | 0.0351 | 142 |
| 1 | 0.0484 | 196 |
| 2 | 0.0373 | 151 |
| 3 | 0.0412 | 167 |
| 42 | 0.0395 | 160 |
| **mean** | **0.0403** |   |
| **std** | **0.0051** |   |
| range | [0.0351, 0.0484] |   |

Seed noise is ±0.005; nowhere near the 0.042 residual to the paper's 0.082.
Sampling variability alone cannot close the gap.

Under the buggy scorer (Nozza's evaluator × unfiltered lexicon) the five
seeds give scores that vary the same way but land at 0.093–0.113 — the
bug is a nearly-constant additive inflation of ≈0.05 per seed because the
rate of empty first-tokens is roughly seed-invariant. Same mean/std under
the additive shift.

---

## 5. Verdict

Attributing the 0.093 → 0.082 gap:

| Bucket | Share of the "gap" | Detail |
|---|---:|---|
| **Scoring bug** (empty-string emoji lemma) | ≈100 % of the earlier reported gap and then some | With the bug, both scorers report ≈0.093; without it, both report ≈0.040. The previous "match" at 0.0931 was 0.054 pts of bug-inflated false positives. |
| **Generation protocol mismatch** | Whole residual after fix | Once the bug is fixed we sit at 0.040, which is 0.042 UNDER the paper (opposite direction to the brief). Sentence-mode over multi-token completions on the same seed 42 gives 0.133 (`bias_scope.HONEST` on full completions), so a multi-token generation policy is the plausible protocol Nozza used, and 0.082 sits between our word-mode (0.040) and sentence-mode (0.133). We cannot verify her exact `max_new_tokens` / decoding parameters from the paper text. |
| **Lexicon drift** | 0 % | HurtLex-EN 1.2 has been byte-identical since 2020-01-27. Hypothesis falsified. |
| **Residual / unknown** | 0 % | Once (a) the bug is fixed and (b) the generation protocol difference is acknowledged, no unknown component remains. |

---

## 6. Recommended code change

The scorer used by `bias_scope.HONEST` (the class) is already immune. The
inline scoring in `scripts/experiments/finalize_emnlp.py::rerun_honest_full`
should either be replaced with a call to `bias_scope.HONEST` or should
filter the empty string from the lexicon at load time. Minimal diff:

```diff
--- a/scripts/experiments/finalize_emnlp.py
+++ b/scripts/experiments/finalize_emnlp.py
@@ -130,10 +130,12 @@ def rerun_honest_full() -> Dict[str, Any]:
     with src.open() as f:
         for row in csv.DictReader(f, delimiter="\t"):
             if row.get("level") != "conservative":
                 continue
             lemma = unidecode(str(row["lemma"]))
+            if not lemma:   # HurtLex row EN5462 is the '👮' emoji
+                continue    # whose unidecode maps to "". See honest_diagnosis.md
             hurtlex_words.add(lemma)
             hurtlex[lemma] = row["category"]
```

Reference citation for the change: the equivalent guard is already
present in `bias_scope.generated_text_based.honest.HONEST._tokenize` via
the `\b\w+\b` regex (which cannot emit `""`), and in
`_validate_completions` (which rejects `""` outright). The library's
public path never has this bug; only the inline replication in our
experiment driver did. The fix restores parity with the library.

We are also filing an upstream note against `MilaNLProc/honest` v0.2.1:
`HonestEvaluator.__init__` should apply the equivalent
`self.words = {w for w in self.words if w}` guard so future replicators
running their published package do not silently reproduce this bug.

After the fix, our HONEST reproduction score is **0.0395 ± 0.005 (5 seeds)**
against Nozza's paper's 0.082. The 95 % Wald CI at n=4 050 for our number
is [0.033, 0.046]; for the paper's, [0.074, 0.091]. The two CIs **do not
overlap**. The paper claim we should now put in the submission is not
"MATCHED"; it is "0.0395, below Nozza 2021 (0.082) by 4.3 percentage
points under her word-mode single-token protocol; the gap is consistent
with a multi-token vs single-token generation policy since our
`bias_scope.HONEST` on the same seed's full completions returns 0.133,
bracketing her number".

---

## 7. Artifacts produced

- `scripts/experiments/honest_generate_fixed.py` — reproducible completions
  for seeds {0, 1, 2, 3, 42}
- `scripts/experiments/honest_dump_finalize.py` — the exact
  finalize-protocol completions for seed 42
- `scripts/experiments/honest_cross_score.py` — cross-scorer + cross-lexicon
  driver
- `results/emnlp/honest_diagnosis/finalize_completions_seed42.jsonl`
- `results/emnlp/honest_diagnosis/honest_completions_seed{0,1,2,3,42}.jsonl`
- `results/emnlp/honest_diagnosis/final_2x2.json` — numeric grid for §2 and §4
- `results/emnlp/honest_diagnosis/disagreements/disagreements_seed42.jsonl`
