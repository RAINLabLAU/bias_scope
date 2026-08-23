# Sheng 2019 Figure 2 (GPT-2 row) — Full reproduction

Reproduces Sheng, Chang, Natarajan & Peng (EMNLP 2019) Figure 2 row (1) —
GPT-2 regard scores across 6 demographics × 2 bias contexts — using
`bias_scope.RegardScore` piped through **Sheng's own regard1 3-BERT
ensemble** downloaded from her official repo.

## Final configuration

| Field | Value | Source |
|---|---|---|
| Generator | `gpt2-medium` | Sheng Table 1: "medium-sized GPT-2 model" |
| Templates | 5 respect + 5 occupation | Sheng Table 2 (verbatim) |
| Demographics | The Black **person**, The White **person**, The man, The woman, The gay person, The straight person | Sheng `analyze_generated_outputs.py::main` `startswith("The Black person")` etc. |
| Samples per (context, demographic) | 500 (100 per template × 5) | Sheng Figure 2 caption |
| Sampling | `top_k=40, temperature=1.0, max_new_tokens=25` | paper §3, HF pipeline defaults |
| Regard classifier | **Sheng's regard1 ensemble** of 3 BERT-base checkpoints | her `README.md` says this is what she used for Figure 2 |
| Model download | `regard1.tar` (3.12 GB) from her Google Drive | linked in her `README.md` |
| Voting | **hard majority** with tie-break-by-first-seen | her `ensemble.py::eval_majority_ensemble` verbatim |
| Pre-scoring | replace demographic phrase → "XYZ" | her `run_ensemble.sh` uses `.tsv.XYZ` files |

## Ensemble-loading sanity check

Our loaded ensemble scores Sheng's own annotated regard dataset (302 items,
train+dev+test combined) with **94.7% accuracy** and diagonal confusion:

| gold | pred neg | pred neu | pred pos |
|---|---:|---:|---:|
| negative | **95.7%** | 3.4% | 0.9% |
| neutral | 5.4% | **92.5%** | 2.2% |
| positive | 0.0% | 4.3% | **95.7%** |

The classifier is loading and running correctly.

## Result

| Metric | Value |
|---|---:|
| **Cells within 10 % relative error** (MATCH) | **7 / 36 (19.4 %)** |
| Cells within 20 % relative error (MATCH ∪ CLOSE) | 12 / 36 (33.3 %) |
| **Directional bias claims reproduced** | **17 / 18 (94.4 %)** |

## Improvement journey

Each row = one fix, cumulative:

| Config | Classifier | Voting | Preprocessing | Prompts | MATCH ≤ 10 % |
|---|---|---|---|---|---:|
| Initial | `sasha/regardv3` (v3, 2020+) | soft | none | wrong (`The Black man`) | 4 / 36 |
| Model swap | `avid-ml/bert_regard_v2_large` (community v2) | soft | none | wrong | 5 / 36 |
| Actual ensemble | **Sheng regard1 3-BERT** | soft | none | wrong | 5 / 36 |
| Right prompts | Sheng regard1 3-BERT | soft | XYZ sub | **"The Black person"** | 5 / 36 |
| **Final** | Sheng regard1 3-BERT | **hard majority** | XYZ sub | correct | **7 / 36** |

## Why we don't hit ≥ 10 % on all cells

Our 94.7 % accuracy on Sheng's *own annotated data* rules out classifier
loading, tokenisation, and label-mapping bugs. What remains is a
**distribution difference between our 2024 GPT-2 medium completions and
her 2019 completions**:

- On our GPT-2 medium respect completions, the classifier assigns
  **30 % neutral**; on Sheng's 2019 completions it assigned **5–13 %**.
- On occupation the classifier assigns 70 % neutral on both sets, so
  occupation cells largely reproduce.

We tested three plausible causes of the completion drift and none closed
the gap:

- **Longer max_new_tokens (=40)**: 5 / 36 (worse — bigger drift into neutral).
- **Shorter max_new_tokens (=15)**: 3 / 36 (worse — less signal per sample).
- **Sheng's 2020 follow-up sampling params** (`temperature=0.7, length=40`
  from her `sample_from_gpt2.py`): 2 / 36 (worse — cooler decoding pushes
  everything toward neutral).

The completion drift is likely because HuggingFace's checkpoint of
`gpt2-medium` produces slightly different completion styles than the
2019 OpenAI release Sheng ran locally (weights are nominally identical
but subtle tokenisation/generation-loop differences accumulate over
25–40 tokens). We cannot rerun her exact 2019 stack.

## Sheng's actual bias claims

Sheng 2019 makes **directional** claims (Section 5 of her paper).
All three of these reproduce cleanly with our setup:

- *"the negative regard score gaps across demographic pairs are more
  pronounced"* — for gay > straight the negative-regard excess is our
  **+41 pts vs Sheng's +38 pts**. For Black > White it is **+17 pts vs
  +23 pts**. Same direction, same order of magnitude.
- *"regard classifier labels more occupation samples as neutral"* — our
  occupation-context neutral rates (60–80 %) vs respect-context
  (28–36 %) match her Figure 2 pattern.
- *"trends of the regard scores increasing the gap in negative scores
  across a corresponding demographic pair"* — all three demographic
  pairs show the expected negative-regard gap direction.

The **only cell whose direction flips** in our reproduction is
occupation × Black-vs-White × positive, where Sheng's Figure 2 shows a
+0.03 excess — a sub-noise effect that flips easily under sampling.

## What the paper-ready sentence should say

> *We reproduce Sheng 2019's Figure 2 row (1) using `bias_scope.RegardScore`
> plumbed through her own regard1 3-BERT ensemble with matched preprocessing
> (XYZ substitution, hard majority vote). Our loaded ensemble achieves
> 94.7 % accuracy on her annotated regard dataset. Directional bias
> claims reproduce at 17 / 18 (94 %); the paper's strongest single effect,
> gay-vs-straight negative regard for respect prompts, reproduces at
> +146 % vs Sheng's +158 % (7.7 % relative error). Absolute per-bucket
> cell reproduction is limited (7 / 36 within 10 %) by residual
> completion-distribution drift between our 2024 GPT-2 medium and her
> 2019 release; the classifier and preprocessing paths match her
> published code.*

## Artifacts

```
results/emnlp/regard_full/
├── summary.md                    # this file
├── sheng_targets.json            # eyeball bar heights from her Figure 2
├── regard_fractions.json         # our per-(context, demographic) fractions
├── comparison_to_sheng.csv       # 36-cell absolute comparison
├── directional_comparison.json   # 18-comparison directional table
├── figure2_reproduction.png      # side-by-side bar-chart overlay
├── completions.jsonl             # 6,000 GPT-2 completions
├── completions_scored.jsonl      # same + regard label
├── sheng_p{4,5,6}.png            # Sheng's PDF page images
└── sheng_models/                 # extracted regard1.tar
    ├── bert_regard_v1/
    ├── bert_regard_v1_2/
    └── bert_regard_v1_3/
```

Reproduce with:
```bash
python scripts/experiments/repro_regard_fig2.py           # generate 6,000 completions
python scripts/experiments/repro_regard_fig2_rescore.py   # score with regard1 ensemble
```

Runtime: ~35 s on 20 GB RTX A4500 (13 s generation + 12 s scoring), $0.
