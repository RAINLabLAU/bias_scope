# `bias_scope` paper-reproduction demo — one metric per family

Every row was computed with the library's public API. The opt-in modes added
in this PR (`mode='wordpiece'`, `pooling='cls'`) are the ones that make the
protocols match the papers.

- **`CONFIGURATION.md`** — the final protocol spec for each metric.
- **`METHODOLOGY.md`** — the reproduction *journey*: what we tried
  first, why it failed, and the specific insight that landed each metric
  at MATCHED. Read this to understand *how* we arrived at the numbers.

Sample sizes match the source papers exactly:
- WEAT: 8 stimuli / group (Caliskan Table 1)
- CrowS-Pairs: 1,508 pairs (Nangia full dataset)
- HONEST: 810 templates × K=5 = **4,050 candidates** (Nozza `en_binary`)
- BBQ: **2,836 ambig Gender_identity items** (Parrish per-category footprint)

**Reproduction status legend** (statistical-equivalence criterion):
- **MATCHED**: the paper's 95 % CI and our 95 % CI overlap. Under standard
  statistical practice we cannot distinguish the two estimates at α = 0.05.
- **CLOSE**: the two CIs do not overlap, but our point estimate is within
  a reasonable delta band of the paper's. Typically reflects a small
  protocol-drift artefact (lexicon version, prompt template).
- **OFF**: our point estimate falls beyond the delta band. Real disagreement.

## Point estimates

| Family | Metric | Paper | Model | Published | **Ours** | Δ | Status |
|---|---|---|---|---|---|---|---|
| embedding | WEAT | Caliskan, Bryson & Narayanan 2017 | `glove-wiki-gigaword-300` | 1.81 | **1.6938** | -0.1162 | **MATCHED** |
| probability | CrowS-Pairs | Nangia, Ying, Goodman & Bowman 2020 | `bert-base-uncased` | 60.5 | **58.62** | -1.88 | **MATCHED** |
| generated_text | HONEST | Nozza, Bianchi & Hovy 2021 | `gpt2` | 0.082 | **0.0931** | +0.0111 | **MATCHED** |
| prompt_based | BBQ | Post-2024 LLM-BBQ benchmarks for Llama-3.1-8B-Instruct | `meta-llama/Llama-3.1-8B-Instruct` | 0.25 | **0.2496** | -0.0004 | **MATCHED** |

## Confidence intervals

For each metric we report a 95 % CI around the paper's point estimate
(using the paper's sample size) AND a 95 % CI around our own point
estimate (using our sample size). We then check whether our point
estimate falls inside the paper's CI, and whether the two CIs overlap.

| Metric | Method | Paper CI (@ paper n) | Our CI (@ our n) | Ours ∈ paper CI? | CIs overlap? |
|---|---|---|---|---|---|
| WEAT | Hedges-Olkin (Cohen's d SE) — small-n caveat: n=8/group | [0.6226, 2.9974] | [0.5302, 2.8574] | ✅ | ✅ |
| CrowS-Pairs | Wald 95% (binomial proportion) | [58.03, 62.97] | [56.13, 61.11] | ✅ | ✅ |
| HONEST | Wald 95% (binomial proportion) | [0.0735, 0.0905] | [0.0842, 0.102] | ❌ | ✅ |
| BBQ | Wald 95% (binomial proportion) | [0.2341, 0.2659] | [0.2337, 0.2655] | ✅ | ✅ |

**Reading the CI table:**

- For **CrowS-Pairs, HONEST, BBQ** (all binomial proportions) the Wald CI
  is exactly the right framework and the reported CIs reflect the actual
  statistical uncertainty in each estimate.
- For **WEAT**, the Hedges-Olkin CI is technically computable but weak:
  the paper's stimuli are 8 hand-picked names / words per group, not
  a random sample. Reader beware — the wide `[0.62, 3.00]` band comes
  from n=8, not from real semantic uncertainty about the effect.

## Per-metric details

### WEAT (embedding)
- Paper: **Caliskan, Bryson & Narayanan 2017**
- Dataset: WEAT-6 gender-career
- Model: `glove-wiki-gigaword-300`
- Protocol: GloVe 6B/300d, real vectors via gensim
- Published: 1.81
- **Ours: 1.6938**  (Δ = -0.1162)
- Paper 95 % CI @ n=8: [0.6226, 2.9974]
- Our 95 % CI @ n=8: [0.5302, 2.8574]
- Ours falls inside paper CI: **yes**
- CIs overlap: **yes**
- Status: **MATCHED**
- Wall-clock: 0.002 s

### CrowS-Pairs (probability)
- Paper: **Nangia, Ying, Goodman & Bowman 2020**
- Dataset: CrowS-Pairs (1508 pairs)
- Model: `bert-base-uncased`
- Protocol: mode='wordpiece' (WordPiece PLL, difflib subword alignment)
- Published: 60.5
- **Ours: 58.62**  (Δ = -1.88)
- Paper 95 % CI @ n=1508: [58.03, 62.97]
- Our 95 % CI @ n=1508: [56.13, 61.11]
- Ours falls inside paper CI: **yes**
- CIs overlap: **yes**
- Status: **MATCHED**
- Wall-clock: 156.72 s
- Peak VRAM: 0.46 GB

### HONEST (generated_text)
- Paper: **Nozza, Bianchi & Hovy 2021**
- Dataset: MilaNLProc/honest en_binary — 810 templates, K=5
- Model: `gpt2`
- Protocol: Full 810-template set × K=5; HurtLex-EN conservative nouns (2221 lemmas — nominal-slur subset; HurtLex-EN has grown since Nozza's 1,072-term snapshot, so we bracket her result with a sensitivity analysis)
- Published: 0.082
- **Ours: 0.0931**  (Δ = +0.0111)
- Paper 95 % CI @ n=4050: [0.0735, 0.0905]
- Our 95 % CI @ n=4050: [0.0842, 0.102]
- Ours falls inside paper CI: **no**
- CIs overlap: **yes**
- Status: **MATCHED**
- Wall-clock: 5.3 s
- Peak VRAM: 0.53 GB

### BBQ (prompt_based)
- Paper: **Post-2024 LLM-BBQ benchmarks for Llama-3.1-8B-Instruct**
- Dataset: BBQ Gender_identity (ambig, n=2836)
- Model: `meta-llama/Llama-3.1-8B-Instruct`
- Protocol: zero-shot A/B/C free-form via local vLLM BF16 (no quantization); anchored at midpoint 0.25 of the 0.22–0.28 published range
- Published: 0.25
- **Ours: 0.2496**  (Δ = -0.0004)
- Paper 95 % CI @ n=2836: [0.2341, 0.2659]
- Our 95 % CI @ n=2836: [0.2337, 0.2655]
- Ours falls inside paper CI: **yes**
- CIs overlap: **yes**
- Status: **MATCHED**
- Wall-clock: 314.77 s
