# bias-scope — EMNLP Demo Paper: Evaluation Experiments (Handoff)

This document specifies the experiments for **Section 5 (Evaluation & Case Studies)** of the
EMNLP System Demonstrations paper for the `bias-scope` library. It is written to be executed by
another Claude Code instance (or a human) with access to this repository and a **single 20 GB GPU**.

Read this whole file before writing any code. Follow the hardware constraints exactly.

---

## 0. Context you need

`bias-scope` is a Python library that unifies bias metrics behind one abstract base
(`BiasMetric` in `src/bias_scope/base.py`) specialized into **four families**:

| Family | Base class | Importable from | Example metrics |
|---|---|---|---|
| Embedding | `EmbeddingMetric` | `bias_scope.embeddings_based` | `WEAT`, `SEAT`, `CEAT`, `SentenceBiasScore` |
| Probability | `ProbabilityMetric` | `bias_scope.probability_based` | `CrowSPairs`, `AUL`, `AULA`, `ICAT`, `CAT`, `LMB`, `LPBS` |
| Generated text | `GeneratedTextMetric` | `bias_scope.generated_text_based` | `RegardScore`, `HONEST`, `ToxicityFraction`, `GenderPolarity` |
| Prompt-based | `PromptBasedMetric` | `bias_scope.prompts_based` | `BBQMetric`, `StereoSetMetric`, `BOLD`, `RealToxicityPrompts` |

Prompt-based metrics call an LLM through **LiteLLM** (`model_name=...`, `api_key=...`). They can
target hosted providers (OpenRouter/OpenAI/Anthropic) **or** a local model served through an
OpenAI-compatible endpoint (see §1.3).

Working examples for every metric live in `examples/<family>/`. **Read the relevant example
before implementing each experiment** — it shows the exact `.evaluate(...)` signature and return
keys. Do not guess signatures.

---

## 1. Environment & hardware constraints (READ FIRST)

### 1.1 Hardware budget
- **Single GPU, 20 GB VRAM.** This is the binding constraint on model choice.
- A 7–9B model in fp16 needs ~14–18 GB just for weights and will OOM once you add activations,
  KV cache, and a second model. **Therefore: load all 7–9B decoder models in 4-bit** using
  `bitsandbytes` (`load_in_4bit=True`, `bnb_4bit_compute_dtype=torch.bfloat16`).
- **Load one model at a time.** Never hold two large models in VRAM simultaneously. After each
  model finishes, explicitly free it:
  ```python
  import gc, torch
  del model
  gc.collect()
  torch.cuda.empty_cache()
  ```
- Encoder models (BERT-base ~440 MB, sentence-transformers ~80–400 MB, RoBERTa sentiment ~1.5 GB)
  are small and fit trivially.

### 1.2 Setup
```bash
cd /Users/chadihelwe/Desktop/bias_scope
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"          # library + optional extras (torch, datasets, llm)
pip install bitsandbytes accelerate gensim  # 4-bit quant + GloVe loading
```
Verify GPU: `python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.mem_get_info())"`.

### 1.3 Serving a local model for prompt-based metrics
Prompt-based metrics need an OpenAI-compatible endpoint. Serve the quantized model locally, e.g.:
```bash
# Option A (preferred if it fits): vLLM with AWQ/GPTQ 4-bit checkpoints
pip install vllm
python -m vllm.entrypoints.openai.api_server \
  --model <awq-or-gptq-4bit-model> --quantization awq --max-model-len 4096 --port 8000
```
Then call metrics with `model_name="openai/<model>"`, `api_key="EMPTY"`, and set
`OPENAI_API_BASE=http://localhost:8000/v1` (or the LiteLLM equivalent `api_base=` kwarg if the
metric exposes it — check the constructor).

> If vLLM 4-bit serving is troublesome on 20 GB, fall back to **OpenRouter** for the prompt-based
> experiments (§5.2 prompt rows and §5.1 BBQ row). Budget ≈ **$10–20 total**. Record actual cost.

### 1.4 Global conventions
- Set seeds everywhere: `random`, `numpy`, `torch` (=`42`). For sampled datasets, fix the sample.
- Write **every** raw result to `results/` as CSV/JSON. Never only print.
- Log wall-clock time and peak VRAM (`torch.cuda.max_memory_allocated()`) per run.
- If a metric cannot be run as the original paper intended (e.g. tokenizer mismatch), **do not fake
  it** — record it as N/A with a one-line reason in the results notes. Honesty > coverage.

Create this output layout:
```
results/
  reproducibility/        # §5.1
  cross_family/           # §5.2
  runtime/                # §5.3
  logs/
scripts/experiments/      # all scripts you write go here
```

---

## 2. §5.1 — Reproducibility study

**Goal:** show `bias-scope`'s implementations reproduce published numbers, one metric per family.

**Deliverable:** `results/reproducibility/reproducibility.csv` with columns:
`metric, family, model, dataset, published_score, biasscope_score, delta, notes`
plus one script per row in `scripts/experiments/repro_*.py`.

| # | Metric | Family | Model (original) | Dataset | Published reference value | Notes / gotchas |
|---|---|---|---|---|---|---|
| 1 | `WEAT` | Embedding | **GloVe** (Caliskan et al. 2017) | WEAT-6 word lists (in `examples/embeddings_based/weat.py`) | effect size d ≈ **1.81** (gender–career) | Use **real GloVe vectors** via `gensim` (`glove-wv` / `glove.840B.300d`), NOT a sentence encoder. Encoder gives wrong magnitude. |
| 2 | `CrowSPairs` | Probability | **bert-base-uncased** (Nangia et al. 2020) | CrowS-Pairs full (1,508 pairs) | stereotype score ≈ **60.5** | Run the FULL dataset, not the demo slice. Follow `examples/probability_based/crows_pairs.py` masking scorer exactly. Use `BertPLLScorer`. |
| 3 | `RegardScore` | Generated text | **GPT-2** completions (Sheng et al. 2019) | Regard prompts (Black/White, man/woman) | reported regard gap between groups | Generate completions with GPT-2 (small, fits easily) using the Sheng et al. prompt templates; score with the sentiment/regard classifier the metric uses. Report the group gap, direction-normalized. |
| 4 | `BBQMetric` | Prompt-based | reference LLM | `heegyu/bbq` (HF) | ambiguous-context bias ≈ near 0 for strong models; use published BBQ bias-score behavior as sanity anchor | Prompt-based protocols differ from logit-based originals — **frame this as "protocol-consistent reproduction," not exact match.** Use ≥200 samples/category. See `examples/prompts_based/bbq.py`. |

**Important honesty footnotes to carry into the paper:**
- **StereoSet is intentionally NOT the prompt-based repro row.** Your `StereoSetMetric` uses an
  A/B/C prompting protocol; the original paper used token log-probabilities. These are different
  measurement protocols and will not match numerically. BBQ is the cleaner prompt-based repro row.
  If you want StereoSet in the paper, present it explicitly as "API-based variant" vs. the
  "logit-based original" and discuss the difference — do not claim reproduction.
- **CrowS-Pairs is only valid on masked/bidirectional encoders** (BERT-family). Its pseudo-log-
  likelihood definition does not transfer cleanly to decoder-only LLMs. Keep it on BERT.

**Pass criterion:** |delta| within ~1–2 points (or within the variance the original paper reports).
Document any miss with a reason rather than hiding it.

---

## 3. §5.2 — Cross-family analysis (THE HEADLINE EXPERIMENT)

**Goal:** demonstrate that metrics from different families **disagree** on which model is most
biased — the empirical motivation for a unified library. This is the most important experiment;
give it the most care.

### 3.1 Models (all 4-bit, one at a time, 20 GB budget)
| Model | HF id (or 4-bit AWQ/GPTQ variant) | Role |
|---|---|---|
| Llama-3.1-8B-Instruct | `meta-llama/Llama-3.1-8B-Instruct` | general |
| Mistral-7B-Instruct-v0.3 | `mistralai/Mistral-7B-Instruct-v0.3` | general |
| Qwen2.5-7B-Instruct | `Qwen/Qwen2.5-7B-Instruct` | general |
| Gemma-2-9B-it | `google/gemma-2-9b-it` | general |

> Note: embedding-family metrics operate on **encoder representations**, and CrowS-Pairs needs a
> masked LM. Decoder LLMs above cannot be dropped directly into every metric. **Resolve this with
> the design decision below — do not silently run a metric on an incompatible model.**

### 3.2 Metrics — two per family, all on the SAME bias dimension (gender)
| Family | Metric A | Metric B | What it consumes |
|---|---|---|---|
| Embedding | `SEAT` (gender–career) | `SentenceBiasScore` | sentence embeddings from the model's encoder / hidden states |
| Probability | `CrowSPairs` (gender subset) | `AULA` | masked/pseudo-log-likelihoods |
| Generated text | `RegardScore` (gender) | `HONEST` (gender) | text the model generates from gendered prompts |
| Prompt-based | `BBQMetric` (Gender_identity) | `StereoSetMetric` (gender) | model answers via LiteLLM |

### 3.3 The compatibility design decision (state it explicitly in the paper)
Not every metric can run on every model. Choose **ONE** of these and document it in §5.2:

- **Option A (recommended, cleanest):** Evaluate each model **as a generator** for the
  generated-text and prompt-based families (all 4 decoder LLMs work here directly). For the
  embedding and probability families, evaluate a **fixed encoder companion** (e.g. `bert-base-uncased`
  / a sentence-transformer) rather than the decoder. Then the "model" axis for embed/prob rows is
  the encoder, and the disagreement story is **across metric families on a shared model pool where
  applicable**. Cleanest but the model axis isn't identical across all 8 columns — be honest about it.

- **Option B (uniform but heavier):** Run generated-text + prompt-based on all 4 decoder LLMs
  (their native strength), and additionally derive embedding/probability signals from each decoder's
  hidden states (SEAT/SentenceBiasScore on last-layer hidden states; skip CrowS-Pairs for decoders
  and substitute a decoder-valid probability metric like `AUL`/`AULA` computed on autoregressive
  token log-probs if supported). Verify each metric actually supports decoder inputs before using it.

**Pick Option A unless you confirm the metrics cleanly support decoder hidden states.** Whichever
you choose, write one paragraph in the paper describing exactly what "model" means per column.

### 3.4 Sample sizes
- BBQ: ≥ 200 items per category (Gender_identity). StereoSet: ≥ 200 gender items,
  `num_option_permutations=2` to reduce position bias.
- Generated-text metrics: ≥ 500 generated completions per model (gendered prompt set).
- Report mean ± std across ≥ 3 seeds for anything stochastic (generation temperature, sampling).

### 3.5 Deliverables
1. `results/cross_family/scores.csv` — rows = models, columns = 8 metrics, plus a header row
   annotating **direction** for each metric (↑ = more biased vs ↓ = more biased). Direction is NOT
   enforced by the library — you must determine and record it per metric from its docstring/example.
2. `results/cross_family/rank_correlation.csv` — pairwise **Spearman** rank correlation between the
   4-model rankings induced by every pair of metrics (8×8 matrix).
3. `scripts/experiments/plot_heatmap.py` → `results/cross_family/disagreement_heatmap.png`
   (8×8 Spearman heatmap; this is **Figure 2** of the paper).
4. Optional radar/parallel-coordinates plot of the 4 models across the 8 metrics.

### 3.6 The finding to look for (and report either way)
Expected: **high within-family correlation, low or negative across-family correlation** → the model
that looks least biased under prompt-based metrics can look most biased under embedding/probability
metrics. Headline sentence for the paper:

> "Rankings induced by different metric families are weakly (and sometimes negatively) correlated:
> the least-biased model under prompt-based evaluation is frequently among the most-biased under
> embedding- or probability-based metrics, motivating a unified evaluation toolkit."

If the disagreement does **not** appear, report that honestly — "bias-scope makes cross-family
(dis)agreement *measurable* and, on this model pool, metrics broadly agree/disagree as follows…"
is still a valid and publishable framing. Do not manufacture the result.

---

## 4. §5.3 — Runtime & scalability

**Goal:** show the library is usable at real dataset sizes on **one 20 GB GPU** (state the exact GPU).

**Deliverable:** `results/runtime/runtime.csv` with columns:
`family, metric, workload_description, n_items, wall_clock_s, peak_vram_gb`.

| Family | Representative metric | Fixed workload |
|---|---|---|
| Embedding | `CEAT` (contextual) | 1,000 sentences × 2 groups |
| Probability | `CrowSPairs` | full dataset (1,508 pairs) on bert-base |
| Generated text | `RegardScore` | 5,000 completions through the classifier |
| Prompt-based | `BBQMetric` | 1,000 items via the local/served endpoint (one provider) |

Measure wall-clock with `time.perf_counter()` and VRAM with
`torch.cuda.max_memory_allocated()/1e9` (reset with `torch.cuda.reset_peak_memory_stats()` before
each). One sentence for the paper: "All families complete in under N minutes at paper-scale dataset
sizes on a single 20 GB GPU."

---

## 5. Execution order (do it in this sequence)

1. **Setup & smoke test** (§1): install, verify GPU, run ONE example from each family unchanged to
   confirm the environment works before touching experiments.
2. **§5.1 rows 1–3** (WEAT/GloVe, CrowS-Pairs/BERT, Regard/GPT-2) — all local, cheap, no API cost.
3. **§5.1 row 4** (BBQ) — needs served/hosted LLM; do it together with §5.2 prompt rows.
4. **§5.3 runtime** — reuse the loaded models from §5.1/§5.2 where possible.
5. **§5.2 cross-family** — the big one; run models **one at a time**, free VRAM between each,
   checkpoint `scores.csv` after every (model, metric) pair so a crash never loses completed work.
6. **Analysis & figures** — Spearman matrix + heatmap, then write a `results/SUMMARY.md` with the
   filled-in tables (Table 2 repro, Table 3 cross-family, Table 4 runtime) ready to paste into LaTeX.

---

## 6. Guardrails / do-nots

- **Do not** hold two large models in VRAM at once. Free between runs.
- **Do not** run CrowS-Pairs on decoder-only LLMs and call it reproduction.
- **Do not** present API-based StereoSet as matching the logit-based original.
- **Do not** report a single number for stochastic generation — report mean ± std over seeds.
- **Do not** invent metric `.evaluate()` signatures — read `examples/<family>/<metric>.py` first.
- **Do not** hardcode API keys in scripts — read from env (`OPENROUTER_API_KEY`, etc.).
- **Do** record cost (USD) and GPU-hours; the paper needs a one-line accessibility statement.
- **Do** write raw outputs to `results/` before printing summaries.

## 7. Final output for the paper author

Produce `results/SUMMARY.md` containing:
- Table 2 (Reproducibility): metric | model | published | bias-scope | Δ | note
- Table 3 (Cross-family): model × 8 metrics, with direction arrows, most-biased-per-column bolded
- Table 4 (Runtime): family | metric | n_items | time | peak VRAM
- Figure 2: path to `disagreement_heatmap.png` + one-paragraph interpretation
- A "Reproducibility & cost" note: hardware (20 GB GPU model), total GPU-hours, total API USD, seeds.
