# Agent results across models

| model | kind | WEAT | SEAT | CEAT | CrowSPairs | AUL | AULA | CAT | ICAT | RegardScore | GenderPolarity | DemographicRepresentation | StereotypicalAssociations | HONEST | EMT |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bert-base-cased | encoder | 0.3792 (16) | 0.9246 (128) | 0.4328 (1000)* | 57.63 (262) | 53.05 (262) | 53.82 (262) | 64.19 (229) | 59.11 (229) |  |  |  |  |  |  |
| bert-base-uncased | encoder | 0.6113 (16) | 1.044 (128) | 0.203 (1000)* | 55.73 (262) | 46.56 (262) | 43.89 (262) | 69 (229) | 51.99 (229) |  |  |  |  |  |  |
| roberta-base | encoder | -0.6074 (16) | 1.099 (128) | 0.7752 (1000)* | 54.96 (262) | 56.49 (262) | 53.44 (262) | 55.46 (229) | 61.07 (229) |  |  |  |  |  |  |
| sentence-transformers/all-MiniLM-L6-v2 | embedding | 1.021 (16) | 1.402 (128) | 1.016 (1000)* |  |  |  |  |  |  |  |  |  |  |  |
| sentence-transformers/all-mpnet-base-v2 | embedding | 1.257 (16) | 1.042 (128) | 0.804 (1000)* |  |  |  |  |  |  |  |  |  |  |  |
| Qwen/Qwen2.5-0.5B-Instruct | causal | 0.847 (16) | 0.2512 (128) | 0.06523 (1000)* |  |  |  |  |  | 0 (80) | 0.032 (500) | 0.3824 (51) | 0.5 (4) | 0.016 (1000) | 0.03202 (25)* |
| Qwen/Qwen2.5-1.5B-Instruct | causal | 0.6307 (16) | 0.3193 (128) | 0.1252 (1000)* |  |  |  |  |  | -0.025 (80) | 0.034 (500) | 0.2015 (67) | 0.4359 (13) | 0.02 (1000) | 0.008042 (25)* |
| Qwen/Qwen2.5-3B-Instruct | causal | -0.993 (16) | 0.3051 (128) | 0.08709 (1000)* |  |  |  |  |  | 0.025 (80) | 0.02 (500) | 0.2708 (48) | 0.26 (5) | 0.026 (1000) | 0.02597 (25)* |
| google/gemma-3-1b-it (incomplete) | causal | 0.04788 (16) | 0 (128) |  |  |  |  |  |  | 0.04 (200) | 0.012 (500) | 0.3182 (22) | 0.4 (5) | 0.0335 (2000) | 0.03937 (100)* |
| gpt2 | causal | 0.4006 (16) | -0.05892 (128) | 0.08952 (1000)* |  |  |  |  |  | 0.025 (80) | 0.044 (500) | 0.2564 (78) | 0.4667 (15) | 0.083 (1000) | 0.07369 (25)* |
| gpt2-medium | causal | -0.2031 (16) | -0.1091 (128) | 0.0473 (1000)* |  |  |  |  |  | 0.04 (100) | 0.0612 (500) | 0.2917 (96) | 0.463 (9) | 0.068 (1000) | 0.04463 (50)* |
| meta-llama/Llama-3.2-1B-Instruct (incomplete) | causal | 1.218 (16) |  |  |  |  |  |  |  | 0.02 (200) | 0.01 (500) | 0.1429 (28) | 0.5 (2) | 0.03 (2000) | 0.04894 (100)* |

No scores at all (every run failed before a metric scored): `google/gemma-2-2b-it`. The transcripts record why.

## Reading the table

- Each cell is `score (n)` exactly as `summarize_report` printed it, from that model's
  latest run in which every feedable metric scored (or its latest run, marked
  *incomplete*, when none did). `n` is what the metric counts: pairs, sentences,
  sampled contexts, prompts, templates x K, or - for StereotypicalAssociations -
  target words that co-occurred with a group word, which is why it is small.
- `*` marks a result whose protocol records a deviation from the paper's resources:
  CEAT's contexts come from BOLD's Wikipedia sentences rather than the authors' Reddit
  sample and are pooled as sentences (RL-071); EMT is scored by `unitary/toxic-bert`
  rather than the Perspective API (RL-072). Those numbers are not comparable to the
  papers' tables. The badge next to each score in the transcript is the metric's
  fidelity to its paper; `deviation:` lines there say what this run substituted.
- Causal models are run in bf16 (PLAN.md Section 1); the embedding metrics now use the
  same copy of the model, so they are bf16 numbers too, and bf16 moves an effect size
  by up to 0.1 between CPU and GPU on the same weights (RL-077). Encoders and
  sentence encoders are fp32.
- SEAT and CEAT read the hidden state at position 0. On a model whose tokenizer
  prepends a BOS token (Llama, Gemma) that state is the same for every sentence, so
  they decline (empty cell) or return a degenerate 0 (RL-068). GPT-2 and Qwen add no
  BOS.
- A model with no scores at all is listed below the table with the reason.
- Regenerate with `python scripts/agent/results_table.py --out <this file>`; the logs
  behind every cell are in `README.md`, the procedure in `REPRODUCE.md`.

