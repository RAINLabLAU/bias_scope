# Agent results across models

| model | kind | WEAT | SEAT | CEAT | CrowSPairs | AUL | AULA | CAT | ICAT | RegardScore | GenderPolarity | DemographicRepresentation | StereotypicalAssociations | HONEST | EMT |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| *neutral value* | | 0 | 0 | 0 | 50 | 50 | 50 | 50 | 100 | 0 | 0 | 0 | 0 | 0 | 0 |
| bert-base-cased | encoder | 0.3792 | 0.9246 | 0.4328* | 57.63 | 53.05 | 53.82 | 64.19 | 59.11 |  |  |  |  |  |  |
| bert-base-uncased | encoder | 0.6113 | 1.044 | 0.203* | 55.73 | 46.56 | 43.89 | 69 | 51.99 |  |  |  |  |  |  |
| roberta-base | encoder | -0.6074 | 1.099 | 0.7752* | 54.96 | 56.49 | 53.44 | 55.46 | 61.07 |  |  |  |  |  |  |
| sentence-transformers/all-MiniLM-L6-v2 | embedding | 1.021 | 1.402 | 1.016* |  |  |  |  |  |  |  |  |  |  |  |
| sentence-transformers/all-mpnet-base-v2 | embedding | 1.257 | 1.042 | 0.804* |  |  |  |  |  |  |  |  |  |  |  |
| Qwen/Qwen2.5-0.5B-Instruct | causal | 0.847 | 0.2512 | 0.06523* |  |  |  |  |  | 0 | 0.032 | 0.3824 | 0.5 | 0.016 | 0.03202* |
| Qwen/Qwen2.5-1.5B-Instruct | causal | 0.6307 | 0.3193 | 0.1252* |  |  |  |  |  | -0.025 | 0.034 | 0.2015 | 0.4359 | 0.02 | 0.008042* |
| Qwen/Qwen2.5-3B-Instruct | causal | -0.993 | 0.3051 | 0.08709* |  |  |  |  |  | 0.025 | 0.02 | 0.2708 | 0.26 | 0.026 | 0.02597* |
| google/gemma-3-1b-it (incomplete) | causal | 0.04788 | 0 |  |  |  |  |  |  | 0.04 | 0.012 | 0.3182 | 0.4 | 0.0335 | 0.03937* |
| gpt2 | causal | 0.4006 | -0.05892 | 0.08952* |  |  |  |  |  | 0.025 | 0.044 | 0.2564 | 0.4667 | 0.083 | 0.07369* |
| gpt2-medium | causal | -0.2031 | -0.1091 | 0.0473* |  |  |  |  |  | 0.04 | 0.0612 | 0.2917 | 0.463 | 0.068 | 0.04463* |
| meta-llama/Llama-3.2-1B-Instruct (incomplete) | causal | 1.218 |  |  |  |  |  |  |  | 0.02 | 0.01 | 0.1429 | 0.5 | 0.03 | 0.04894* |

No scores at all (every run failed before a metric scored): `google/gemma-2-2b-it`. The transcripts record why.

## What each column means

| metric | neutral (no bias) | range | direction | what the number is |
|---|---|---|---|---|
| WEAT | 0 | -inf to inf | 0 is no bias; the sign says which side | effect size d of the association between two target word sets and two attribute sets in the model's embeddings; |d| 0.2 / 0.5 / 0.8 small / medium / large |
| SEAT | 0 | -inf to inf | 0 is no bias; the sign says which side | the same effect size on WEAT's words placed in template sentences |
| CEAT | 0 | -inf to inf | 0 is no bias; the sign says which side | the same effect size pooled over 1,000 random draws of contexts for each word |
| CrowSPairs | 50 | 0 to 100 | higher is more biased | % of minimal pairs where the model prefers the more stereotypical sentence |
| AUL | 50 | 0 to 100 | higher is more biased | % of pairs where the model prefers the stereotypical sentence, all tokens unmasked |
| AULA | 50 | 0 to 100 | higher is more biased | AUL with tokens weighted by attention |
| CAT | 50 | 0 to 100 | higher is more biased | stereotype score: % of StereoSet items where the stereotypical fill beats the anti-stereotypical one |
| ICAT | 100 | 0 to 100 | lower is more biased | language-modelling score x how close the stereotype score is to 50; 100 = ideal, 0 = worst |
| RegardScore | 0 | -1 to 1 | 0 is no bias; the sign says which side | difference between two groups in the share of continuations judged negative-regard (group A minus group B) |
| GenderPolarity | 0 | -1 to 1 | 0 is no bias; the sign says which side | mean over continuations of (male words - female words) / (male + female); +1 all male, -1 all female |
| DemographicRepresentation | 0 | 0 to 1 | higher is more biased | total variation distance between how often each group's words appear in the continuations and a uniform distribution |
| StereotypicalAssociations | 0 | 0 to 1 | higher is more biased | mean over target adjectives of that distance, counting only continuations where the adjective co-occurs with a group word |
| HONEST | 0 | 0 to 1 | higher is more biased | share of continuations containing a HurtLex hurtful term |
| EMT | 0 | 0 to 1 | higher is more biased | expected maximum toxicity: mean over prompts of the most toxic of K continuations |

## Reading the table

- Each cell is the score exactly as `summarize_report` printed it, from that model's
  latest run in which every feedable metric scored (or its latest run, marked
  *incomplete*, when none did). The item count `n` behind each score is in the
  transcript (`README.md`) and in the table produced with `--with-counts`; for
  StereotypicalAssociations it is the number of target words that co-occurred with
  a group word, which is small.
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

