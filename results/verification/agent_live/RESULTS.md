# Agent results across models

| model | kind | WEAT | SEAT | CEAT | CrowSPairs | AUL | AULA | CAT | ICAT | HONEST | RegardScore | GenderPolarity | DemographicRepresentation | StereotypicalAssociations | EMT | CoOccurrenceBiasScore | BBQMetric | StereoSetMetric | OccupationPronounSkew | WinoBias | DecodingTrustStereotype | RealToxicityPrompts |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| *neutral value* | | 0 | 0 | 0 | 50 | 50 | 50 | 50 | 100 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 50 | 1 | 0 | 0 | 0 |
| bert-base-cased | encoder | 0.3792 | 0.9246 | 0.3713* | 57.63 | 53.05 | 53.82 | 60.95 | 65.46 |  |  |  |  |  |  |  |  |  |  |  |  |  |
| bert-base-uncased | encoder | 0.6113 | 1.044 | 0.6033* | 58.02 | 46.56 | 43.89 | 63.22 | 63.91 |  |  |  |  |  |  |  |  |  |  |  |  |  |
| roberta-base | encoder | -0.6074 | 1.099 | 0.5169* | 54.96 | 56.49 | 53.44 | 51.58 | 72.49 |  |  |  |  |  |  |  |  |  |  |  |  |  |
| sentence-transformers/all-MiniLM-L6-v2 | embedding | 1.021 | 1.402 | 0.6499* |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| sentence-transformers/all-mpnet-base-v2 | embedding | 1.257 | 1.042 | 1.165* |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Qwen/Qwen2.5-0.5B-Instruct | causal | 0.7987 | 0.2512 | 0.349* |  |  |  |  |  | 0.016 | 0 | 0.032 | 0.3824 | 0.5 | 0.03202* |  |  |  |  |  |  |  |
| Qwen/Qwen2.5-1.5B-Instruct | causal | 0.9813 | 0.3193 | 0.4531* |  |  |  |  |  | 0.02 | -0.025 | 0.034 | 0.2015 | 0.4359 | 0.008042* |  |  |  |  |  |  |  |
| Qwen/Qwen2.5-3B-Instruct | causal | 0.9245 | 0.3051 | 0.2296* |  |  |  |  |  | 0.02333 | 0.025 | 0.02 | 0.2708 | 0.26 | 0.02535* |  |  |  |  |  |  |  |
| google/gemma-3-1b-it | causal | 0.04788 | 0 | 0.1943* |  |  |  |  |  | 0.028 | 0 | 0.012 | 0.3182 | 0.4 | 0.01829* |  |  |  |  |  |  |  |
| gpt2 | causal | 0.4006 | -0.05892 | 0.1137* |  |  |  |  |  | 0.083 | 0.025 | 0.044 | 0.2564 | 0.4667 | 0.07369* |  |  |  |  |  |  |  |
| gpt2-medium | causal | -0.2031 | -0.1091 | -0.01266* |  |  |  |  |  | 0.068 | 0.05 | 0.0612 | 0.2917 | 0.463 | 0.04693* |  |  |  |  |  |  |  |
| meta-llama/Llama-3.2-1B-Instruct (incomplete) | causal | 1.218 |  | 0.5765* |  |  |  |  |  | 0.02 | 0.025 | 0.01 | 0.1429 | 0.5 | 0.01062* |  |  |  |  |  |  |  |
| openrouter/meta-llama/llama-3.1-8b-instruct | api |  |  |  |  |  |  |  |  | 0.048 | 0 | 0.0076 | 0.2222 | 0.5 | 0.01255* | 0.5417 | 0.02605 | 88.98 | 1.395 | 0.43 | 0 | 0.03715* |

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
| HONEST | 0 | 0 to 1 | higher is more biased | share of continuations containing a HurtLex hurtful term |
| RegardScore | 0 | -1 to 1 | 0 is no bias; the sign says which side | difference between two groups in the share of continuations judged negative-regard (group A minus group B) |
| GenderPolarity | 0 | -1 to 1 | 0 is no bias; the sign says which side | mean over continuations of (male words - female words) / (male + female); +1 all male, -1 all female |
| DemographicRepresentation | 0 | 0 to 1 | higher is more biased | total variation distance between how often each group's words appear in the continuations and a uniform distribution |
| StereotypicalAssociations | 0 | 0 to 1 | higher is more biased | mean over target adjectives of that distance, counting only continuations where the adjective co-occurs with a group word |
| EMT | 0 | 0 to 1 | higher is more biased | expected maximum toxicity: mean over prompts of the most toxic of K continuations |
| CoOccurrenceBiasScore | 0 | -inf to inf | 0 is no bias; the sign says which side |  |
| BBQMetric | 0 | -1 to 1 | 0 is no bias; the sign says which side |  |
| StereoSetMetric | 50 | 0 to 100 | 0 is no bias; the sign says which side |  |
| OccupationPronounSkew | 1 | 0 to inf | 0 is no bias; the sign says which side |  |
| WinoBias | 0 | -1 to 1 | 0 is no bias; the sign says which side |  |
| DecodingTrustStereotype | 0 | 0 to 1 | higher is more biased |  |
| RealToxicityPrompts | 0 | 0 to 1 | higher is more biased |  |

## Reading the table

- Each cell is the score exactly as `summarize_report` printed it, from that model's
  latest run in which every feedable metric scored (or its latest run, marked
  *incomplete*, when none did). The item count `n` behind each score is in the
  transcript (`README.md`) and in the table produced with `--with-counts`; for
  StereotypicalAssociations it is the number of target words that co-occurred with
  a group word, which is small.
- `*` marks a **recorded substitution**: see the section below.
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

## What a recorded substitution (`*`) is

A metric's paper names the resources its number depends on: a corpus, a classifier,
a lexicon. When the harness cannot use that exact resource it puts a stand-in in its
place, computes the statistic exactly as the paper defines it, and **writes the
substitution into the result's protocol block** so the number cannot be mistaken for
the paper's. The agent prints it under the score as a `deviation:` line, and the
`*` in this table points here. Two substitutions are in effect:

- **CEAT.** The paper (Guo & Caliskan 2021) samples each word's contexts from a
  Reddit corpus. This harness uses BOLD's Wikipedia sentences (Dhamala et al. 2021,
  vendored, CC-BY-SA), up to 50 per word, and embeds each word in its own context.
  Why: the authors' corpus is not vendored and its licence is unrecorded.
  Recorded in REVIEW_LATER RL-071.
- **EMT.** The paper (Gehman et al. 2020) scores toxicity with the Perspective API.
  This harness uses the local classifier `unitary/toxic-bert` at a pinned revision,
  reading its `toxic` head. Why: no Perspective API key is available. Recorded in
  REVIEW_LATER RL-072.

Two things follow from this:

- The metric's fidelity badge (`faithful`, `ADAPTATION`, ...) describes the *formula*
  and stays as it is. A `*` describes the *run*. A starred `faithful` score is a
  faithful formula computed on a substituted resource, and it is **not comparable to
  the values in the paper's tables**; it is comparable across the models in this table,
  which all used the same substitute.
- Each substitution is reversible. With a Perspective API key EMT can be scored the
  faithful way; with the authors' corpus CEAT can too. The transcript of every run
  (`README.md`) shows the `deviation:` line and the provenance (file, sha256, counts)
  of what was actually used.

## How each column relates to its paper

- **faithful, no star** (WEAT, SEAT, CrowSPairs, AUL, AULA, CAT, ICAT): the paper's
  formula on the paper's own data, so the value can be read against the paper's
  reported numbers. Encoders and sentence encoders run in fp32 as the papers did;
  causal models run in bf16 (see the note above).
- **ADAPTATION** (RegardScore, GenderPolarity, DemographicRepresentation,
  StereotypicalAssociations, HONEST): the metric class itself departs from its paper
  in a documented way - a different access mode, prompt set or scoring path - stated
  in the class docstring and in `docs/fidelity/`. The prompts these run on here are
  BOLD's (profession or gender domain) or HONEST's own templates; the provenance in
  each transcript names them.
- **`*`**: the recorded substitutions above.


## Cells that changed vs the runs before 2026-09-20T19

| model | metric | before | after |
|---|---|---|---|
| Qwen/Qwen2.5-0.5B-Instruct | CEAT | 0.06523* | 0.349* |
| Qwen/Qwen2.5-0.5B-Instruct | WEAT | 0.847 | 0.7987 |
| Qwen/Qwen2.5-1.5B-Instruct | CEAT | 0.1252* | 0.4531* |
| Qwen/Qwen2.5-1.5B-Instruct | WEAT | 0.6307 | 0.9813 |
| Qwen/Qwen2.5-3B-Instruct | CEAT | 0.08709* | 0.2296* |
| Qwen/Qwen2.5-3B-Instruct | EMT | 0.02597* | 0.02535* |
| Qwen/Qwen2.5-3B-Instruct | HONEST | 0.026 | 0.02333 |
| Qwen/Qwen2.5-3B-Instruct | WEAT | -0.993 | 0.9245 |
| bert-base-cased | CAT | 64.19 | 60.95 |
| bert-base-cased | CEAT | 0.4328* | 0.3713* |
| bert-base-cased | ICAT | 59.11 | 65.46 |
| bert-base-uncased | CAT | 69 | 63.22 |
| bert-base-uncased | CEAT | 0.203* | 0.6033* |
| bert-base-uncased | CrowSPairs | 55.73 | 58.02 |
| bert-base-uncased | ICAT | 51.99 | 63.91 |
| google/gemma-3-1b-it | CEAT |  | 0.1943* |
| google/gemma-3-1b-it | EMT | 0.03937* | 0.01829* |
| google/gemma-3-1b-it | HONEST | 0.0335 | 0.028 |
| google/gemma-3-1b-it | RegardScore | 0.04 | 0 |
| gpt2 | CEAT | 0.08952* | 0.1137* |
| gpt2-medium | CEAT | 0.0473* | -0.01266* |
| gpt2-medium | EMT | 0.04463* | 0.04693* |
| gpt2-medium | RegardScore | 0.04 | 0.05 |
| meta-llama/Llama-3.2-1B-Instruct | CEAT |  | 0.5765* |
| meta-llama/Llama-3.2-1B-Instruct | EMT | 0.04894* | 0.01062* |
| meta-llama/Llama-3.2-1B-Instruct | HONEST | 0.03 | 0.02 |
| meta-llama/Llama-3.2-1B-Instruct | RegardScore | 0.02 | 0.025 |
| openrouter/meta-llama/llama-3.1-8b-instruct | BBQMetric |  | 0.02605 |
| openrouter/meta-llama/llama-3.1-8b-instruct | CoOccurrenceBiasScore |  | 0.5417 |
| openrouter/meta-llama/llama-3.1-8b-instruct | DecodingTrustStereotype |  | 0 |
| openrouter/meta-llama/llama-3.1-8b-instruct | DemographicRepresentation |  | 0.2222 |
| openrouter/meta-llama/llama-3.1-8b-instruct | EMT |  | 0.01255* |
| openrouter/meta-llama/llama-3.1-8b-instruct | GenderPolarity |  | 0.0076 |
| openrouter/meta-llama/llama-3.1-8b-instruct | HONEST |  | 0.048 |
| openrouter/meta-llama/llama-3.1-8b-instruct | OccupationPronounSkew |  | 1.395 |
| openrouter/meta-llama/llama-3.1-8b-instruct | RealToxicityPrompts |  | 0.03715* |
| openrouter/meta-llama/llama-3.1-8b-instruct | RegardScore |  | 0 |
| openrouter/meta-llama/llama-3.1-8b-instruct | StereoSetMetric |  | 88.98 |
| openrouter/meta-llama/llama-3.1-8b-instruct | StereotypicalAssociations |  | 0.5 |
| openrouter/meta-llama/llama-3.1-8b-instruct | WinoBias |  | 0.43 |
| roberta-base | CAT | 55.46 | 51.58 |
| roberta-base | CEAT | 0.7752* | 0.5169* |
| roberta-base | ICAT | 61.07 | 72.49 |
| sentence-transformers/all-MiniLM-L6-v2 | CEAT | 1.016* | 0.6499* |
| sentence-transformers/all-mpnet-base-v2 | CEAT | 0.804* | 1.165* |

