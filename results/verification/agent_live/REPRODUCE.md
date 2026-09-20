# Reproducing the agent experiments, step by step

This explains how to run the same experiments that produced `RESULTS.md` and
`README.md` in this directory, and what the code does behind each step.
Nothing here needs you to paste evaluation data or to know the metrics'
internals.

## 1. Install

```bash
git clone <this repo> && cd bias_scope
pip install -e ".[all,dev]"           # the library, torch, transformers, datasets
pip install -e ".[agent-openai]"      # the agent, with the OpenAI-shaped client OpenRouter uses
```

A CUDA GPU is assumed below (`--device cuda`). Everything also runs on CPU with
`--device cpu`; the causal-model runs then take much longer because they
generate text.

## 2. Restore the authors' data

The datasets the agent can load are the metric authors' own releases. They are
not committed (size, licences); one command clones them at pinned commits into
the git-ignored `third_party/` directory and downloads HurtLex:

```bash
python scripts/sources/fetch_sources.py --all
```

Two more resources download themselves from the Hugging Face Hub on first use,
at pinned revisions: the RealToxicityPrompts dataset and the `unitary/toxic-bert`
classifier. Hub models you evaluate (gpt2, bert-base-uncased, ...) download the
same way. Gated models (Llama, Gemma) need `hf auth login` once; after the
download, run them with `HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1` set, because
one embedding loader probes the Hub for a file gated repos do not serve and
gets a 401 even with a valid token (REVIEW_LATER RL-074).

## 3. Choose the agent's LLM

The agent is one LLM in a tool-calling loop. It is configured by environment
variables only. The experiments used DeepSeek over OpenRouter:

```bash
echo 'OPENROUTER_API_KEY=sk-or-...' > .env      # .env is git-ignored
set -a; . ./.env; set +a
export BIASSCOPE_AGENT_PROVIDER=openrouter
export BIASSCOPE_AGENT_MODEL=deepseek/deepseek-v4.1-flash
```

Any of `anthropic`, `openai`, `gemini`, `local` (Ollama etc.), `openrouter`,
`litellm` works; see the main README for the key each one reads.

## 4. Run one model

```bash
python scripts/agent/live_conversation.py --scenario causal --model-id gpt2 --device cuda
```

`--scenario` is the kind of model: `encoder` (a masked LM such as BERT),
`causal` (a decoder-only LM such as GPT-2, Qwen, Llama), `embedding` (a
sentence encoder with no LM head). It only chooses the words of the three
scripted user turns and the dtype (fp32 for encoders, bf16 for causal LMs).

The script plays the user. It sends three turns and prints the agent's replies:

1. "Here is the model, set it up. Which metrics can run on it and why not the others?"
2. "Plan an evaluation, axis gender. Use only datasets you can load yourself. Show the plan and provenance. Do not run."
3. "Yes, run it, and summarise every metric with its score, meaning and fidelity label."

When it finishes it writes one JSON transcript into this directory, named
`<scenario>__<agent model>__<target model>__<timestamp>.json`, and prints:

- `tools:` the order of every tool the agent called;
- `scores (from the library, not the prose):` the numbers the library returned;
- `untraceable figures:` any decimal in the agent's final message that appears in no tool result (a fabricated score would show up here);
- `coverage:` recommended vs feedable vs scored, and whether the run is complete.

## 5. Run the batch and build the tables

```bash
run() { python scripts/agent/live_conversation.py --scenario "$1" --model-id "$2" --device cuda; }
for m in gpt2 gpt2-medium Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-1.5B-Instruct \
         meta-llama/Llama-3.2-1B-Instruct google/gemma-3-1b-it \
         Qwen/Qwen2.5-3B-Instruct; do run causal "$m"; done
# gated checkpoints already downloaded: prefix the command with
#   HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 BIASSCOPE_AGENT_INSPECT_LIVE=0
for m in bert-base-uncased bert-base-cased roberta-base; do run encoder "$m"; done
for m in sentence-transformers/all-MiniLM-L6-v2 sentence-transformers/all-mpnet-base-v2; do
  run embedding "$m"
done

python scripts/agent/summarize_runs.py --check                                    # one row per score
python scripts/agent/results_table.py --out results/verification/agent_live/RESULTS.md   # the pivot table
python scripts/agent/results_table.py --format latex --out results/verification/agent_live/RESULTS.tex
python scripts/agent/render_transcripts.py --out results/verification/agent_live/README.md  # the logs
```

All three readers take their numbers from the recorded tool output, never from
the agent's prose. The table uses each model's latest run in which every
feedable metric scored.

## 6. What happens in the backend during one run

Every tool the agent calls is a plain Python function in
`src/bias_scope_agent/tools.py`; the agent decides *when* to call them, the
library decides *what* they return.

1. **`construct_backend`** loads the model through `HuggingFaceBackend` and
   derives what it can be asked for, its *access*: `embeddings` for every
   model, `completions` for a causal LM, masked-token `logits` for an encoder
   only if its checkpoint really ships a masked-LM head (the config is checked,
   then the loaded weights).
2. **`recommend_metrics_tool`** returns every metric whose needs are a subset
   of that access. This is the library's `recommend_metrics`; the agent adds
   no selection logic. `explain_exclusions_tool` says why each other metric
   is out.
3. **`list_datasets`** lists the data the harness can load itself, and which
   metrics each feeds. **`prepare_inputs`** loads one dataset server-side and
   returns a *handle* plus provenance (file, sha256, item counts, decoding,
   seed). The data never passes through the agent, because agents were caught
   altering it in transit. For the five generation-based datasets this step
   also runs the model under evaluation to produce continuations, seeded and
   cached under `cache/generations/`.
4. **`plan_suite`** turns the chosen metrics into a plan with an id and lists
   what each still needs.
5. **`confirm_plan`** is the gate. `run_suite` is refused by the dispatcher
   unless the plan was shown, a real turn boundary passed, and the user's
   reply was an unambiguous yes.
6. **`run_suite`** resolves the handles and calls `BiasSuite.run`, which
   constructs each metric, calls its `run()`, and collects a `BiasResult` per
   metric: score, item count `n`, confidence interval where the metric has
   item-level scores, and a protocol block (model, dtype, seed, dataset,
   revision, decoding, resources). A metric that cannot score declines with a
   reason, which is reported as a skip, never as a zero.
7. **`summarize_report`** renders the library's report: one line per metric
   with its fidelity badge (`faithful`, `ADAPTATION`, `ORIGINAL`, `UNAUDITED`)
   and `n`, plus a `deviation:` line whenever a provider substituted a
   resource the paper used.

What each dataset does:

| Dataset | Feeds | What the backend does |
|---|---|---|
| `crows_pairs` | CrowSPairs, AUL, AULA | Loads Nangia et al.'s CSV, filters to the axis; the metric scores each pair with the encoder's masked-LM head. |
| `stereoset` | CAT, ICAT | Loads Nadeem et al.'s dev.json, intrasentence items with one-word fills. |
| `weat`, `seat` | WEAT, SEAT | Loads Caliskan's word sets / May's template sentences; the metric embeds them with the model. |
| `ceat_contexts` | CEAT | For each WEAT word, collects BOLD Wikipedia sentences containing it (substitute corpus, recorded as a deviation); CEAT embeds and resamples them 1,000 times. |
| `bold_regard` | RegardScore | Generates continuations of BOLD gender prompts; Sheng et al.'s regard classifier scores them. |
| `bold_gender_polarity` | GenderPolarity | Generates continuations of 500 BOLD profession prompts; counts BOLD's male/female words. |
| `bold_helm_bias` | DemographicRepresentation, StereotypicalAssociations | Same generations; HELM's word lists as groups, HELM's adjectives as targets. |
| `honest` | HONEST | Turns Nozza et al.'s templates into prompts, samples 20 short continuations each, checks them against HurtLex. |
| `rtp_toxicity` | EMT | Samples 25 continuations per RealToxicityPrompts prompt, scores each with `unitary/toxic-bert` (substitute for the Perspective API, recorded as a deviation). |

Two things to keep in mind when reading a number:

- A `faithful` badge describes the metric's formula. A `deviation:` line under
  it means this *run* substituted a resource, and the number is not comparable
  to the paper's tables.
- Generation-based scores depend on the prompts, the decoding and the seed.
  All three are in the provenance and the protocol block, so a rerun on the
  same model reproduces the same generations from the cache.

## 7. Interactive use

The same agent, with you typing instead of the script:

```bash
bias-scope-agent
```

Ask it about a model, ask for a plan, confirm, and it runs the same tools in
the same order. The scripted runner exists so that runs are recorded and
comparable.

## 8. Adding a dataset

A provider is one `DatasetSpec` (name, metrics it feeds, axes, source) and one
builder function that reads the authors' file and returns the metric's
`evaluate` arguments plus provenance, registered in
`src/bias_scope_agent/datasets.py`. The existing providers in
`datasets_generated.py` are the pattern to copy; each has tests under
`tests/test_bias_scope_agent/` that run on a stub backend without any download.
