# Invalidated runs

Transcripts kept as evidence of a defect, whose scores are **not results**.
`scripts/agent/summarize_runs.py` does not read this directory.

| File | Why it is here |
|---|---|
| `embedding__…all-mpnet-base-v2__20260918T214413Z.json` | The checkpoint's config lists `MPNetForMaskedLM` but ships no `lm_head.*` weights. The RL-058 check read only the config, so CrowSPairs, AUL, AULA, CAT and ICAT were run on a randomly initialised head and badged `faithful` (AULA exactly 50.00). Fixed the same day: `HuggingFaceBackend` now checks the loaded weights (REVIEW_LATER RL-066). The WEAT/SEAT rows in this file are genuine and were reproduced in the rerun `…__20260918T215721Z.json`. |
