# TruthfulQA

**Cited source:** Lin, Hilton & Evans (2022), *TruthfulQA: Measuring How
Models Mimic Human Falsehoods*, ACL 2022. Paper-era repository evidence is
[`8b48f682acc3a71cd04e32e46e6da40ddb1a5860`](https://github.com/sylinrl/TruthfulQA/tree/8b48f682acc3a71cd04e32e46e6da40ddb1a5860).

## Current implementation

`TruthfulQA` implements original MC1/MC2 aggregation over caller-supplied full
reference-answer log-likelihoods. MC1 compares the designated **Best Answer**
strictly against all false answers; MC2 is normalized probability mass on all
true answers. Generation accepts saved GPT-judge/GPT-info ``P(" yes")`` values
and applies the original ``>= 0.5`` threshold separately for truth and
informativeness, plus their per-answer conjunction.

`ReferenceOverlapTruthfulness` is the former token-F1 reference-overlap
diagnostic. It is not a published TruthfulQA score.

## Paper reproduction boundary

Private research helpers use the preserved 817-question, 38-category
`data/v0/TruthfulQA.csv` artifact and its original MC task. They reject the
later 790-question root CSV and 2025 binary-MC update. Downloadable GPT-2,
GPT-Neo, and UnifiedQA families can be reconstructed locally through original
prompt/candidate likelihood collection, with runtime and resolved revision
metadata recorded at execution time.

Historical generation-score parity is unavailable: the paper's fine-tuned
OpenAI GPT-judge/GPT-info model artifacts were not released. Released training
JSONL files are not equivalent to those historical deployed models. A modern
judge is an adaptation, not ACL-2022 parity.
