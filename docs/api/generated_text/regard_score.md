# RegardScore

<!-- metric-card:start -->
| | |
|---|---|
| Family | generated_text |
| Model access | `completions` |
| Neutral value | 0 |
| Direction | signed; 0 is neutral |
| Range | -1 to 1 |
| Languages | en |
| Fidelity | **adaptation**: same comparison as the cited paper, but a different access mode or scoring path that can change the numbers. Scores regard with sasha/regardv3, Sheng's published checkpoint, rather than her own regard1 3-BERT majority-vote ensemble that produced the paper's numbers - a different checkpoint trained on the v2 dataset with an added 'other' bucket. v0.1.1 defaulted to a SENTIMENT classifier, which was a mismatch: Sheng et al. [Audit note](../../fidelity/regard_score.md). |
| Source | The Woman Worked as a Babysitter: On Biases in Language Generation, EMNLP 2019 — https://arxiv.org/abs/1909.01326 |
| Reference code | https://github.com/ewsheng/nlg-bias @ 7f8d08ea4f33 |
<!-- metric-card:end -->


::: bias_scope.generated_text_based.regard_score.RegardScore

`RegardScore` measures regard, not sentiment. Its default `sasha/regardv3`
checkpoint is a BiasScope adaptation and is not demonstrated to be the
historical Sheng classifier used in BOLD. Use private BOLD reproduction helpers
with externally supplied historical-style labels for BOLD paper aggregation.
