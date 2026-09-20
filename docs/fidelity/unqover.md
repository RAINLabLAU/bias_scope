# UnQoverMetric

**Cited source:** Tao Li, Daniel Khashabi, Tushar Khot, Ashish Sabharwal, and
Vivek Srikumar (2020), *UNQOVERing Stereotyping Biases via Underspecified
Questions*, Findings of EMNLP. [Paper](https://aclanthology.org/2020.findings-emnlp.311/).
**Reference implementation:** [allenai/unqover](https://github.com/allenai/unqover)
at `3e47969b78acc0de436f9d17c9ab1b2f6a108ff0`, Apache-2.0.

## Status

`UnQoverMetric` is an **adaptation**. It preserves the four-variant algebra as
a chat diagnostic, but asks a model to emit option token A or B and normalizes
those two token probabilities. This is materially different from the paper's
unnormalized QA answer-span score or masked-LM subject-token score. Its
`net_bias_score` and `per_bias_type` are BiasScope outputs, not paper headline
metrics, and must not be compared directly with paper results.

Private `bias_scope.prompts_based._unqover_reproduction` helpers reproduce the
pinned `analysis.py` formulas over externally supplied official `output.json`
prediction artifacts: geometric-mean subject scores, positional error
(`delta`), attribute-negation error (`epsilon`), corrected `C`, gamma, mu, and
eta. They do not download artifacts at import time.

## Reproduction boundaries

- **Official prediction-dump parity:** supported when a caller supplies an
  official artifact; its runtime SHA-256 is recorded. The pinned repository's
  archive has no immutable upstream checksum.
- **Historical local reconstruction:** not implemented here. The original QA
  checkpoints and masked-LM identifiers are recoverable, but a modern rerun
  would need explicit environment provenance and is not byte-for-byte exact.
- **Exact paper reproduction:** requires a verified paper-era prediction dump
  or complete historical model/data environment.
