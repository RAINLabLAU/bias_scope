# TofNof

**Cited source:** Hong et al. 2025, *Measuring Sycophancy of Language Models in
Multi-turn Dialogues*. [arXiv:2505.23840](https://arxiv.org/abs/2505.23840).
Identifier confirmed against arXiv during the 4.0 gate — PLAN.md Appendix E
listed the id without a title, and it resolves correctly.
**Reference implementation:** none located — `code_status: none_found` with the
search log in `sources/SOURCES.yaml`.
**Sections and files read:** §"Evaluation Metric" (eq. 1 ToF, eq. 2 NoF, the
binary alignment labels and the GPT-4o judge); Figure 1 (the debate scenario).
**Family / access:** prompt / `chat`, `judge`

## Definition in the paper

Each response `y_i^(t) ∈ {0, 1}` is a binary label: does the model still align
with the expected stance at turn `t`? A **GPT-4o judge** assigns those labels
(Figure 1 caption).

```
ToF = E_i [ min_t 1[ y_i^(t) ≠ y_i^Expected ] ]        (1)
NoF = E_i [ Σ_{t=1}^{T−1} 1[ y_i^(t) ≠ y_i^(t−1) ] ]   (2)
```

ToF is the mean earliest turn at which the model diverges — higher is better,
meaning it resists longer. NoF counts stance reversals across the dialogue —
lower is better.

## Current BiasScope implementation

`tof_nof.py`. Records the turn of first flip and the total number of flips, and
also reports a `flip_rate`. The structure matches eqs. 1 and 2.

## Verdict

**adaptation.**

The formulas match. The deviation is in how `y_i^(t)` is obtained: the paper
uses a **GPT-4o judge** to label stance alignment, and this class uses its own
judge configuration. Since the label *is* the measurement, the judge model and
its prompt are part of the metric, not an implementation detail — two TofNof
numbers from different judges are not comparable.

This is precisely the case PLAN.md 7.1 anticipates: "judge choice is part of the
protocol; report judge model and prompt version with every number".

## Required action

1. Refactor onto the `Judge` abstraction of PLAN.md 7.1, so the judge model and
   the prompt-file hash land in `protocol["judge_model"]` and
   `protocol["judge_prompt_version"]`. `TofNof` is named in 7.1 as the metric to
   refactor first, and this audit confirms why.
2. `flip_rate` is BiasScope's addition, not the paper's; keep it, but keep ToF
   and NoF as the reported metrics.
3. Two `BiasResult` scores are needed here, or one plus a breakdown — ToF and
   NoF are complementary and the paper reports both. Decide when `run()` is
   wired up.

## Validation possible

- **Tier 1:** the paper reports ToF and NoF across scenarios and models. The
  numbers depend on the GPT-4o judge, so a local-judge reproduction measures
  something related but not identical, and must be recorded as such.
- **Tier 2:** not possible — no code release located.
- **Tier 3:** null applies (a model that never flips → ToF = T, NoF = 0).
  Swap antisymmetry does not apply; these are stability counts, not paired
  comparisons. Document as an exemption.

## Known limitations of the metric itself

- Fully judge-dependent. The paper's own numbers are GPT-4o's labels, so the
  metric inherits that model's reading of what counts as a stance reversal.
- ToF is censored at `T`: a model that never flips is indistinguishable from one
  that would flip at turn `T+1`, so ToF is not comparable across different
  dialogue lengths.
- Sycophancy is not a social bias in the sense the rest of the library measures.
  It belongs in the library's scope discussion (PLAN.md 7.2's inclusion
  criteria), the same way TruthfulQA does.
