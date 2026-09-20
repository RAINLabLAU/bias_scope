# AnalogicalReasoningBias and CounterfactualFairness

These classes implement no cited paper's metric, so one note.

PLAN.md 4.1's `original` status means "BiasScope's own operationalization,
inspired by a cited idea", and requires that such a class **not carry the cited
paper's metric name** and say "original" in its docstring. The audit here is
therefore short: confirm there is no cited metric being claimed, and check the
naming.

## AnalogicalReasoningBias

**Cited inspiration:** Abid, Farooqi & Zou 2021, *Persistent Anti-Muslim Bias in
Large Language Models*, AIES 2021
([arXiv:2101.05783](https://arxiv.org/abs/2101.05783)); Bolukbasi et al. 2016,
*Man is to Computer Programmer as Woman is to Homemaker?*
([arXiv:1607.06520](https://arxiv.org/abs/1607.06520)).
**Reference implementation:** none — `code_status: none_found`.

Neither paper defines a metric called "analogical reasoning bias". Bolukbasi et
al. propose the *direct* and *indirect bias* measures over a gender subspace,
and an analogy-generation procedure; Abid et al. measure completion and analogy
outcomes for Muslim-referencing prompts but define no reusable score. This class
prompts a chat model with analogy completions and scores the answers — a
reasonable idea, but its own.

**Verdict: original.** The name does not claim either paper's metric, which is
correct. `MetricInfo.reference` records both as inspiration only.

## CounterfactualFairness → IdentitySwapConsistency

**Cited inspiration:** Kusner, Loftus, Russell & Silva 2017, *Counterfactual
Fairness*, NeurIPS 2017 ([arXiv:1703.06856](https://arxiv.org/abs/1703.06856)).
**Reference implementation:** none located.

Kusner et al. define counterfactual fairness as a **causal criterion on a
predictor**: a predictor is counterfactually fair if its distribution is
unchanged in the counterfactual world where a protected attribute differs, given
a causal model. It is a definition, not a score, and it requires a causal graph.

This class measures the **cosine similarity between response embeddings** under
identity swaps. That is an operationalization of the same intuition, but it is
not Kusner's criterion: no causal model, no predictor, no counterfactual
distribution.

**Verdict: original.**

**Resolved in 0.2.0:** renamed to `IdentitySwapConsistency`, because
`CounterfactualFairness` is the exact title of Kusner's paper and reads as a
claim to implement it (PLAN.md 4.2, 5.2). The statistic is unchanged, so the old
name stays importable until 0.3.0 behind a `DeprecationWarning`.

## Validation possible (these original metrics)

- **Tier 1:** none. Record `no_published_reference` — that is the correct status
  for an original metric, not a gap to be filled.
- **Tier 2:** not applicable; there is no reference implementation to compare
  against.
- **Tier 3:** the metamorphic properties still apply and are the **primary**
  validation for original metrics. Null and swap antisymmetry are meaningful for
  these original metrics and should be prioritised in Phase 3, since they are the only
  objective check available.

## Note for the paper

These three are the clearest examples of why PLAN.md's `original` status exists.
The v0.1 paper counted 40 metrics without distinguishing which implement a cited
statistic and which are BiasScope's own; Section 7.2's "Recount for the paper: N
faithful + N adaptation + N original, by family" is the fix, and these three
belong in the third column.
