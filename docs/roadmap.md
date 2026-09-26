# Roadmap and considered-but-excluded

Two lists. The first is what is planned; the second is what was looked at and
left out, with the [inclusion criterion](inclusion_criteria.md) it failed. The
second list is the more useful one: it is the answer to "why isn't X here?".

## Considered and excluded

| Candidate | Failing criterion | Detail |
|---|---|---|
| Bias in gradients / attention-head attribution methods | 3 (family) | Needs access to internals beyond embeddings and logits. The four families are defined by what the metric reads from the model, and adding a fifth would make the access guards unenforceable. |
| Debiasing methods (INLP, Sent-Debias, hard-debias) | 3 (family) | Interventions, not measurements. BiasScope measures; a debiased model is just another model to measure. |
| Perspective API toxicity as a *required* backend | 4 (proprietary) | Kept as an option, never as the default. The toxicity metrics take a scorer callable, so an open classifier works and the choice is recorded in the protocol. |
| HolisticBias FGB / PGB as reproducible metrics | 2 (specification) | The 217-class style classifier was never released. Present as `mismatch` rather than removed, because the *datasets* are usable and the honest label is more useful than a silent omission. See REVIEW_LATER RL-013. |
| Bias Enumeration Algorithm (Eloundou et al. 2024, Sec. 3.4) | 2 (specification) and output shape | Produces natural-language "axes of difference" rather than a number, so it has no `BiasResult`. Its sibling, the harmful-stereotype metric, is implemented. |
| Single-number "bias leaderboards" and composite indices | 5 / design | An explicit non-goal. Averaging metrics that disagree about direction, neutral value and family produces a number no one can act on. `Report` has no `overall` field and a test asserts it never gains one. |
| Human-annotation-only measures (crowd stereotype ratings) | 4 (dependency) | No model access at all; nothing for the library to compute. |
| WEFE, LangFair, `evaluate`, lm-evaluation-harness, HELM | — (not metrics) | Libraries, not metrics. Cited as related work; the harness and `evaluate` are wrapped as adapters (PLAN.md Phase 5) so their numbers can be compared against ours. |

## Planned

**Multilingual coverage.** Loaders for MBBQ, KoBBQ, CBBQ, the French CrowS-Pairs
extension and SHADES exist in `bias_scope.multilingual`, and no dataset is
redistributed (PLAN.md Section 12: loader only). `MetricInfo.languages` is
backed by the authors' own shipped files — CBS declares 12 languages because
`configuration.py` has 12 keys, HONEST declares 6 because `resources/` has 6
template files. What remains is running each metric end to end in a non-English
language and recording the result, which needs the Phase 7 study runs.

**Adapters** (PLAN.md Phase 5). Wrapping lm-evaluation-harness and Hugging Face
`evaluate` so their numbers return as `BiasResult`, with equivalence tests
against the native implementations. Any disagreement gets documented with its
cause — that is a finding, not a bug to hide.

**Remaining fidelity upgrades.** Three `adaptation` metrics have a documented
path to `faithful`: UnQover via a masked-LM path, TruthfulQA MC1/MC2, and
SocialGroupSubstitution's Wasserstein-1 form. Each is tracked in
`REVIEW_LATER.md` with the specific change required.

**The two mismatches** stay `mismatch` until the HolisticBias classifier is
released. They will not be quietly relabelled.

## How to propose a metric

Open an issue with the paper, the reference code URL and commit, and which of
the five criteria it meets. The first implementation step is the Section 4.0
gate — a `sources/SOURCES.yaml` entry naming the sections and file ranges
actually read — not code.
