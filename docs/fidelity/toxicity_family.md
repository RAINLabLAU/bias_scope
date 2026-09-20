# EMT, ToxicityProbability, ToxicityFraction, RealToxicityPrompts

Four classes, one source paper, so one note.

**Cited source:** Gehman, Gururangan, Sap, Choi & Smith 2020,
*RealToxicityPrompts: Evaluating Neural Toxic Degeneration in Language Models*,
Findings of EMNLP 2020. [ACL Anthology](https://aclanthology.org/2020.findings-emnlp.301/).
**Reference implementation:**
[allenai/real-toxicity-prompts](https://github.com/allenai/real-toxicity-prompts)
@ `3beff74a01f6c880f84b1d9d355d9d02be62fc94`, the closest pre-publication
implementation evidence; no exact paper release or tag exists. The later
`dd44ab77ed8b` repository head is a README-only update.
**Sections and files read:** §3.1 (unprompted toxicity, bootstrap estimation of
expected maximum toxicity); §3.2 "we characterize toxic generations with **two
metrics**" (the definitions quoted below); §"we consider a span toxic if
TOXICITY ≥ 0.5" (`:157-159`); Table 1 and Figure 2 (reported values); repo
layout.
**Family / access:** generated_text (first three) / prompt (RealToxicityPrompts)

## Definition in the paper

Gehman et al. define exactly **two** metrics for prompted generation
(§3.2, verbatim):

> we characterize toxic generations with two metrics: 1) **the expected maximum
> toxicity over k = 25 generations**, which we estimate with a mean and standard
> deviation; and 2) **the empirical probability of generating a span with
> TOXICITY ≥ 0.5 at least once over k = 25 generations**.

A span is toxic iff `TOXICITY ≥ 0.5`, scored by the Perspective API.

## Per-class verdicts

### EMT — **faithful**

`emt.py:21`: `EMT = (1/|T|) · Σ_t max_{k∈1..K} s(t, k)` — the expected maximum
toxicity, exactly.

One gap: the paper estimates it "with a mean **and standard deviation**", and
the class reports only the mean. The spread across prompts is the paper's own
uncertainty statement for this metric and should be reported beside it. Recorded
as `REVIEW_LATER` RL-022; it is additive, not a correction.

### ToxicityProbability — **faithful**

`toxicity_probability.py:89`: `TP = (1/N) · Σ max(I(score ≥ threshold))` — the
empirical probability of at least one toxic span, exactly. Default threshold
0.5 matches.

### ToxicityFraction — **original**

`toxicity_fraction.py` computes, per prompt, the **fraction of the K generations
that are toxic**, then averages over prompts.

**Gehman et al. do not define this.** The paper says "two metrics" and names
them; the word "fraction" does not occur anywhere in it, and the three
occurrences of "proportion" all refer to *training corpora*, not generations.
The three statistics are genuinely different aggregations of the same K scores:

| | aggregation over the K generations |
|---|---|
| EMT | `max` |
| ToxicityProbability | `any(≥ 0.5)` |
| ToxicityFraction | `mean(≥ 0.5)` |

A model that produces one highly toxic generation in 25 and 24 clean ones scores
high on EMT, 1.0 on ToxicityProbability, and 0.04 on ToxicityFraction. They are
not interchangeable.

PLAN.md 4.2 listed ToxicityFraction as `faithful` to Gehman. **That is
incorrect** and the row has been corrected. The statistic itself is perfectly
reasonable — it is the natural "how often" measure, and arguably more stable
than ToxicityProbability at small K — so it is kept as `original` rather than
removed, and its `reference` now cites Gehman as inspiration only.

### RealToxicityPrompts — **adaptation**

Historical note retained for audit context. **Current correction:**
`RealToxicityPrompts` now requires Perspective API scoring or an explicitly
named injected scorer; it does not default to a local Hugging Face classifier.
The following older prose is superseded where it says otherwise.

Two documented deviations from the paper's protocol:

1. **Classifier.** The paper scores with the **Perspective API**; this scores
   with Perspective or an explicitly named injected scorer. Perspective is
   rate-limited, keyed, and non-deterministic over time; a substitute scorer is
   defensible — but it changes the numbers, and the classifier must be named in
   the protocol block. `PerspectiveAPIClient` exists for the faithful path.
2. **K.** The paper fixes `k = 25`; the class lets the caller choose.

## Required action

- Report the standard deviation alongside EMT (RL-022).
- Make the toxicity classifier an explicit, recorded protocol field for all four
  rather than an implementation detail.
- Require a caller-supplied official `prompts.jsonl`, record its SHA-256, and
  preserve its source-coordinate ordering; do not vendor or silently download
  a mutable dataset copy.

## Validation possible

- **Tier 1:** Table 1 reports expected maximum toxicity and toxicity probability
  for GPT-1/2/3 and CTRL, prompted and unprompted. Good targets — but they were
  produced with Perspective, so a local-classifier reproduction is measuring
  something else and must be recorded as such, not as a failed match.
- **Tier 2:** Apache-2.0 and modern-Python friendly; the scoring aggregation is
  simple enough that equivalence on cached scores is easy and worth doing.
- **Tier 3:** null and monotonicity apply to all three aggregations. Swap
  antisymmetry does not — these are rates over one set of generations, not
  paired comparisons. Document as an exemption.

## Known limitations of the metric itself

- Everything rests on the toxicity classifier, which has documented biases of
  its own against dialect and reclaimed terms — so a "more toxic" reading can
  reflect the classifier rather than the model.
- The 0.5 threshold is a convention, and all three metrics move with it.
- K interacts with all three: EMT and ToxicityProbability both increase with K
  by construction, so cross-paper comparison requires identical K.

## RealToxicityPrompts paper reproduction

The normal public metric remains `RealToxicityPrompts`. It requires Perspective
API scoring or an explicitly named injected scorer; a substitute classifier is
an adaptation. Its LiteLLM chat generation is also a modern adaptation of the
paper's raw causal-LM protocol.

Research validation is an internal standalone path:
`scripts/paper/reproduce_realtoxicityprompts.py`. It accepts caller-supplied
official `prompts.jsonl`, starts with raw-causal local `gpt2`, seed 42, 25
samples, top-p 0.9, 20 new tokens, fp32, and the released default generation
batch size of 32. The released runner generated/scored all 99,442 source rows,
then Table 2 aggregated the 99,016 rows with usable prompt toxicity. Planning
does not download data/models.

The runner distinguishes `historical_score_parity` (requiring unreleased 2020
generated texts and scores), `current_perspective_reconstruction` (the
practical mode), and `substitute_scorer` (an adaptation). Perspective changes
over time and model/tokenizer revisions were unpinned, so Table 2 equality is
not promised. The released notebook's strict `> .5` prompt and continuation
thresholds, partial bundle retention, and sample SD are private reconstruction
behavior only; they do not alter corrected BiasScope scoring.
