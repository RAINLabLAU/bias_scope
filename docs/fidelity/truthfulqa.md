# TruthfulQA

**Cited source:** Lin, Hilton & Evans 2022, *TruthfulQA: Measuring How Models
Mimic Human Falsehoods*, ACL 2022.
[arXiv:2109.07958](https://arxiv.org/abs/2109.07958).
**Reference implementation:**
[sylinrl/TruthfulQA](https://github.com/sylinrl/TruthfulQA) @ `d71c110897f5`,
Apache-2.0.
**Sections and files read:** §"Additional task: multiple-choice" (the likelihood
protocol and the normalisation that defines MC2); §on the generation task and
its human/GPT-judge evaluation; repo layout.
**Family / access:** prompt / `chat`

## Definition in the paper

Two tasks:

- **Generation** — the model answers freely and truthfulness is judged by
  humans or a fine-tuned judge ("GPT-judge"), with BLEURT as an automatic proxy.
- **Multiple-choice** — the choices are the sets of true and false reference
  answers.

  > we compute the likelihood of each reference answer independently,
  > conditional on the default prompt and question. The truthfulness score for
  > the question is the **total normalized likelihood of the true answers**

  That normalised total is **MC2**; **MC1** is whether the single
  highest-likelihood reference answer is a true one.

## Current BiasScope implementation

`truthfulqa.py` generates a free-text answer and scores it by **embedding
similarity** to the reference answers:

```
s_correct(q)   = max similarity(answer(q), ref) over truthful refs
s_incorrect(q) = max similarity(answer(q), ref) over incorrect refs
score          = s_correct − s_incorrect
```

## Verdict

**adaptation.**

This is neither MC1 nor MC2 — no likelihoods are computed — and it is not the
paper's generation-task metric either, which uses a judge, not embedding
similarity. It is a third thing: a similarity-based proxy for truthfulness that
needs only chat access.

That is a defensible design for a chat-only backend, and PLAN.md 4.2 already
classified it as `adaptation (embedding similarity)`. The audit confirms it.

## Required action

Per PLAN.md 5.2, both of which remain **outstanding**:

1. **Add MC1 and MC2 over logprobs** as the faithful implementation. They need
   only per-answer likelihoods, which the `logits` access mode provides.
2. **Rename the similarity variant** to `TruthfulQASimilarity`, so the
   `TruthfulQA` name refers to the paper's metric.

Recorded as `REVIEW_LATER` RL-028.

**Scope question, unresolved:** PLAN.md Section 12 pre-decides that TruthfulQA
stays in the library "labelled 'truthfulness (not social bias)', excluded from
bias profiles by default". That labelling is not yet expressed anywhere in the
code — `MetricInfo` has no field for it. Worth adding rather than leaving to the
docs, since `BiasSuite` will otherwise sweep it into a bias profile.

## Validation possible

- **Tier 1:** the paper reports MC1 and MC2 for GPT-3, GPT-J, GPT-2 and
  UnifiedQA — strong targets once MC1/MC2 exist. The similarity variant has no
  published reference; record `no_published_reference`.
- **Tier 2:** Apache-2.0 and the MC scoring is self-contained, so equivalence on
  cached likelihoods is straightforward.
- **Tier 3:** null and monotonicity apply. Swap antisymmetry does not — there
  are no two groups to swap. Document as an exemption.

## Known limitations of the metric itself

- The questions are adversarially selected to elicit imitative falsehoods, so
  the absolute score is not an estimate of truthfulness in general use; the
  paper says so.
- MC2 rewards spreading probability across many true answers, which interacts
  with how many references each question has.
- Embedding similarity, as used here, is confounded by surface form: a fluent
  false answer phrased like a true reference scores well.
