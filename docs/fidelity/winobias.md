# WinoBias

**Cited source:** Zhao, Wang, Yatskar, Ordonez & Chang 2018, *Gender Bias in
Coreference Resolution: Evaluation and Debiasing Methods*, NAACL 2018.
[arXiv:1804.06876](https://arxiv.org/abs/1804.06876).
**Reference implementation:**
[uclanlp/corefBias](https://github.com/uclanlp/corefBias) @ `0bce984dd081`,
MIT — data and swap lists both usable.
**Sections and files read:** abstract and §1 (the pro/anti F1 gap, "an average
difference of 21.1 in F1 score"); the definitions of the pro- and
anti-stereotypical conditions and of Type 1 vs Type 2 sentences;
`WinoBias/wino/data/*.txt.{dev,test}` (the bracketed-span format).
**Family / access:** prompt / `chat`
**Status: new in v0.2.0** (PLAN.md 7.2).

## Definition in the paper

Each sentence mentions two occupations and one pronoun. In the
**pro-stereotypical** condition the pronoun co-refers with the occupation whose
stereotypical gender it matches; the **anti-stereotypical** sentence is the same
sentence with the link reversed. The bias effect is

```
gap = accuracy_pro − accuracy_anti
```

Zhao et al. report an **average gap of 21.1 F1**. Zero means the system resolves
both conditions equally well. Type 1 requires world knowledge (no syntactic
cue); Type 2 is resolvable from syntax alone, and the paper reports them
separately.

Because every item has exactly one correct antecedent and the splits are the
same sentences, accuracy and F1 coincide on this dataset — the paper reports F1
because its systems emitted coreference clusters, not because the quantity
differs here.

## Verdict

**faithful.**

The implemented statistic is the paper's gap, computed over the paper's own
data files. Items are paired one-to-one and the metric **refuses unequal
splits**, because comparing a 3-item pro split against a 2-item anti split
would compare different sentences.

Answer matching is deliberately lenient about case, articles and trailing
punctuation: a model answering "the developer" for "[The developer]" is
correct, and counting that wrong would inflate the measured bias with a parsing
artefact rather than a coreference failure.

## The caveat that must travel with any number

**The 21.1 figure is for 2018-era coreference systems**, not prompted LLMs. A
chat model answering a multiple-choice coreference question is doing a different
task through a different interface. Reproducing "21.1" is not the goal and a
mismatch against it is not a failure — the honest record for an LLM run is a
fresh number with `no_published_reference`, not a comparison.

## Required action

- Add the encoder path: the same items scored by likelihood comparison, which
  is closer to the original setting and works with `logits` access. Until then
  the metric is chat-only.
- Vendor or load the Type 1 and Type 2 splits with SHA-256s so a run states
  which split it used. `load_winobias_file` reads them from the clone today,
  which is fine for development and not for a recorded result.

## Validation possible

- **Tier 1:** the paper's per-system Type 1 / Type 2 tables and the 21.1
  average — reachable only for a 2018-style coreference system, so for LLM runs
  record `no_published_reference` (see the caveat above).
- **Tier 2:** MIT licence and the scoring is a simple accuracy, so an
  equivalence check against `corefBias` on identical predictions is
  straightforward.
- **Tier 3:** null (a system with equal accuracy on both splits scores 0) and
  swap antisymmetry (exchanging the pro and anti splits negates the gap) both
  apply and are tested.

## Known limitations of the metric itself

- Binary gender throughout: the pronouns are he/she and the occupation lists are
  gendered accordingly.
- The occupation list comes from US Department of Labor statistics and is dated;
  the stereotype it encodes is a 2018 snapshot of one labour market.
- Type 2 is solvable from syntax, so a gap there indicates something different
  from a gap on Type 1 — reporting a single pooled number hides that, which is
  why the paper splits them and why `sentence_type` is recorded here.
