# What gets into BiasScope

Every metric in the library met all five criteria below. Every metric that was
considered and rejected is in [the roadmap](roadmap.md) with the criterion it
failed. Both lists exist so the answer to "why is X missing?" is a citation
rather than an opinion.

## The five criteria

**1. Peer-reviewed, or demonstrably adopted.**
A paper at a refereed venue, or a method in wide use with a public artefact —
Anthropic's political-neutrality evaluation has no paper but a released
repository, a stated protocol and third-party use, so it qualifies. A blog post
describing a number with no protocol does not.

**2. The formula or the reference code is fully specified.**
Enough to implement without guessing. If the paper leaves a step open, the
authors' code must close it; if neither does, the metric is not implementable
and a `MetricInfo` claiming otherwise would be false. FGB and PGB are in the
library only as `mismatch` for exactly this reason: HolisticBias's 217-class
style classifier was never released, so the published numbers cannot be
reproduced by anyone.

**3. It fits one of the four families.**
`embedding`, `probability`, `generated_text`, `prompt` — defined by what the
metric reads from the model, not by what it is about. A metric that needs
gradients, activations, or training-time intervention is out of scope; the
families exist to make the access requirements (`embeddings` / `logits` /
`completions` / `chat`) checkable before a run starts.

**4. No proprietary-only dependency.**
Everything must be runnable with open components. A judge model is fine —
judge-based metrics take a callable, so any LiteLLM-reachable model works and
the choice is recorded in the protocol. A metric that only works against one
company's private API, or that needs a classifier the authors never released,
is not.

**5. It measures something about social bias, or is explicitly labelled
otherwise.**
TruthfulQA and TofNof are in the library, labelled "truthfulness (not social
bias)", and `recommend_metrics` excludes them from bias profiles by default.
Keeping them is useful; presenting them as bias metrics is not.

## What inclusion is not

**Inclusion is not endorsement.** A metric can meet all five criteria and still
be a poor measurement — the literature on CrowS-Pairs and StereoSet item
validity (Blodgett et al. 2021) is the standard example, and it is cited in
those metrics' fidelity notes. Every `docs/fidelity/<metric>.md` ends with a
"Known limitations of the metric itself" section for this reason.

**Inclusion is not a fidelity claim.** A metric enters the library with an
audited status, and three of the five are not "this is the paper's metric":

- `faithful` — matches the paper, or matches the reference code where the two
  differ, with the difference documented.
- `adaptation` — a stated, reasoned deviation. FirstPersonFairness's estimator
  is exact but its judge template was published only in abbreviated form, so it
  is an adaptation.
- `original` — a reasonable statistic that is **not** the cited paper's metric.
  These were mislabelled through v0.1.1 and were renamed rather than deleted
  (PLAN.md Section 12).
- `mismatch` — known not to reproduce the paper. Two remain, both blocked on an
  unreleased classifier.
- `unaudited` — sources could not be read. None remain: the one paper once
  thought to be paywalled turned out to be open access, and its metric
  (`SentenceBiasScore`) has since been audited.

**Never presented as the cited paper's metric:** an `adaptation`, an `original`,
or a `mismatch`. That is a Section 1 rule, not a style preference.

## The gate before implementation

No metric is audited, renamed, or implemented before its paper *and* the
authors' code have been read and recorded in `sources/SOURCES.yaml` with section
numbers and file ranges. `scripts/sources/check_manifest.py` enforces the
record, and a metric whose manifest entry is still `pending` cannot be given a
fidelity status. Where no code exists, the manifest carries `code_status:
none_found` and the log of searches that established it.
