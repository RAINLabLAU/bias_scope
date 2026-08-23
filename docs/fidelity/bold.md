# BOLD (prompt-based)

**Cited source:** Dhamala, Sun, Kumar, Krishna, Pruksachatkun, Chang &
Gupta 2021, *BOLD: Dataset and Metrics for Measuring Biases in Open-Ended
Language Generation*, FAccT 2021.
[arXiv:2101.11718](https://arxiv.org/abs/2101.11718), read as
`sources/papers/bold.pdf`.
**Reference implementation:**
[amazon-science/bold](https://github.com/amazon-science/bold) @
`3ad652c773f5d1e30d5f6f61657ed934d768ecad`, CC-BY-SA-4.0.
**Sections and files read:** §"Bias metrics" — the paper augments "existing bias
metrics like sentiment and regard with novel bias metrics: psycholinguistic
norms, toxicity, and gender polarity" (`:124-126`); the sentiment subsection
naming VADER and its ±0.5 thresholds (`:303-317`); the repo's dataset and
scoring layout.
**Family / access:** prompt / `completions`

## Definition in the paper

BOLD is a **dataset plus a suite of five metrics**, not a single score. The
paper's contribution is 23,679 prompts across five domains (profession, gender,
race, religion, political ideology), evaluated with:

1. **Sentiment** — VADER compound score, with ±0.5 marking positive/negative.
2. **Toxicity** — a toxicity classifier.
3. **Regard** — Sheng et al. 2019's regard classifier.
4. **Psycholinguistic norms** — NRC-VAD weighted averages.
5. **Gender polarity** — unigram and embedding (`Wavg`/`Wmax`) variants.

Results are reported **per domain per metric**, not aggregated into one number.

## Current BiasScope implementation (v0.1.1)

`src/bias_scope/prompts_based/bold.py`. The docstring (`:30`, `:42`) is explicit
that it scores each continuation "with a lightweight lexical bias heuristic
`b(g(p))`", and `evaluate` applies `self._bias_score` to the prompt, the
reference text, and the generation (`:157-159`).

## Verdict

**faithful**, as of v0.2.0 — as a *benchmark runner*, which is what BOLD is.

**History: v0.1.1 was a `mismatch`.** The audit below is of that version.

BOLD defines no "lexical bias heuristic". The five metrics it does define are
all implemented elsewhere in this library already —`RegardScore`,
`PsycholinguisticNorms`, `GenderPolarity`, and the toxicity family — so the
class named `BOLD` is the one thing in the library that is *not* one of BOLD's
metrics. A user calling `BOLD(...).evaluate(...)` and citing Dhamala et al.
would be citing a paper for a statistic it does not contain.

## Resolution (v0.2.0)

1. `BOLD` is now a **runner**, not a metric: it loads prompts by domain,
   generates a continuation per prompt, and hands the continuations to whichever
   of the paper's five metrics the caller supplies. It reports
   `scores[domain][metric][group]` and the per-metric spread across groups.
2. **No aggregate score.** The paper reports per-domain per-metric values and so
   does this; collapsing them would be the composite bias score PLAN.md lists as
   a non-goal. `evaluate` deliberately returns no `bias_score` key.
3. **The lexical heuristic was removed, not renamed.** Unlike LPBS, DisCo and
   the HELM pair, there was no defensible statistic underneath it — an
   undocumented ad-hoc word-list score with no cited basis. PLAN.md Section 12's
   "keep under a new name" rule is for reasonable statistics that were merely
   mislabelled. This is a **breaking change** and is listed as such in
   `CHANGELOG.md`.
4. Nothing is wired in by default: a caller must name the scorers. That keeps
   the protocol block honest about which classifier produced each number, and
   an unrecognised scorer name is reported in `unknown_scorers` so a typo cannot
   masquerade as one of BOLD's five metrics.

The eight tests of the removed heuristic were deleted with it; the runner has
seventeen, and module coverage went from 92% to 98%.

## Required action (now largely done)

1. **Rebuild `BOLD` as a benchmark runner**, per PLAN.md 5.2: load the prompts
   by domain, generate, then feed the completions to the five existing metrics
   and return per-domain tables. It stops being a metric class and becomes an
   orchestration class — which is what the paper describes.
2. **Drop the lexical heuristic.** Unlike LPBS/DisCo/FGB/PGB, PLAN.md Section 12
   does not require preserving this one under a new name, and it should not be:
   it is an undocumented ad-hoc word-list score with no cited basis, not a
   defensible statistic worth keeping. Removing it is a breaking change and
   needs a `CHANGELOG` entry under "Breaking".
3. Vendor or load the BOLD prompt set (CC-BY-SA-4.0 — attribution and
   share-alike; note the share-alike term before vendoring into an MIT repo, and
   prefer a loader over redistribution). Tracked as `REVIEW_LATER` RL-014.

## Validation possible

- **Tier 1:** the paper reports per-domain per-metric values for GPT-2; those
  are targets for the five underlying metrics, not for the runner.
- **Tier 2:** the repo ships the dataset and analysis; equivalence is meaningful
  once the runner exists.
- **Tier 3:** properties apply to the five metrics individually. The runner
  itself is a feature, so it is verified by the feature criteria of Section 1,
  not the metric criteria.

## Known limitations of the metric itself

- Each of the five metrics inherits its classifier's own biases; the paper
  validates agreement with human judgement but does not eliminate this.
- The prompt set is Wikipedia-derived and English-only.
