# FirstPersonFairness

**Cited source:** Eloundou, Beutel, Robinson, Gu-Lemberg, Brakman, Mishkin,
Shah, Heidecke, Weng, Kalai, "First-Person Fairness in Chatbots", OpenAI 2024 —
https://arxiv.org/abs/2410.19803 (read: the arXiv PDF)
**Reference implementation:** **none located.** Search log in
`sources/SOURCES.yaml`; PLAN.md Section 4.0's procedure was followed and
returned nothing.
**Sections and files read:** Sec. 3.1 (name-substitution response generation),
Sec. 3.2 (the nine quality criteria — not implemented, see below), **Sec. 3.3
(the harmful-stereotype pair metric, equations 1-3 and H(A,B))**, Figure 3 (the
LMRA template), Sec. 3.3 "Addressing LMRA over-sensitivity", Sec. 4.2 and 4.3
(results and the human-agreement study)
**Family / access:** prompt / `chat`

## Definition in the paper

Sec. 3.3. For a prompt `x`, with `y_A` and `y_B` the responses generated for
names from groups A and B, the rating `h(x, {y_A, y_B}, A, B)` is `y_A` if the
`y_A`-`y_B` assignment would be a harmful A-B stereotype, `y_B` for the reverse,
and `⊥` if neither — or if `y_A = y_B`. Then

    h_F(x, A, B) = Pr[h(x, {y_A, y_B}, A, B) = y_A]        (1)
    h_R(x, A, B) = Pr[h(x, {y_A, y_B}, A, B) = y_B]        (2)
    h(x, A, B)   = h_F(x, A, B) − h_R(x, A, B)             (3)
    H(A, B)      = E_{x~φ}[h(x, A, B)]

The probabilities come from "single-token probabilities", and each pair is
judged twice with the responses swapped "to avoid order bias".

## Current BiasScope implementation (v0.2.0)

`src/bias_scope/prompts_based/first_person_fairness.py`. `net_harm()` is
equations (1)-(3); `FirstPersonFairness.evaluate()` is the expectation over
prompts. `HARMFUL_STEREOTYPE_TEMPLATE` is Figure 3.

## Verdict

**adaptation.**

The estimator is the paper's exactly — the net score, the swapped second pass
with its option crossing, the identical-response rule, and the reporting of
`H_forward` and `H_reverse` beside `H`. What is missing is the prompt. PLAN.md
7.2 requires the authors' judge instructions to be mirrored verbatim, and the
paper prints Figure 3 marked **"slightly abbreviated"**; the production template
is not published, and no repository, appendix data release, or artefact was
found. So the wording cannot be shown to match, and the status stays
`adaptation` rather than `faithful`. Nothing else about the metric deviates.

The nine quality criteria of Sec. 3.2 and the Bias Enumeration Algorithm of
Sec. 3.4 are not implemented. Both depend on prompts in appendices that are
likewise abbreviated, and the Bias Enumeration Algorithm produces natural
language axis descriptions rather than a number, so it does not fit the
`BiasResult` contract at all.

## The subtraction is the whole method

Language model outputs are stochastic, so a stereotype-shaped pair appears by
chance even when both groups draw from the same distribution. The paper's own
case: two response types at 50/50 for both genders gives a 25% chance of a
stereotype pair — and an equal 25% chance of its reverse. `H_forward` alone
would report 0.25 for a model with no name effect whatsoever.

This is why `H_forward` and `H_reverse` are both in `details` and why a test
pins the 25/25 case at exactly 0. **A large `H_forward` reported without its
reverse is not a bias measurement.**

The same logic covers the judge: a rater that always answers "A" regardless of
which response it is shown first has expressed no preference, and the two
orderings cancel to `h = 0` with `h_F = h_R = 0.5`. Crossing the options on the
swapped call is what makes that work; getting it backwards would halve every
estimate and quietly restore the order bias. Both are pinned by tests.

## Validation

- **Tier 1:** the paper's numbers come from non-public ChatGPT conversations
  that cannot be redistributed, so no published cell is reproducible.
- **Tier 2:** not possible — no reference implementation and no released
  per-item outputs.
- **Tier 3:** null (identical response distributions give 0; the paper's chance
  case nets to 0), swap antisymmetry (exchanging the two responses negates `h`),
  and the order-bias cancellation above.

## Known limitations of the metric itself

- **Judge-bound, and the judge is the measurement.** Sec. 3.3 reports that
  naming which group got which response made the judge "label nearly any
  difference as a harmful stereotype", to the point of rating a pair and its
  swap both harmful. Hiding the assignment is a mitigation, not a fix, and
  results move with the judge model.
- **Human agreement is uneven.** Sec. 4.3 finds LMRA ratings correlate strongly
  with human ratings for **gender** and much less well for **race** and for the
  feature labels. A race-axis number from this metric carries less evidential
  weight than a gender one, and the paper says so.
- **Three stacked sources of randomness** — name sampling, response sampling,
  and rater variability — so a single run's `H` needs its interval, and `run()`
  supplies one.
- **The prompt distribution φ is the population.** The paper's `H` is an
  expectation over real user prompts; over a synthetic prompt set it measures
  that set, not chatbot use.
