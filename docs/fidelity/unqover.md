# UnQoverMetric

**Cited source:** Li, Patel, Du, Clark & Sabharwal 2020, *UnQovering
Stereotyping Biases via Underspecified Questions*, Findings of EMNLP 2020.
[arXiv:2010.02428](https://arxiv.org/abs/2010.02428).
**Reference implementation:**
[allenai/unqover](https://github.com/allenai/unqover) @ `3e47969b78ac`,
Apache-2.0.
**Sections and files read:** §4.1.1 (positional dependence, eq. 2 `δ`); §4.1.2
(attribute/negation independence, eqs. 3-4 `ε`); §4.2 (the bias score `C`);
§"Subject-Attribute Bias" (eq. 7 `γ`); §"Model Bias Intensity" (eq. 8 `μ`);
§"Count-based Metric" (eq. 9 `η`); repo layout.
**Family / access:** prompt / `chat`

## Definition in the paper

UnQover's contribution is that a naive comparison of model scores is confounded
by two reasoning errors, both of which it measures and cancels:

- **Positional dependence** (eq. 2): `δ = avg |S(x₁|τ₁,₂(a)) − S(x₁|τ₂,₁(a))|`
  — the same subject scored differently purely by its position.
- **Attribute independence** (eqs. 3-4): a robust model should satisfy
  `S(x₁|τ₁,₂(a)) = S(x₂|τ₁,₂(ā))` under negation; `ε` measures the violation.

Hence the **four prompt variants** per example: subject order flipped × question
negated. The bias score `C` is built to cancel both errors, then aggregated:

```
γ(x₁,a) = avg over x₂, τ of C(x₁,x₂,a,τ)          (7)
μ       = avg over x₁ of max over a of |γ(x₁,a)|  (8)   "model bias intensity"
η(x₁,a) = avg over x₂, τ of sgn[C(x₁,x₂,a,τ)]     (9)   count-based
```

`μ ∈ [0, 1]`; the paper introduces `η` because "a few high scoring outliers can
skew our bias estimates".

## Current BiasScope implementation

`unqover.py` implements the four-variant design and reports `bias_intensity`
(`μ`, eq. 8) and `count_bias_intensity` (`η`, eq. 9), plus the positional and
attribute error diagnostics. The aggregation structure matches the paper
closely — including the choice to report both `μ` and `η`, which is the paper's
own recommendation and is easy to skip.

## Verdict

**adaptation.**

The metric machinery is faithful. The deviation is in **access mode**: the paper
scores QA models by answer-span probability and masked LMs by fill probability,
while this class asks a chat model an A/B question and reads **token
logprobs** for the two option letters (`_extract_ab_probabilities`).

That is a real difference, not a cosmetic one. `S(x|τ)` in the paper is the
model's confidence in a *subject* as the answer; here it is the model's
probability of emitting a *letter*. Those are correlated but not the same
quantity, and the letter probabilities are sensitive to prompt formatting in a
way span probabilities are not — which is exactly the class of confound
UnQover was designed to remove.

## Required action

- State the access-mode deviation in the class docstring, not just here.
- Add the masked-LM path (the paper's §3.2 formulation), which the library can
  support directly through the `logits` access mode and which would be
  `faithful`.
- Chat backends that expose no logprobs cannot run this metric at all; that
  should be an explicit, early error rather than a downstream failure.

## Validation possible

- **Tier 1:** the paper reports `μ` and `η` for RoBERTa, BERT and DistilBERT
  across gender, nationality, ethnicity and religion. Reachable via the masked-LM
  path; not comparable via the chat path.
- **Tier 2:** Apache-2.0 and modern-Python friendly — a good equivalence
  candidate once the masked-LM path exists.
- **Tier 3:** null (identical subjects → `γ = 0`) and swap antisymmetry
  (swapping `x₁` and `x₂` negates `γ`) both apply, and swap antisymmetry is the
  sharp test here because cancelling positional dependence is the metric's whole
  purpose.

## Known limitations of the metric itself

- `μ` takes a max over attributes, so it is sensitive to a single extreme
  attribute; `η` exists to complement it and both should be reported.
- The subject and attribute lists determine the result, and the paper's are
  English and US-centric.
- Underspecified questions have no correct answer by construction, so the metric
  measures preference, not error.
