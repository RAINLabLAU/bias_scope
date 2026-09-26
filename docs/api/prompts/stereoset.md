# StereoSet

<!-- metric-card:start -->
| | |
|---|---|
| Family | prompt |
| Model access | `chat` |
| Neutral value | 50 |
| Direction | signed; 0 is neutral |
| Range | 0 to 100 |
| Languages | en |
| Fidelity | **adaptation**: same comparison as the cited paper, but a different access mode or scoring path that can change the numbers. Two deviations from the likelihood protocol: the model makes a forced three-way A/B/C choice rather than having its likelihoods ranked pairwise, so lms and ss are prompt analogues rather than paper-equivalent scores; and scores are aggregated flat rather than averaged per target term. [Audit note](../../fidelity/stereoset_family.md). |
| Source | StereoSet: Measuring stereotypical bias in pretrained language models, ACL 2021 — https://arxiv.org/abs/2004.09456 |
| Reference code | https://github.com/moinnadeem/StereoSet @ ead7d086a64a |
<!-- metric-card:end -->


`StereoSetMetric` is a prompt-based A/B/C adaptation. Its forced-choice LMS and
SS are not paper-equivalent likelihood scores, and its results must not be
compared directly with published StereoSet tables. The paper-faithful evaluator
is intentionally private reproduction infrastructure.

::: bias_scope.prompts_based.stereoset.StereoSetMetric

## Example

```python
# ──────────────────────────────────────────────────────────────
# StereoSet — Prompt-based Stereotype Preference
#
# Measures whether a model prefers stereotype, anti-stereotype,
# or unrelated continuations in a three-way multiple-choice
# format.  Uses the StereoSet dataset with A/B/C prompts.
#
# Returns:
#   language_model_score — related-choice rate in [0, 100]
#   stereotype_score     — stereotype preference in [0, 100]
#   icat_score           — combined quality × fairness score
#   stereotype_rate      — fraction of stereotype choices
#   antistereotype_rate  — fraction of anti-stereotype choices
#   unrelated_rate       — fraction of unrelated choices
#
# NOTE: Requires a valid LLM API key and internet access.
#       Uses McGill-NLP/stereoset dataset from HuggingFace.
# ──────────────────────────────────────────────────────────────

from bias_scope.prompts_based import StereoSetMetric

# --- Initialize ---
metric = StereoSetMetric(
    model_name="openrouter/meta-llama/llama-3.1-8b-instruct",
    api_key="your-openrouter-key-here",
    task_type="intersentence",   # or "intrasentence"
    dataset_split="validation",
)

# --- Evaluate ---
# subset selects which bias type to test
# Valid subsets: gender, profession, race, religion
result = metric.evaluate(
    num_samples=10,
    subset="gender",
    num_option_permutations=2,   # reduce position bias by testing multiple orderings
)

print(f"Language Model Score:    {result['language_model_score']:.1f}")
print(f"Stereotype Score:       {result['stereotype_score']:.1f}")
print(f"ICAT Score:             {result['icat_score']:.1f}")
print(f"Stereotype rate:        {result['stereotype_rate']:.2%}")
print(f"Anti-stereotype rate:   {result['antistereotype_rate']:.2%}")
print(f"Unrelated rate:         {result['unrelated_rate']:.2%}")
print(f"Rows evaluated:         {result['num_rows_evaluated']}")
print(f"Selected subset:        {result['selected_subset']}")
print()
print("Per-category breakdown:")
for category, scores in result.get("per_category", {}).items():
    print(f"  {category}: LMS={scores['language_model_score']:.1f}, "
          f"SS={scores['stereotype_score']:.1f}, "
          f"ICAT={scores['icat_score']:.1f}")
print()
print("Interpretation:")
print("  LMS ~ 100     → model picks meaningful continuations (good)")
print("  SS  ~  50     → no stereotype preference (unbiased)")
print("  SS  >  50     → model prefers stereotypical continuations")
print("  ICAT = LMS × min(SS, 100-SS) / 50  (higher = better)")

```
