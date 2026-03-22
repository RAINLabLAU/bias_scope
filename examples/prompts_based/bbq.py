# ──────────────────────────────────────────────────────────────
# BBQ — Bias Benchmark for Question Answering
#
# Measures whether a model relies on stereotypes when answering
# ambiguous multiple-choice questions.  Each question has a
# context with missing information and three answer options;
# the correct answer is always "Can't be determined."  Choosing
# a stereotyped subject instead reveals bias.
#
# Returns:
#   bias_score — fraction of non-correct answers (0 = no bias)
#   accuracy   — fraction of correct answers (1 = perfect)
#   per_category — bias score broken down by BBQ category
#
# NOTE: Requires a valid LLM API key and internet access.
#       Uses the heegyu/bbq dataset from HuggingFace.
# ──────────────────────────────────────────────────────────────

from bias_scope.prompts_based import BBQMetric

# --- Initialize ---
metric = BBQMetric(
    model_name="openrouter/meta-llama/llama-3.1-8b-instruct",
    api_key="your-openrouter-key-here",
)

# --- Evaluate ---
# subset selects which BBQ category to test
# Valid subsets: Age, Disability_status, Gender_identity, Nationality,
#   Physical_appearance, Race_ethnicity, Race_x_SES, Race_x_gender,
#   Religion, SES, Sexual_orientation
result = metric.evaluate(
    num_samples=20,
    subset="Age",
)

print(f"Bias score:       {result['bias_score']:.2%}")
print(f"Accuracy:         {result['accuracy']:.2%}")
print(f"Rows evaluated:   {result['num_rows_evaluated']}")
print(f"Selected subset:  {result['selected_subset']}")
print()
print("Per-category bias scores:")
for category, score in result["per_category"].items():
    print(f"  {category:>25s}: {score:.2%}")
print()
print("Interpretation:")
print("  bias_score = 0%   → model always picks 'Can't be determined' (no bias)")
print("  bias_score = 100% → model never picks the correct answer")
print("  accuracy = 100%   → perfect on ambiguous questions")
print("  Compare across subsets to find category-specific biases")
