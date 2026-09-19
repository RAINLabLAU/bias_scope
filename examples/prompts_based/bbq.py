# ──────────────────────────────────────────────────────────────
# BBQ — Bias Benchmark for Question Answering
#
# Chat adaptation for categorical BBQ-style answers. For official artifact
# reproduction, use the private target_loc-based reproduction helpers.
#
# Returns a signed, chat-adaptation score. This is not official paper-model
# reproduction; private reproduction helpers use authoritative target_loc.
#   bias_score — signed stereotype-alignment score (0 = neutral)
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
print("  bias_score > 0  → more metadata-target-aligned selections")
print("  accuracy = 100%   → perfect on ambiguous questions")
print("  Compare across subsets to find category-specific biases")
