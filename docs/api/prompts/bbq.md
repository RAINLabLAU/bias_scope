# BBQ

<!-- metric-card:start -->
| | |
|---|---|
| Family | prompt |
| Model access | `chat` |
| Neutral value | 0 |
| Direction | signed; 0 is neutral |
| Range | -1 to 1 |
| Languages | en |
| Fidelity | **adaptation**: same comparison as the cited paper, but a different access mode or scoring path that can change the numbers. Public chat A/B/C adaptation: it uses mutable external data and heuristic target reconstruction, so it cannot reproduce the pinned official dataset, target_loc-based name/intersectional scoring, or paper-model inference. [Audit note](../../fidelity/bbq.md). |
| Source | BBQ: A Hand-Built Bias Benchmark for Question Answering, Findings of ACL 2022 — https://arxiv.org/abs/2110.08193 |
| Reference code | https://github.com/nyu-mll/BBQ @ bea11bd97d79 |
<!-- metric-card:end -->


> `BBQMetric` is a chat adaptation, not a paper-model reproduction. Its A/B/C
> choices can be scored categorically, but official result reproduction is
> provided privately through `_bbq_reproduction` with caller-supplied pinned
> artifacts and authoritative `target_loc` metadata.

::: bias_scope.prompts_based.bbq.BBQMetric

## Example

```python
# ──────────────────────────────────────────────────────────────
# BBQ — Bias Benchmark for Question Answering
#
# Chat adaptation for categorical BBQ-style answers. The returned signed score
# is not a reproduction of the paper's historical model inference or pinned
# artifact protocol; private helpers provide that caller-supplied path.
#
# Returns:
#   bias_score — signed stereotype-alignment score (-1 to +1; 0 neutral)
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

```
