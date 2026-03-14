# ──────────────────────────────────────────────────────────────
# Demographic Representation Bias
#
# Detects gender bias by analysing pronoun distributions in
# model completions of occupation-based sentence templates from
# the WinoBias dataset.  Counts he/she/they pronouns across
# completions and computes:
#   representation_ratio — male/female pronoun ratio (1.0 = balanced)
#   l1_distance          — distance from uniform distribution
#   per_occupation       — pronoun breakdown per occupation
#
# NOTE: Requires a valid LLM API key and internet access.
# ──────────────────────────────────────────────────────────────

from bias_scope.prompts_based import DemographicRepresentationBias

# --- Initialize ---
metric = DemographicRepresentationBias(
    model_name="huggingface/meta-llama/Meta-Llama-3-8B-Instruct",
    api_key="your-api-key-here",
)

# --- Evaluate ---
# Uses WinoBias "type1_pro" subset by default
# num_templates=5 limits to 5 occupations (for speed)
# num_samples=3 generates 3 completions per template (use 50 for real eval)
result = metric.evaluate(
    num_templates=3,
    num_samples=1,
    subset="type1_pro",  # valid: type1_pro, type1_anti, type2_pro, type2_anti
)

print(f"Representation ratio (he/she): {result['representation_ratio']:.2f}")
print(f"L1 distance from uniform:      {result['l1_distance']:.3f}")
print()
print("Per-occupation pronoun distributions:")
for occupation, dist in result["per_occupation"].items():
    he = dist.get("he", 0.0)
    she = dist.get("she", 0.0)
    they = dist.get("they", 0.0)
    print(f"  {occupation:>20s}: he={he:.0%}  she={she:.0%}  they={they:.0%}")
print()
print("Interpretation:")
print("  ratio = 1.0  → equal he/she usage (balanced)")
print("  ratio > 1.0  → model over-uses 'he' (male-skewed)")
print("  ratio < 1.0  → model over-uses 'she' (female-skewed)")
print("  l1 = 0       → perfectly uniform {he, she, they} distribution")
print("  l1 → 1.33    → completely skewed toward one pronoun")
