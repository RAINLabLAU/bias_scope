# ──────────────────────────────────────────────────────────────
# Counterfactual Fairness
#
# Measures whether a model's outputs remain semantically
# consistent when only the identity term in a prompt changes
# (e.g., "Muslim" → "Christian").  Both prompts are sent to the
# model; the two responses are embedded and their cosine
# similarity is computed.
#
# Returns:
#   avg_similarity      — mean cosine similarity (1 = identical)
#   bias_score          — 1 - avg_similarity (0 = no bias)
#   low_similarity_rate — fraction of pairs below 0.8 threshold
#   per_identity_group  — avg similarity per identity term
#
# NOTE: Requires a valid LLM API key and internet access.
#       Uses StereoSet (McGill-NLP/stereoset) from HuggingFace.
#       Also downloads all-MiniLM-L6-v2 for response embedding.
# ──────────────────────────────────────────────────────────────

from bias_scope.prompts_based import CounterfactualFairness

# --- Initialize ---
metric = CounterfactualFairness(
    model_name="huggingface/meta-llama/Meta-Llama-3-8B-Instruct",
    api_key="your-api-key-here",
    dataset_config="intersentence",
    dataset_split="validation",
)

# --- Evaluate ---
# subset selects which StereoSet bias type to test
# Valid subsets: gender, profession, race, religion
result = metric.evaluate(
    num_samples=10,
    subset="religion",
)

print(f"Avg similarity:      {result['avg_similarity']:.3f}")
print(f"Bias score:          {result['bias_score']:.3f}")
print(f"Low similarity rate: {result['low_similarity_rate']:.2%}")
print(f"Pairs evaluated:     {result['num_pairs_evaluated']}")
print(f"Selected subset:     {result['selected_subset']}")
print()
print("Per-identity group:")
for identity, sim in result["per_identity_group"].items():
    print(f"  {identity:>15s}: {sim:.3f}")
print()
print("Interpretation:")
print("  avg_similarity = 1.0   → identical responses for swapped identities (ideal)")
print("  bias_score = 0.0       → no identity-dependent variation")
print("  low_similarity_rate    → fraction of pairs with very different outputs")
print("  Compare per_identity_group to find most affected identity terms")
