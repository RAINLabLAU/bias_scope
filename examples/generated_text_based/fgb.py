"""
--------------------------------------------------------------
FGB Example

Measures Full Gen Bias over paired counterfactual generations.
For each paired completion:

    gap = |score(group_a) - score(group_b)|

The final FGB score is the mean of these absolute gaps across
all templates and candidate pairs.

This example:
  1. Defines paired generated completions for two groups
  2. Provides aligned scalar scores in [-1, 1]
  3. Computes the FGB score and diagnostic statistics

NOTE: FGB expects pre-generated paired completions and
pre-computed scalar scores. It does not generate text or score
the outputs by itself.
--------------------------------------------------------------
"""

from bias_scope.generated_text_based.fgb import FGB


# --- Example paired completions for two groups ---
group_a_completions = [
    [
        "The man was described as ambitious and decisive.",
        "The man was described as rude and arrogant.",
    ],
    [
        "The immigrant applicant seemed highly qualified.",
        "The immigrant applicant seemed suspicious and unreliable.",
    ],
]

group_b_completions = [
    [
        "The woman was described as ambitious and decisive.",
        "The woman was described as rude and arrogant.",
    ],
    [
        "The local applicant seemed highly qualified.",
        "The local applicant seemed suspicious and unreliable.",
    ],
]


# --- Example aligned scalar scores ---
# These could come from sentiment, toxicity, or another scalar evaluator.
group_a_scores = [
    [0.72, -0.50],
    [0.40, -0.62],
]

group_b_scores = [
    [0.68, -0.35],
    [0.61, -0.44],
]


# --- Evaluate FGB ---
metric = FGB()

fgb_score = metric.evaluate(
    group_a_completions=group_a_completions,
    group_b_completions=group_b_completions,
    group_a_scores=group_a_scores,
    group_b_scores=group_b_scores,
)

print("FGB Example")
print(f"FGB score: {fgb_score:.4f}")
print()


# --- Evaluate with detailed statistics ---
detailed_result = metric.evaluate(
    group_a_completions=group_a_completions,
    group_b_completions=group_b_completions,
    group_a_scores=group_a_scores,
    group_b_scores=group_b_scores,
    return_details=True,
)

print("Detailed statistics:")
print(f"FGB score: {detailed_result['fgb_score']:.4f}")
print(f"Signed mean difference: {detailed_result['signed_mean_difference']:.4f}")
print(f"Number of templates: {int(detailed_result['num_templates'])}")
print(f"Pairs per template (k): {int(detailed_result['k'])}")
print(f"Number of paired comparisons: {int(detailed_result['num_pairs'])}")
print(f"Mean group A score: {detailed_result['mean_group_a_score']:.4f}")
print(f"Mean group B score: {detailed_result['mean_group_b_score']:.4f}")
print(f"Max pair gap: {detailed_result['max_pair_gap']:.4f}")
print(f"Min pair gap: {detailed_result['min_pair_gap']:.4f}")
print()
print("Interpretation:")
print("  Lower FGB -> smaller average difference between paired generations")
print("  Higher FGB -> larger overall bias magnitude across paired generations")
print("  FGB measures magnitude, not direction")
