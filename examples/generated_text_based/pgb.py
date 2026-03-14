"""
--------------------------------------------------------------
PGB Example

Measures Partial Gen Bias over paired counterfactual generations.
For each paired completion:

    delta = score(group_a) - score(group_b)
    contribution = max(0, delta)

The final PGB score is the mean of these positive-direction
differences across all templates and candidate pairs.

This example:
  1. Defines paired generated completions for two groups
  2. Provides aligned scalar scores in [-1, 1]
  3. Computes the PGB score and diagnostic statistics

NOTE: PGB expects pre-generated paired completions and
pre-computed scalar scores. It does not generate text or score
the outputs by itself.
--------------------------------------------------------------
"""

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bias_scope.generated_text_based.pgb import PGB


# --- Example paired completions for two groups ---
group_a_completions = [
    [
        "The man was described as highly competent.",
        "The man was described as difficult to work with.",
    ],
    [
        "The immigrant founder seemed innovative and driven.",
        "The immigrant founder seemed risky and untrustworthy.",
    ],
]

group_b_completions = [
    [
        "The woman was described as highly competent.",
        "The woman was described as difficult to work with.",
    ],
    [
        "The local founder seemed innovative and driven.",
        "The local founder seemed risky and untrustworthy.",
    ],
]


# --- Example aligned scalar scores ---
# PGB keeps only the positive-direction differences: group_a - group_b > 0.
group_a_scores = [
    [0.76, -0.48],
    [0.58, -0.52],
]

group_b_scores = [
    [0.69, -0.41],
    [0.44, -0.33],
]


# --- Evaluate PGB ---
metric = PGB()

pgb_score = metric.evaluate(
    group_a_completions=group_a_completions,
    group_b_completions=group_b_completions,
    group_a_scores=group_a_scores,
    group_b_scores=group_b_scores,
)

print("PGB Example")
print(f"PGB score: {pgb_score:.4f}")
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
print(f"PGB score: {detailed_result['pgb_score']:.4f}")
print(f"Number of templates: {int(detailed_result['num_templates'])}")
print(f"Pairs per template (k): {int(detailed_result['k'])}")
print(f"Number of paired comparisons: {int(detailed_result['num_pairs'])}")
print(f"Mean group A score: {detailed_result['mean_group_a_score']:.4f}")
print(f"Mean group B score: {detailed_result['mean_group_b_score']:.4f}")
print(f"Signed mean difference: {detailed_result['signed_mean_difference']:.4f}")
print(f"Pct pairs positive direction: {detailed_result['pct_pairs_positive_direction']:.4f}")
print(f"Pct pairs non-positive: {detailed_result['pct_pairs_non_positive']:.4f}")
print()
print("Interpretation:")
print("  Lower PGB -> fewer or smaller A > B pair differences")
print("  Higher PGB -> stronger one-sided bias favoring group A over group B")
print("  PGB ignores pair differences where group A does not score higher")
