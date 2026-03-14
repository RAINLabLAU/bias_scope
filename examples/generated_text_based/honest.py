"""
--------------------------------------------------------------
HONEST Example

Measures hurtfulness in generated completions using a hurtful
lexicon. A completion is counted as hurtful if it contains at
least one term from the lexicon.

Equation:
    HONEST = hurtful_count / (|T| * K)

This example:
  1. Defines generated completions
  2. Provides a hurtful lexicon with optional category labels
  3. Computes the HONEST score and diagnostic statistics

NOTE: This is a lexicon-based metric. It does not require a model.
--------------------------------------------------------------
"""

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bias_scope.generated_text_based.honest import HONEST


# --- Example generated completions ---
completions = [
    [
        "The woman is a caring teacher.",
        "The woman is an idiot.",
        "The woman is a respected doctor.",
    ],
    [
        "The immigrant is hardworking and kind.",
        "The immigrant is a thief.",
        "The immigrant is dangerous and worthless.",
    ],
]


# --- Example hurtful lexicon ---
# Dict form lets the metric report category-specific rates.
hurtlex = {
    "idiot": "derogatory",
    "thief": "crime",
    "worthless": "dehumanizing",
    "dangerous": "threat",
}


# --- Evaluate HONEST ---
metric = HONEST()

honest_score = metric.evaluate(
    completions=completions,
    hurtlex=hurtlex,
)

print("HONEST Example")
print(f"HONEST score: {honest_score:.4f}")
print()


# --- Evaluate with detailed statistics ---
detailed_result = metric.evaluate(
    completions=completions,
    hurtlex=hurtlex,
    return_details=True,
)

print("Detailed statistics:")
print(f"HONEST score: {detailed_result['honest_score']:.4f}")
print(f"Hurtful percentage: {detailed_result['hurtful_percentage']:.2f}")
print(f"Number of templates: {int(detailed_result['num_templates'])}")
print(f"Candidates per template (k): {int(detailed_result['k'])}")
print(f"Number of candidates: {int(detailed_result['num_candidates'])}")
print(f"Number of hurtful candidates: {int(detailed_result['num_hurtful_candidates'])}")
print(f"Average template hurtful rate: {detailed_result['avg_template_hurtful_rate']:.4f}")

for key, value in detailed_result.items():
    if key.startswith("category_rate::"):
        print(f"{key}: {value:.4f}")

print()
print("Interpretation:")
print("  Lower HONEST -> fewer generated candidates contain hurtful lexicon terms")
print("  Higher HONEST -> more generated candidates contain hurtful lexicon terms")
print("  Category rates help identify the type of harmful content detected")
