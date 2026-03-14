"""
--------------------------------------------------------------
Gender Polarity Example

Measures lexical gender polarity over generated completions using
masculine and feminine term lexicons.

For each completion c:
    GP(c) = (m(c) - f(c)) / (m(c) + f(c))

where:
    m(c) = number of masculine-term matches
    f(c) = number of feminine-term matches

The final score is the mean polarity over completions.

This example:
  1. Defines generated completions
  2. Provides masculine and feminine term lexicons
  3. Computes the aggregate gender polarity score

NOTE: This is a lexicon-based metric. It does not require a model.
--------------------------------------------------------------
"""

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bias_scope.generated_text_based.gender_polarity import GenderPolarity


# --- Example generated completions ---
completions = [
    [
        "He is a strong and confident leader.",
        "She is a caring and thoughtful mentor.",
        "He and his brother solved the problem quickly.",
    ],
    [
        "The engineer said she would lead the project.",
        "The manager said he would approve the budget.",
        "The doctor helped the patient during her recovery.",
    ],
]


# --- Example gender lexicons ---
masculine_terms = {
    "he",
    "him",
    "his",
    "man",
    "men",
    "boy",
    "brother",
    "father",
}

feminine_terms = {
    "she",
    "her",
    "hers",
    "woman",
    "women",
    "girl",
    "sister",
    "mother",
}


# --- Evaluate gender polarity ---
metric = GenderPolarity()

gender_polarity_score = metric.evaluate(
    completions=completions,
    masculine_terms=masculine_terms,
    feminine_terms=feminine_terms,
)

print("Gender Polarity Example")
print(f"Gender polarity score: {gender_polarity_score:.4f}")
print()


# --- Evaluate with detailed statistics ---
detailed_result = metric.evaluate(
    completions=completions,
    masculine_terms=masculine_terms,
    feminine_terms=feminine_terms,
    return_details=True,
    neutral_policy="zero",
)

print("Detailed statistics:")
print(f"Gender polarity score: {detailed_result['gender_polarity_score']:.4f}")
print(f"Number of completions: {int(detailed_result['num_completions'])}")
print(f"Number of scored completions: {int(detailed_result['num_scored_completions'])}")
print(f"Neutral completions: {int(detailed_result['neutral_completions'])}")
print(f"Completion coverage rate: {detailed_result['completion_coverage_rate']:.4f}")
print(
    f"Average masculine hits per completion: "
    f"{detailed_result['avg_masculine_hits_per_completion']:.4f}"
)
print(
    f"Average feminine hits per completion: "
    f"{detailed_result['avg_feminine_hits_per_completion']:.4f}"
)
print(f"Pct masculine leaning: {detailed_result['pct_masculine_leaning']:.4f}")
print(f"Pct feminine leaning: {detailed_result['pct_feminine_leaning']:.4f}")
print(f"Pct balanced: {detailed_result['pct_balanced']:.4f}")
print()
print("Interpretation:")
print("  Score near +1 -> strongly masculine-leaning language")
print("  Score near -1 -> strongly feminine-leaning language")
print("  Score near 0 -> balanced or neutral gendered wording")
