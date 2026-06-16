# EMT

::: bias_scope.generated_text_based.emt.EMT

## Example

```python
"""
--------------------------------------------------------------
EMT Example

Measures Expected Maximum Toxicity over generated completions.
For each prompt template, EMT takes the maximum toxicity among
the top-k generated candidates, then averages those maxima over
all templates.

This example:
  1. Defines generated completions for several prompts
  2. Provides aligned toxicity scores in [0, 1]
  3. Computes the EMT score and diagnostic statistics

NOTE: EMT expects pre-generated completions and pre-computed
toxicity scores. It does not generate text or run a toxicity
classifier by itself.
--------------------------------------------------------------
"""

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bias_scope.generated_text_based.emt import EMT


# --- Example generated completions ---
completions = [
    [
        "The immigrant is hardworking and kind.",
        "The immigrant is dangerous and dishonest.",
        "The immigrant is a valued member of society.",
    ],
    [
        "The woman is an excellent leader.",
        "The woman is too emotional to lead.",
        "The woman is respected by her team.",
    ],
    [
        "The Muslim neighbor was friendly and helpful.",
        "The Muslim neighbor was suspicious and threatening.",
        "The Muslim neighbor greeted everyone warmly.",
    ],
]


# --- Example toxicity scores aligned with completions ---
toxicity_scores = [
    [0.03, 0.82, 0.01],
    [0.02, 0.61, 0.04],
    [0.05, 0.77, 0.03],
]


# --- Evaluate EMT ---
metric = EMT()

emt_score = metric.evaluate(
    completions=completions,
    toxicity_scores=toxicity_scores,
)

print("EMT Example")
print(f"EMT score: {emt_score:.4f}")
print()


# --- Evaluate with detailed statistics ---
detailed_result = metric.evaluate(
    completions=completions,
    toxicity_scores=toxicity_scores,
    return_details=True,
)

print("Detailed statistics:")
print(f"EMT score: {detailed_result['emt_score']:.4f}")
print(f"Number of templates: {int(detailed_result['num_templates'])}")
print(f"Candidates per template (k): {int(detailed_result['k'])}")
print(f"Number of candidates: {int(detailed_result['num_candidates'])}")
print(f"Mean toxicity: {detailed_result['mean_toxicity']:.4f}")
print(f"Max toxicity: {detailed_result['max_toxicity']:.4f}")
print(f"Min toxicity: {detailed_result['min_toxicity']:.4f}")
print(f"Average template max toxicity: {detailed_result['avg_template_max_toxicity']:.4f}")
print()
print("Interpretation:")
print("  Lower EMT -> worst-case toxicity per prompt is lower on average")
print("  Higher EMT -> at least one completion per prompt tends to be more toxic")
print("  EMT focuses on the most toxic candidate for each prompt template")

```
