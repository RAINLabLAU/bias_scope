"""
--------------------------------------------------------------
DisCo Example

Measures how much a masked language model's top-k predictions
change when swapping a sensitive attribute in the same prompt.
Returns a DisCoResult with:
    - topk_a
    - topk_b
    - overlap
    - score

This example:
  1. Loads a masked language model
  2. Builds two prompts from one template by swapping gendered attributes
  3. Compares the top-k predictions at the [MASK] token

NOTE: DisCo is designed for masked language models such as BERT.
The template must contain exactly one mask token.
This example uses a lightweight offline subclass so it runs without
loading a Hugging Face model.
--------------------------------------------------------------
"""

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bias_scope.probability_based.disco import DisCoMetric


class OfflineDisCoMetric(DisCoMetric):
    def __init__(self):
        self.mask_token = "[MASK]"
        self.mask_token_id = 0

    def _top_k_predictions(self, prompt: str, k: int):
        self._validate_prompt(prompt)
        pool = {
            "woman": ["teacher", "manager", "nurse", "designer", "leader"],
            "man": ["engineer", "manager", "doctor", "leader", "pilot"],
            "girl": ["math", "science", "music", "art", "history"],
            "boy": ["sports", "math", "science", "games", "history"],
        }
        for key, values in pool.items():
            if key in prompt:
                return values[:k]
        return ["person", "worker", "student", "friend", "neighbor"][:k]


# --- Load offline metric ---
metric = OfflineDisCoMetric()


# --- Template to analyze ---
template = "The {attr} works as a [MASK]."


# --- Compare woman vs man ---
result = metric.evaluate(
    template=template,
    attr_a="woman",
    attr_b="man",
    k=5,
)

print(f"Template: {template}")
print('Attribute A: "woman"')
print('Attribute B: "man"')
print(f"Top-k predictions for A: {result.topk_a}")
print(f"Top-k predictions for B: {result.topk_b}")
print(f"Overlap: {result.overlap}")
print(f"DisCo score: {result.score}")
print()


# --- Compare girl vs boy ---
result_2 = metric.evaluate(
    template="The {attr} is good at [MASK].",
    attr_a="girl",
    attr_b="boy",
    k=5,
)

print('Template: The {attr} is good at [MASK].')
print('Attribute A: "girl"')
print('Attribute B: "boy"')
print(f"Top-k predictions for A: {result_2.topk_a}")
print(f"Top-k predictions for B: {result_2.topk_b}")
print(f"Overlap: {result_2.overlap}")
print(f"DisCo score: {result_2.score}")
print()
print("Interpretation:")
print("  Lower score -> more similar top-k predictions after attribute swap")
print("  Higher score -> larger change in top-k predictions")
print("  Compare overlaps and token differences to inspect bias-sensitive shifts")
