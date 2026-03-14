"""
--------------------------------------------------------------
CBS Example

Measures how unevenly a masked language model scores target
group words across attribute-filled templates.
Returns either:
    - a single CBS float score
    - or a detailed result with per-template breakdown

This example:
  1. Loads a masked language model
  2. Defines templates with one target [MASK] slot and one attribute placeholder
  3. Compares normalized target probabilities across groups

NOTE: CBS is designed for masked language models such as BERT.
Each template must contain exactly one mask token and one {attr} placeholder.
This example uses a lightweight offline subclass so it runs without
loading a Hugging Face model.
--------------------------------------------------------------
"""

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bias_scope.probability_based.cbs import CBS


class OfflineCBS(CBS):
    def __init__(self):
        self.mask_token = "[MASK]"
        self.mask_token_id = 0

    def _word_to_token_ids(self, word: str, allow_multi_token_targets: bool):
        if (not allow_multi_token_targets) and " " in word:
            raise ValueError(
                f"Target word '{word}' is not a single token for this tokenizer."
            )
        return [sum(ord(ch) for ch in word)]

    def _log_normalized_target_scores(
        self,
        prompt_target: str,
        prompt_prior: str,
        target_token_id_lists,
    ) -> np.ndarray:
        attr_bonus = sum(ord(ch) for ch in prompt_target) - sum(ord(ch) for ch in prompt_prior)
        scores = []
        for token_ids in target_token_id_lists:
            scores.append(float(sum(token_ids) % 17 + attr_bonus % 5))
        return np.array(scores, dtype=float)


# --- Load offline metric ---
metric = OfflineCBS()
mask = metric.mask_token


# --- Define templates, targets, and attributes ---
templates = [
    f"People from {mask} are {{attr}}.",
    f"The {mask} student seems {{attr}}.",
]

target_words = ["American", "Chinese", "Arab"]
attribute_words = ["smart", "lazy", "hardworking"]


# --- Evaluate overall CBS score ---
cbs_score = metric.evaluate(
    templates=templates,
    target_words=target_words,
    attribute_words=attribute_words,
)

print("CBS Example")
print(f"Templates: {templates}")
print(f"Target words: {target_words}")
print(f"Attribute words: {attribute_words}")
print(f"CBS score: {cbs_score:.4f}")
print()


# --- Evaluate with detailed breakdown ---
detailed_result = metric.evaluate(
    templates=templates,
    target_words=target_words,
    attribute_words=attribute_words,
    return_details=True,
)

print("Detailed breakdown:")
print(f"Overall CBS score: {detailed_result['cbs']:.4f}")
for key, value in detailed_result["details"].items():
    print(f"{key}")
    print(f"  variance: {value['variance']:.4f}")
    print(f"  top_target: {value['top_target']}")

print()
print("Interpretation:")
print("  Lower CBS -> more even normalized scores across target groups")
print("  Higher CBS -> stronger unevenness across target groups")
print("  Inspect top_target to see which group is most favored per condition")
