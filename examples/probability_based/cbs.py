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
--------------------------------------------------------------
"""

from bias_scope.probability_based.cbs import CBS


# --- Load masked language model ---
metric = CBS(model_name="bert-base-uncased")
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
