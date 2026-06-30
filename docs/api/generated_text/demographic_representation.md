# DemographicRepresentation

::: bias_scope.generated_text_based.demographic_representation.DemographicRepresentation

## Example

```python
# --------------------------------------------------------------
# Demographic Representation
#
# Measures how demographic groups are represented in generated
# text by counting group-term mentions.  Optionally compares
# observed proportions to a reference distribution using KL
# divergence and Jensen-Shannon divergence.
#
# Key outputs:
#   distribution — observed mention proportions per group
#   reference    — comparison metrics (JSD, KL) if compare_to given
#
# No external API required.
# --------------------------------------------------------------

from pprint import pprint
from bias_scope.generated_text_based import DemographicRepresentation

# --- Initialize ---
metric = DemographicRepresentation()

# --- Toy generations ---
generations = [
    "The man walked to work.",
    "The woman drove home.",
    "A man and woman talked.",
]

group_lexicons = {
    "male": ["man", "he", "his"],
    "female": ["woman", "she", "her"],
}

# --- Evaluate with reference comparison ---
result = metric.evaluate(
    generations=generations,
    group_lexicons=group_lexicons,
    compare_to={"male": 0.5, "female": 0.5},
)

print("Distribution:")
pprint(result["distribution"])
print()
print("Reference comparison:")
pprint(result["reference"])
print()
print("Interpretation:")
print("  JSD = 0    -> observed distribution matches the reference exactly")
print("  JSD > 0    -> divergence from the reference (higher = more skewed)")
print("  KL  = 0    -> no information loss from reference to observed")

```
