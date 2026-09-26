# DemographicRepresentation

<!-- metric-card:start -->
| | |
|---|---|
| Family | generated_text |
| Model access | `completions` |
| Neutral value | 0 |
| Direction | higher means more biased |
| Range | 0 to 1 |
| Languages | en |
| Fidelity | **adaptation**: same comparison as the cited paper, but a different access mode or scoring path that can change the numbers. Faithful to HELM's formula. [Audit note](../../fidelity/demographic_representation.md). |
| Source | Holistic Evaluation of Language Models, TMLR 2023 — https://arxiv.org/abs/2211.09110 |
| Reference code | https://github.com/stanford-crfm/helm @ 63754d05db6f |
<!-- metric-card:end -->


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
