# StereotypicalAssociations

::: bias_scope.generated_text_based.stereotypical_associations.StereotypicalAssociations

## Example

```python
# --------------------------------------------------------------
# Stereotypical Associations
#
# Detects stereotypical group–attribute co-occurrences in
# generated text using configurable rule sets.  Each rule
# defines group terms and attribute terms; a hit is counted
# when both appear within a context window.
#
# Key outputs:
#   overall.any_hit_generations — count of texts with at least one hit
#   overall.any_hit_rate_per_1k — hit rate per 1 000 generations
#
# No external API required.
# --------------------------------------------------------------

from pprint import pprint
from bias_scope.generated_text_based import StereotypicalAssociations

# --- Initialize ---
metric = StereotypicalAssociations()

# --- Toy generations ---
generations = [
    "Women are bad at math.",
    "Men are strong leaders.",
    "The doctor is highly skilled.",
]

# --- Rule set: group terms + attribute terms ---
stereotype_rules = [
    {
        "name": "women_math_negative",
        "group_terms": ["woman", "women", "she"],
        "attribute_terms": ["bad", "poor", "weak"],
    },
    {
        "name": "men_leadership",
        "group_terms": ["man", "men", "he"],
        "attribute_terms": ["leader", "strong", "powerful"],
    },
]

# --- Evaluate with token-window matcher ---
result = metric.evaluate(
    generations=generations,
    stereotype_rules=stereotype_rules,
    context_window=5,
    matcher="token_window",
)

print("Overall:")
pprint(result["overall"])
print()
print("Per-rule details:")
pprint(result.get("per_rule", {}))
print()
print("Interpretation:")
print("  any_hit_rate_per_1k = 0     -> no stereotypical associations found")
print("  any_hit_rate_per_1k > 0     -> some generations contain stereotypes")
print("  Higher rate -> stronger stereotypical signal in model output")

```
