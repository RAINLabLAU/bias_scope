# MarkedPersons

<!-- metric-card:start -->
| | |
|---|---|
| Family | generated_text |
| Model access | `completions` |
| Neutral value | 0 |
| Direction | higher means more biased |
| Range | 0 or more |
| Languages | en |
| Fidelity | **faithful**: same formula and protocol as the cited paper. [Audit note](../../fidelity/marked_persons.md). |
| Source | Marked Personas: Using Natural Language Prompts to Measure Stereotypes in Language Models, ACL 2023 — https://arxiv.org/abs/2305.18189 |
| Reference code | https://github.com/myracheng/markedpersonas @ 9b3ae82ad262 |
<!-- metric-card:end -->


::: bias_scope.generated_text_based.marked_persons.MarkedPersons

## Example

```python
# --------------------------------------------------------------
# Marked Persons
#
# Compares token distributions between "marked" (e.g., female-
# associated) and "unmarked" (e.g., generic/male-associated)
# generation corpora to identify terms that are disproportionately
# used when describing a marked group.
#
# Key outputs:
#   summary.vocab_considered — number of unique tokens analyzed
#   top_marked_terms         — tokens most over-represented in
#                              the marked corpus (sorted by z-score)
#
# No external API required.
# --------------------------------------------------------------

from pprint import pprint
from bias_scope.generated_text_based import MarkedPersons

# --- Initialize ---
metric = MarkedPersons()

# --- Marked corpus (e.g., female-associated completions) ---
marked_generations = [
    "She is nurturing and empathetic.",
    "She loves caring for others.",
]

# --- Unmarked corpus (e.g., generic completions) ---
unmarked_generations = [
    "The person is analytical and logical.",
    "They excel at problem solving.",
]

# --- Evaluate ---
result = metric.evaluate(
    marked_generations=marked_generations,
    unmarked_generations=unmarked_generations,
    min_count=1,
    return_top_k=5,
)

print("Summary:")
pprint(result["summary"])
print()
print("Top marked terms:")
pprint(result["top_marked_terms"][:3])
print()
print("Interpretation:")
print("  High z-score -> term appears much more in marked vs unmarked corpus")
print("  delta > 0    -> term is over-represented in marked text")
print("  Useful for identifying language patterns specific to a marked group")

```
