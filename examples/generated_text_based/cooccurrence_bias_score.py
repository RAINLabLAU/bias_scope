# --------------------------------------------------------------
# Co-Occurrence Bias Score
#
# Measures association strength between demographic groups and
# neutral vocabulary in generated text using pointwise mutual
# information (PMI).  Positive PMI means a word co-occurs with
# a group more than expected; negative means less.
#
# Key outputs:
#   summary.mean_abs_score — average absolute PMI across terms
#   summary.top_terms      — most skewed word-group associations
#
# No external API required.
# --------------------------------------------------------------

from pprint import pprint
from bias_scope.generated_text_based import CoOccurrenceBiasScore

# --- Initialize ---
metric = CoOccurrenceBiasScore()

# --- Toy generations with gendered occupations ---
generations = [
    "The man is a doctor and an engineer.",
    "The woman is a nurse and a teacher.",
    "The man works in technology.",
]

group_lexicons = {
    "male": ["man", "he", "his"],
    "female": ["woman", "she", "her"],
}

neutral_vocab = ["doctor", "engineer", "nurse", "teacher", "technology"]

# --- Evaluate ---
result = metric.evaluate(
    generations=generations,
    group_lexicons=group_lexicons,
    neutral_vocab=neutral_vocab,
    return_top_k=3,
)

print("Summary:")
pprint(result["summary"])
print()
print("Interpretation:")
print("  PMI > 0 for a (word, group) pair -> word co-occurs MORE with that group")
print("  PMI < 0 -> word co-occurs LESS with that group")
print("  mean_abs_score close to 0 -> balanced co-occurrence (unbiased)")
