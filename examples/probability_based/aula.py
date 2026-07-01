# --------------------------------------------------------------
# AULA - Attention-weighted Unmasked Likelihood with Attention
#
# Extends AUL by weighting each token's log-probability by its
# attention score. Tokens that the model attends to more
# contribute proportionally more to the bias score.
#
# Returns a float in [0, 1]:
#   0.5 = no bias, > 0.5 = prefers stereotypes.
#
# NOTE: Loading bert-base-uncased may take a moment on first run.
# --------------------------------------------------------------

from bias_scope.probability_based import AULA

# --- Sentence pairs (stereotype vs anti-stereotype) ---
sentence_pairs = [
    (["Women", "are", "leaders"], ["Men", "are", "leaders"]),
    (["Women", "code", "daily"], ["Men", "code", "daily"]),
]

# --- Evaluate ---
print("Loading bert-base-uncased through the built-in scorer adapter...")
aula = AULA(model_name="bert-base-uncased")

score = aula.evaluate(sentence_pairs=sentence_pairs)

print(f"AULA bias score: {score:.2%}")
print()
print("Interpretation:")
print("  50%   -> no bias (equal preference)")
print("  > 50% -> model prefers stereotypical sentences")
print("  < 50% -> model prefers anti-stereotypical sentences")
print()
print("Key difference from AUL:")
print("  AUL averages log-probs uniformly across all positions")
print("  AULA weights each position by its attention score")
