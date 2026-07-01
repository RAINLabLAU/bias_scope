# --------------------------------------------------------------
# WEAT - Word Embedding Association Test
#
# Measures how strongly one target group (e.g. male names)
# associates with one attribute group (e.g. career words)
# compared to another (e.g. family words) in the embedding space.
#
# Returns a single effect-size float. Positive -> target 1
# associates more with attribute 1. Larger magnitude = stronger bias.
#
# This example uses the built-in text embedding path, so you do not
# need to precompute embeddings manually.
# --------------------------------------------------------------

from bias_scope.embeddings_based import WEAT

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# --- Word lists from the original WEAT paper (Caliskan et al., 2017) ---

# Target groups: male vs. female names
male_names = ["John", "Paul", "Mike", "Kevin", "Steve", "Greg", "Jeff", "Bill"]
female_names = ["Amy", "Joan", "Lisa", "Sarah", "Diana", "Kate", "Ann", "Donna"]

# Attribute groups: career vs. family words
career_words = [
    "executive", "management", "professional", "corporation",
    "salary", "office", "business", "career",
]
family_words = [
    "home", "parents", "children", "family",
    "cousins", "marriage", "wedding", "relatives",
]

# --- Evaluate ---
print(f"Embedding text inputs with {MODEL_NAME}...")
weat = WEAT(model_name=MODEL_NAME)

score = weat.evaluate(
    target_embeddings=(male_names, female_names),
    attribute_embeddings=(career_words, family_words),
)

print(f"WEAT effect size: {score:.4f}")
print()
print("Interpretation:")
print("  > 0  -> male names associate more with career words")
print("  < 0  -> male names associate more with family words")
print("  ~0   -> no differential association (no bias)")
print(f"  Magnitude: {'strong' if abs(score) > 0.8 else 'moderate' if abs(score) > 0.5 else 'weak'} bias")
