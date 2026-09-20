# --------------------------------------------------------------
# SEAT - Sentence Encoder Association Test
#
# Adapts WEAT to sentence-level embeddings. Instead of encoding
# bare words, SEAT wraps them in a semantically bleached sentence
# template and compares sentence-level associations.
#
# This example uses the built-in text embedding path, so the metric
# handles sentence encoding for you. It is a minimal illustration:
# a full SEAT run (May et al. 2019) expands each term across several
# templates -- "This is <word>.", "<word> is here.", "This will
# <word>.", "<word> are things." -- not just one.
# --------------------------------------------------------------

from bias_scope.embeddings_based import SEAT

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# --- Word lists from the WEAT paper ---
male_names = ["John", "Paul", "Mike", "Kevin", "Steve", "Greg", "Jeff", "Bill"]
female_names = ["Amy", "Joan", "Lisa", "Sarah", "Diana", "Kate", "Ann", "Donna"]

career_words = [
    "executive", "management", "professional", "corporation",
    "salary", "office", "business", "career",
]
family_words = [
    "home", "parents", "children", "family",
    "cousins", "marriage", "wedding", "relatives",
]

# --- Wrap in a bleached sentence template (the SEAT approach) ---
male_sentences = [f"This is {name}." for name in male_names]
female_sentences = [f"This is {name}." for name in female_names]
career_sentences = [f"This is {word}." for word in career_words]
family_sentences = [f"This is {word}." for word in family_words]

# --- Evaluate ---
print(f"Embedding sentence inputs with {MODEL_NAME}...")
seat = SEAT(model_name=MODEL_NAME)

score = seat.evaluate(
    target_embeddings=(male_sentences, female_sentences),
    attribute_embeddings=(career_sentences, family_sentences),
)

print(f"SEAT effect size: {score:.4f}")
print()
print("Interpretation (same scale as WEAT):")
print("  > 0  -> male-associated sentences closer to career sentences")
print("  < 0  -> male-associated sentences closer to family sentences")
print("  ~0   -> no differential association")
print(f"  Magnitude: {'strong' if abs(score) > 0.8 else 'moderate' if abs(score) > 0.5 else 'weak'} bias")
