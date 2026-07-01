# --------------------------------------------------------------
# Sentence Bias Score
#
# Scores individual sentences for gender bias by weighting each
# word's cosine similarity to a gender direction vector by its
# semantic importance. Returns (female_bias, male_bias).
#
# This example:
#   1. Builds a gender direction from gendered word pairs
#   2. Uses the built-in embed() helper for the direction vectors
#   3. Lets SentenceBiasScore embed tokenized words automatically
# --------------------------------------------------------------

import numpy as np

from bias_scope.embeddings_based import SentenceBiasScore, embed

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# --- Build gender direction from gendered word pairs ---
# Each pair: (feminine, masculine). PCA of their differences
# gives the gender direction. Simplified here to mean-difference.
gendered_pairs = [
    ("she", "he"),
    ("woman", "man"),
    ("her", "his"),
    ("mother", "father"),
    ("daughter", "son"),
    ("girl", "boy"),
    ("female", "male"),
    ("queen", "king"),
]

feminine_words = [p[0] for p in gendered_pairs]
masculine_words = [p[1] for p in gendered_pairs]

fem_emb = embed(feminine_words, model_name=MODEL_NAME)
masc_emb = embed(masculine_words, model_name=MODEL_NAME)

# Gender direction: mean(feminine) - mean(masculine)
gender_direction = fem_emb.mean(axis=0) - masc_emb.mean(axis=0)

# --- Sentence to analyze ---
sentence = "The nurse helps patients recover quickly"
words = sentence.split()

# Uniform importance weights (no encoder-specific attention available)
word_importance = np.ones(len(words)) / len(words)

# Mask explicitly gendered words (none in this sentence)
gender_words_mask = np.array([False] * len(words))

# --- Evaluate ---
print(f"Embedding token inputs with {MODEL_NAME}...")
sbs = SentenceBiasScore(model_name=MODEL_NAME)

female_bias, male_bias = sbs.evaluate(
    word_embeddings=words,
    gender_direction=gender_direction,
    word_importance=word_importance,
    gender_words_mask=gender_words_mask,
)

print(f"Sentence: \"{sentence}\"")
print(f"Female bias score: {female_bias:.4f}")
print(f"Male bias score:   {male_bias:.4f}")
print()

# --- Compare with a male-stereotyped sentence ---
sentence_2 = "The engineer designs complex software systems"
words_2 = sentence_2.split()
imp_2 = np.ones(len(words_2)) / len(words_2)
mask_2 = np.array([False] * len(words_2))

female_2, male_2 = sbs.evaluate(words_2, gender_direction, imp_2, mask_2)

print(f"Sentence: \"{sentence_2}\"")
print(f"Female bias score: {female_2:.4f}")
print(f"Male bias score:   {male_2:.4f}")
print()
print("Interpretation:")
print("  female_bias > 0 -> sentence has feminine associations")
print("  male_bias < 0   -> sentence has masculine associations")
print("  Compare the two sentences to see occupational gender bias")
