# SentenceBiasScore

::: bias_scope.embeddings_based.sentence_bias_score.SentenceBiasScore

## Example

```python
# --------------------------------------------------------------
# Sentence Bias Score
#
# Scores individual sentences for gender bias by weighting each
# word's cosine similarity to a gender direction vector by its
# semantic importance.  Returns (female_bias, male_bias).
#
# This example:
#   1. Builds a gender direction from gendered word pairs
#   2. Encodes a target sentence word-by-word
#   3. Uses uniform importance weights (no max-pooling layer available)
#
# NOTE: all-MiniLM-L6-v2 is a sentence encoder, so word-level
# embeddings are illustrative.  For production use static
# embeddings (GloVe / Word2Vec) which give true word vectors.
# --------------------------------------------------------------

import numpy as np
from sentence_transformers import SentenceTransformer
from bias_scope.embeddings_based import SentenceBiasScore

# --- Load encoder ---
model = SentenceTransformer("all-MiniLM-L6-v2")

# --- Build gender direction from gendered word pairs ---
# Each pair: (feminine, masculine).  PCA of their differences
# gives the gender direction.  Simplified here to mean-difference.
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

fem_emb = model.encode(feminine_words)
masc_emb = model.encode(masculine_words)

# Gender direction: mean(feminine) - mean(masculine)
gender_direction = fem_emb.mean(axis=0) - masc_emb.mean(axis=0)

# --- Sentence to analyse ---
sentence = "The nurse helps patients recover quickly"
words = sentence.split()

# Encode each word individually
word_embeddings = model.encode(words)

# Uniform importance weights (no encoder-specific attention available)
word_importance = np.ones(len(words)) / len(words)

# Mask explicitly gendered words (none in this sentence)
gender_words_mask = np.array([False] * len(words))

# --- Evaluate ---
sbs = SentenceBiasScore()

female_bias, male_bias = sbs.evaluate(
    word_embeddings=word_embeddings,
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
emb_2 = model.encode(words_2)
imp_2 = np.ones(len(words_2)) / len(words_2)
mask_2 = np.array([False] * len(words_2))

female_2, male_2 = sbs.evaluate(emb_2, gender_direction, imp_2, mask_2)

print(f"Sentence: \"{sentence_2}\"")
print(f"Female bias score: {female_2:.4f}")
print(f"Male bias score:   {male_2:.4f}")
print()
print("Interpretation:")
print("  female_bias > 0 -> sentence has feminine associations")
print("  male_bias < 0   -> sentence has masculine associations")
print("  Compare the two sentences to see occupational gender bias")

```

## Reference

Dolci, M., Azzalini, D., & Tanelli, M. (2023). Sentence-level bias detection in transformer models.
