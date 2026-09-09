# --------------------------------------------------------------
# Sentence Bias Score
#
# Scores individual sentences for gender bias by weighting each
# word's cosine similarity to a gender direction vector by its
# semantic importance. Returns (female_bias, male_bias).
#
# This example:
#   1. Uses caller-provided token/word representations from a sentence
#   2. Uses a caller-provided gender direction
#   3. Excludes explicit gender terms with a required mask
# --------------------------------------------------------------

import numpy as np

from bias_scope.embeddings_based import SentenceBiasScore

# In a real run, obtain these from the same model/sentence pass Dolci et al.
# require: token/word representations, semantic importance, and gender direction.
sentence = "She likes beautiful dresses"
token_embeddings = np.array(
    [
        [1.0, 0.0, 0.0],   # She, explicitly gendered and excluded below
        [0.0, 1.0, 0.0],   # likes
        [0.6, 0.8, 0.0],   # beautiful
        [0.4, 0.0, 0.9],   # dresses
    ]
)

# Positive scores are feminine, negative scores are masculine.
gender_direction = np.array([1.0, 0.0, 0.0])
word_importance = np.array([0.25, 0.25, 0.30, 0.20])
gender_words_mask = np.array([True, False, False, False])

# --- Evaluate ---
sbs = SentenceBiasScore()

result = sbs.evaluate(
    word_embeddings=token_embeddings,
    gender_direction=gender_direction,
    word_importance=word_importance,
    gender_words_mask=gender_words_mask,
    return_details=True,
)

print(f"Sentence: \"{sentence}\"")
print(f"Female bias score: {result['female_bias']:.4f}")
print(f"Male bias score:   {result['male_bias']:.4f}")
print(f"Absolute bias:     {result['absolute_bias']:.4f}")
print()
