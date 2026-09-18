# --------------------------------------------------------------
# Sentence Bias Score
#
# Scores individual sentences for gender bias by weighting each
# word's cosine similarity to a gender direction vector by its
# semantic importance. Returns (female_bias, male_bias).
#
# This example:
#   1. Derives the gender direction from gender word-pair embeddings
#      with derive_gender_direction() (Dolci et al. Sec. 3.2).
#   2. Derives word importance from encoder hidden states with
#      derive_word_importance() (Sec. 3.4).
#   3. Builds the exclusion mask with build_gender_words_mask(), given
#      a caller-supplied lexicon -- BiasScope does not ship Dolci et
#      al.'s own 6562-word lexicon (Sec. 3.3); see
#      docs/fidelity/sentence_bias_score.md.
#   4. Calls both evaluate() and run().
# --------------------------------------------------------------

import numpy as np

from bias_scope.embeddings_based import SentenceBiasScore
from bias_scope.embeddings_based.sentence_bias_score import (
    build_gender_words_mask,
    derive_gender_direction,
    derive_word_importance,
)

# --- 1. Gender direction: PCA of gender word-pair embeddings ---
# In a real run these come from the same word embedding model as the
# sentence encoder (Dolci et al. use GloVe for InferSent). Toy 3D
# embeddings here for a runnable, self-contained example.
female_words = np.array([[1.0, 0.1, 0.0], [0.9, 0.0, 0.1], [1.1, -0.1, 0.0]])
male_words = np.array([[-1.0, 0.1, 0.0], [-0.9, 0.0, 0.1], [-1.1, -0.1, 0.0]])
gender_direction = derive_gender_direction(female_words, male_words)

# --- 2. Word importance: max-pooling selection counts ---
# hidden_states[t] is the encoder's per-token hidden state at time step t,
# *before* max-pooling collapses the sentence to one vector.
sentence = "She likes beautiful dresses"
hidden_states = np.array(
    [
        [1.0, 0.0, 0.0, 0.2],  # She
        [0.0, 1.0, 0.0, 0.3],  # likes
        [0.6, 0.8, 0.0, 0.1],  # beautiful
        [0.4, 0.0, 0.9, 0.9],  # dresses
    ]
)
word_importance = derive_word_importance(hidden_states)
token_embeddings = hidden_states[:, :3]  # word-level embeddings for scoring

# --- 3. Exclusion mask: your own gender-word lexicon ---
tokens = ["she", "likes", "beautiful", "dresses"]
my_gender_word_list = ["she", "he", "her", "him", "woman", "man"]
gender_words_mask = build_gender_words_mask(tokens, my_gender_word_list)

# --- 4. Evaluate ---
sbs = SentenceBiasScore()

details = sbs.evaluate(
    word_embeddings=token_embeddings,
    gender_direction=gender_direction,
    word_importance=word_importance,
    gender_words_mask=gender_words_mask,
    return_details=True,
)

print(f"Sentence: \"{sentence}\"")
print(f"Word importance:    {word_importance.round(4).tolist()}")
print(f"Female bias score:  {details['female_bias']:.4f}")
print(f"Male bias score:    {details['male_bias']:.4f}")
print(f"Absolute bias:      {details['absolute_bias']:.4f}")
print()

# run() reports Abs-BiasScore (Eq. 3) as the headline score, with the
# female/male breakdown attached.
result = sbs.run(
    token_embeddings, gender_direction, word_importance, gender_words_mask
)
print(f"BiasResult.score (Abs-BiasScore): {result.score:.4f}")
print(f"BiasResult.breakdown:             {result.breakdown}")
