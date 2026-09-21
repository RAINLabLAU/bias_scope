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
# Canonical WEAT uses precomputed static word embeddings.
# --------------------------------------------------------------

import numpy as np
from bias_scope.embeddings_based import WEAT

# Replace with selected rows from a static embedding table (for example,
# cased GloVe 840B/300d), preserving the paper's word-list order.
male_names = np.array([[1.0, 0.0], [0.9, 0.1]])
female_names = np.array([[0.0, 1.0], [0.1, 0.9]])
career_words = np.array([[1.0, 0.0], [0.95, 0.05]])
family_words = np.array([[0.0, 1.0], [0.05, 0.95]])

details = WEAT().evaluate(
    target_embeddings=(male_names, female_names),
    attribute_embeddings=(career_words, family_words),
    return_details=True,
)

print(f"WEAT effect size: {details['effect_size']:.4f}")
print(f"Strict permutation p-value: {details['p_value']:.4g}")
