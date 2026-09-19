# WEAT

::: bias_scope.embeddings_based.weat.WEAT

## Input and result contract

Canonical WEAT takes precomputed rank-2 word-embedding matrices. X and Y must
contain equal numbers of target stimuli; all four sets must be non-empty,
finite, same-dimensional, and contain no zero-norm vectors.

BiasScope additionally accepts raw string sequences and embeds them with the
configured model. This is a noncanonical BiasScope extension, not the original
static-precomputed-embedding WEAT protocol; document the encoder, tokenizer,
subword/truncation behavior, model, and pooling choice when reporting it.

evaluate() returns the WEAT effect size by default. Pass return_details=True to
also receive the one-sided permutation p-value, whether it was exact, the
number of partitions, and the permutation note. run() returns a BiasResult with
the effect size as score and the same p-value in BiasResult.p_value.

The default p-value uses the paper's strict `>` comparison. Set
`tie_policy="conservative"` only to reproduce the later May et al./sent-bias
`>=` convention. For sampled tests, `n_permutation_samples` is the total number
of partition evaluations: strict mode draws that many random partitions, while
conservative mode counts the observed partition once and draws one fewer.

## Example

```python
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
# Canonical path: precomputed static word embeddings.
# --------------------------------------------------------------

import numpy as np
from bias_scope.embeddings_based import WEAT

# Replace these toy vectors with rows selected from a static embedding table,
# such as the cased GloVe 840B/300d vectors used by Caliskan et al.
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

```

## Reference

Caliskan, A., Bryson, J. J., & Narayanan, A. (2017). Semantics derived automatically from language corpora contain human-like biases.  *Science* , 356(6334), 183-186.
