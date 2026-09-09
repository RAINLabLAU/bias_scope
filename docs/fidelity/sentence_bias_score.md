# SentenceBiasScore

**Cited source:** Dolci, Azzalini & Tanelli 2023, *Sentence-level bias score*,
Data Science and Engineering 8(2), Springer.
**Paper status: `read`.** The Springer article is open access.
**Reference implementation:** none located.
**Family / access:** embedding / `embeddings`

## Status: **faithful**

BiasScope implements the low-level SentenceBiasScore equations from Dolci et al.:
each token/word embedding is projected onto the gender direction with cosine
similarity, multiplied by that token's semantic-importance weight, and aggregated
into separate positive/female and negative/male sums. Explicit gender terms are
excluded by a required boolean mask and therefore contribute zero.

The canonical API deliberately requires caller-provided upstream artifacts:

- token/word embedding matrix with shape `(n_tokens, embedding_dim)`
- gender direction vector with shape `(embedding_dim,)`
- semantic-importance vector with shape `(n_tokens,)`
- explicit gender-word exclusion mask with shape `(n_tokens,)`

BiasScope does not compute the PCA gender direction, derive importance weights
from a transformer encoder, or produce contextual token/word representations.
Raw string inputs are rejected because independently embedding words does not
reproduce Dolci et al.'s sentence-context token/word representations.

## Equation

For each non-excluded token `i`:

```text
word_bias_i = cosine(word_embedding_i, gender_direction)
weighted_i = word_bias_i * importance_i

female_bias = sum(weighted_i > 0)
male_bias   = sum(weighted_i < 0)
absolute_bias = abs(female_bias) + abs(male_bias)
```

The implementation preserves Dolci et al.'s sign convention: positive scores are
female-associated and negative scores are male-associated.
