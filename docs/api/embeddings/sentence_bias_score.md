# SentenceBiasScore

<!-- metric-card:start -->
| | |
|---|---|
| Family | embedding |
| Model access | `embeddings` |
| Neutral value | 0 |
| Direction | higher means more biased |
| Range | 0 or more |
| Languages | en |
| Fidelity | **adaptation**: same comparison as the cited paper, but a different access mode or scoring path that can change the numbers. The scoring equations (Eq. 1-3) are faithful and verified against the paper's own worked example (Table 2) to float precision; run()/BiasResult.score reports Eq. 3 (Abs-BiasScore). derive_gender_direction() and derive_word_importance() implement the paper's PCA and max-pooling procedures (Sec. 3.2, 3.4). [Audit note](../../fidelity/sentence_bias_score.md). |
| Source | Dolci, Azzalini & Tanelli 2023, Data Science and Engineering 8, 177-195, Springer — https://doi.org/10.1007/s41019-023-00211-0 |
<!-- metric-card:end -->


::: bias_scope.embeddings_based.sentence_bias_score.SentenceBiasScore

## Example

```python
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

```

## Deriving the upstream artifacts

`evaluate()`/`run()` need `gender_direction`, `word_importance`, and
`gender_words_mask` as inputs. Three module-level helpers implement the
paper's own procedures for the first two and for matching a lexicon, given
that you supply the raw ingredients (embeddings, hidden states, a lexicon):

- `derive_gender_direction(female_embeddings, male_embeddings)` — PCA of
  gender word-pair difference vectors (Sec. 3.2). `GENDER_WORD_PAIRS` lists
  the paper's ten canonical pairs.
- `derive_word_importance(hidden_states)` — max-pooling selection counts
  (Sec. 3.4), given the encoder's per-token hidden states before pooling.
- `build_gender_words_mask(tokens, gender_word_list)` — case-insensitive
  lexicon matching (Sec. 3.3). **BiasScope does not ship Dolci et al.'s
  6562-word lexicon** — supply your own; see `docs/fidelity/sentence_bias_score.md`.

`run()` reports Eq. 3 (`Abs-BiasScore`, `abs(female_bias) + abs(male_bias)`)
as `BiasResult.score`, and `female_bias`/`male_bias` in `BiasResult.breakdown`.

## Reference

Dolci, T., Azzalini, F., & Tanelli, M. (2023). Improving Gender-Related Fairness in Sentence Encoders: A Semantics-Based Approach. Data Science and Engineering, 8, 177-195. https://doi.org/10.1007/s41019-023-00211-0
