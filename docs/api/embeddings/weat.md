# WEAT

::: bias_scope.embeddings_based.weat.WEAT

## Example

```python
# --------------------------------------------------------------
# WEAT — Word Embedding Association Test
#
# Measures how strongly one target group (e.g. male names)
# associates with one attribute group (e.g. career words)
# compared to another (e.g. family words) in the embedding space.
#
# Returns a single effect-size float.  Positive -> target 1
# associates more with attribute 1.  Larger magnitude = stronger bias.
#
# NOTE: all-MiniLM-L6-v2 is a sentence encoder, so these results
# are illustrative.  For production WEAT evaluations use static
# embeddings such as GloVe or Word2Vec.
# --------------------------------------------------------------

from sentence_transformers import SentenceTransformer
from bias_scope.embeddings_based import WEAT

# --- Load encoder (downloads ~80 MB on first run) ---
model = SentenceTransformer("all-MiniLM-L6-v2")

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

# --- Encode words into embeddings ---
male_emb = model.encode(male_names)
female_emb = model.encode(female_names)
career_emb = model.encode(career_words)
family_emb = model.encode(family_words)

# --- Evaluate ---
weat = WEAT()

score = weat.evaluate(
    target_embeddings=(male_emb, female_emb),
    attribute_embeddings=(career_emb, family_emb),
)

print(f"WEAT effect size: {score:.4f}")
print()
print("Interpretation:")
print("  > 0  -> male names associate more with career words")
print("  < 0  -> male names associate more with family words")
print("  ~0   -> no differential association (no bias)")
print(f"  Magnitude: {'strong' if abs(score) > 0.8 else 'moderate' if abs(score) > 0.5 else 'weak'} bias")

```

## Reference

Caliskan, A., Bryson, J. J., & Narayanan, A. (2017). Semantics derived automatically from language corpora contain human-like biases.  *Science* , 356(6334), 183–186.
