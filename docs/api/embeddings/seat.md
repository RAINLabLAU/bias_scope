# SEAT

::: bias_scope.embeddings_based.seat.SEAT

## Example

```python
# --------------------------------------------------------------
# SEAT — Sentence Encoder Association Test
#
# Adapts WEAT to sentence-level embeddings.  Instead of encoding
# bare words, SEAT wraps them in a sentence template
# (e.g. "This is <word>") and encodes the full sentence.
# Internally delegates to WEAT.
#
# NOTE: all-MiniLM-L6-v2 is a sentence encoder, so encoding full
# sentences is its intended use.  For static word vectors use
# GloVe or Word2Vec instead.
# --------------------------------------------------------------

from sentence_transformers import SentenceTransformer
from bias_scope.embeddings_based import SEAT

# --- Load encoder (downloads ~80 MB on first run) ---
model = SentenceTransformer("all-MiniLM-L6-v2")

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

# --- Wrap in sentence templates (the SEAT approach) ---
male_sentences = [f"This is {name}." for name in male_names]
female_sentences = [f"This is {name}." for name in female_names]
career_sentences = [f"This is about {word}." for word in career_words]
family_sentences = [f"This is about {word}." for word in family_words]

# --- Encode full sentences ---
male_emb = model.encode(male_sentences)
female_emb = model.encode(female_sentences)
career_emb = model.encode(career_sentences)
family_emb = model.encode(family_sentences)

# --- Evaluate ---
seat = SEAT()

score = seat.evaluate(
    target_embeddings=(male_emb, female_emb),
    attribute_embeddings=(career_emb, family_emb),
)

print(f"SEAT effect size: {score:.4f}")
print()
print("Interpretation (same scale as WEAT):")
print("  > 0  -> male-associated sentences closer to career sentences")
print("  < 0  -> male-associated sentences closer to family sentences")
print("  ~0   -> no differential association")
print(f"  Magnitude: {'strong' if abs(score) > 0.8 else 'moderate' if abs(score) > 0.5 else 'weak'} bias")

```

## Reference

May, C., Wang, A., Bordia, S., Bowman, S. R., & Rudinger, R. (2019). On measuring social biases in sentence encoders.  *NAACL-HLT 2019* .
