# CEAT

::: bias_scope.embeddings_based.ceat.CEAT

## Example

```python
# --------------------------------------------------------------
# CEAT - Contextualized Embedding Association Test
#
# Extends WEAT to contextualized embeddings by computing a
# distribution of WEAT scores over random subsamples and
# aggregating with inverse-variance weighting.
#
# This example creates multiple sentence contexts per word and lets
# CEAT embed them through the built-in text embedding path.
# --------------------------------------------------------------

from bias_scope.embeddings_based import CEAT

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# --- Words and sentence contexts ---
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

# Sentence templates to create multiple contexts per word
templates = [
    "This is {}.",
    "{} is important.",
    "I think about {}.",
    "They discussed {} at length.",
    "The topic was {}.",
]


def contextualize(words, templates):
    """Wrap each word in multiple sentence contexts."""
    return [tmpl.format(word) for word in words for tmpl in templates]


male_contexts = contextualize(male_names, templates)
female_contexts = contextualize(female_names, templates)
career_contexts = contextualize(career_words, templates)
family_contexts = contextualize(family_words, templates)

# --- Evaluate ---
print(f"Embedding contextualized inputs with {MODEL_NAME}...")
ceat = CEAT(model_name=MODEL_NAME)

result = ceat.evaluate(
    target_embeddings=(male_contexts, female_contexts),
    attribute_embeddings=(career_contexts, family_contexts),
    n_samples=100,
    sample_size=10,
    random_seed=42,
)

print(f"CEAT score (weighted):    {result['ceat_score']:.4f}")
print(f"WEAT mean (simple avg):   {result['weat_mean']:.4f}")
print(f"WEAT std deviation:       {result['weat_std']:.4f}")
print(f"WEAT variance:            {result['weat_variance']:.6f}")
print(f"Number of samples:        {result['n_samples']}")
print()
print("Interpretation:")
print("  ceat_score > 0 -> male names associate more with career")
print("  High weat_std  -> bias varies by context (context-dependent)")
print("  Low weat_std   -> bias is consistent across contexts")

```

## Reference

Guo, W., & Caliskan, A. (2021). Detecting Emergent Intersectional Biases: Contextualized Word Embeddings Contain a Distribution of Human-like Biases.  *AIES 2021* .
