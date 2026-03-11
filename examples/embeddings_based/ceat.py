# --------------------------------------------------------------
# CEAT — Contextualized Embedding Association Test
#
# Extends WEAT to contextualized embeddings by computing a
# distribution of WEAT scores over random subsamples and
# aggregating with inverse-variance weighting.
#
# This example embeds each word in multiple sentence contexts
# to simulate contextualized representations, then runs CEAT.
#
# NOTE: all-MiniLM-L6-v2 is a sentence encoder, so these results
# are illustrative.  For production use a model like BERT that
# produces different representations per context.
# --------------------------------------------------------------

from sentence_transformers import SentenceTransformer
from bias_scope.embeddings_based import CEAT

# --- Load encoder ---
model = SentenceTransformer("all-MiniLM-L6-v2")

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

# --- Encode each word in every context ---
def encode_in_contexts(words, templates):
    """Encode each word in multiple sentence contexts."""
    sentences = [tmpl.format(w) for w in words for tmpl in templates]
    return model.encode(sentences)

male_emb = encode_in_contexts(male_names, templates)      # 8 × 5 = 40 embeddings
female_emb = encode_in_contexts(female_names, templates)   # 40
career_emb = encode_in_contexts(career_words, templates)   # 40
family_emb = encode_in_contexts(family_words, templates)   # 40

# --- Evaluate ---
ceat = CEAT()

result = ceat.evaluate(
    target_embeddings=(male_emb, female_emb),
    attribute_embeddings=(career_emb, family_emb),
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
