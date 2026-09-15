# SEAT

::: bias_scope.embeddings_based.seat.SEAT

## Input and result contract

Canonical SEAT takes precomputed sentence embeddings for fully formed,
caller-supplied sentence stimuli. Callers are responsible for constructing
May et al.'s semantically bleached templates; BiasScope does not create
templates automatically.

Raw sentence strings are accepted as a BiasScope convenience and encoded
directly. The default CLS pooling uses Hugging Face special-token CLS
representations, so it is not an exact reproduction of May et al.'s original
per-encoder extraction setup. For experimental reproduction, provide precomputed
sentence embeddings extracted with the intended encoder protocol.

evaluate() returns the SEAT effect size by default. return_details=True also
returns the permutation p-value and metadata. run() returns a BiasResult
containing the effect size and p-value.

The effect size is WEAT's, unchanged (`ddof=1`). The permutation p-value follows
May et al.'s Appendix A rather than Caliskan's: the inequality is the non-strict
`Pr[s(Xi,Yi,A,B) >= s(X,Y,A,B)]` (`tie_policy="conservative"`, the default) and
the sampled estimate is floored at 1e-5 (`n_permutation_samples=100_000`, the
default). Pass `tie_policy="strict"` for Caliskan's `>` convention.

## Example

```python
# --------------------------------------------------------------
# SEAT - Sentence Encoder Association Test
#
# Adapts WEAT to sentence-level embeddings. The caller constructs
# sentence templates/stimuli and SEAT compares their representations.
#
# This example uses BiasScope's convenience text-encoding path.
# It does not construct templates automatically.
# --------------------------------------------------------------

from bias_scope.embeddings_based import SEAT

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

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

# --- Wrap in a bleached sentence template (the SEAT approach) ---
# Minimal illustration: a full SEAT run expands each term across several
# templates ("This is <word>.", "<word> is here.", ...), not just one.
male_sentences = [f"This is {name}." for name in male_names]
female_sentences = [f"This is {name}." for name in female_names]
career_sentences = [f"This is {word}." for word in career_words]
family_sentences = [f"This is {word}." for word in family_words]

# --- Evaluate ---
print(f"Embedding sentence inputs with {MODEL_NAME}...")
seat = SEAT(model_name=MODEL_NAME)

score = seat.evaluate(
    target_embeddings=(male_sentences, female_sentences),
    attribute_embeddings=(career_sentences, family_sentences),
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
