"""CEAT requires precomputed contextual token embeddings per stimulus."""

import numpy as np

from bias_scope.embeddings_based import CEAT


# In a real analysis, each row must be the embedding of the named token at one
# natural-language occurrence. Do not substitute sentence or CLS embeddings.
rng = np.random.default_rng(42)


def contextual_token_embeddings(*stimuli, contexts=20, dimension=768):
    return {stimulus: rng.normal(size=(contexts, dimension)) for stimulus in stimuli}


male_names = contextual_token_embeddings("John", "Paul", "Mike")
female_names = contextual_token_embeddings("Amy", "Joan", "Lisa")
career_words = contextual_token_embeddings("career", "office")
family_words = contextual_token_embeddings("family", "home")

result = CEAT().evaluate(
    (male_names, female_names),
    (career_words, family_words),
    n_samples=10_000,
    random_seed=42,
)

print(f"CEAT combined effect size: {result['effect_size']:.4f}")
print(f"Two-sided p-value:         {result['p_value']:.4g}")
print(f"Standard error:             {result['standard_error']:.4f}")
