# CEAT

::: bias_scope.embeddings_based.ceat.CEAT

CEAT requires precomputed, stimulus-aligned contextual **token** embeddings.
It does not accept raw strings, sentence-transformer vectors, CLS vectors, or
a flat bag of contextual occurrences.

Each input group is a mapping from a stimulus to all contextual occurrences of
that same stimulus:

    from bias_scope.embeddings_based import CEAT

    # Each matrix was extracted at the occurrence of its named word in natural text.
    X = {"John": john_token_vectors, "Paul": paul_token_vectors}
    Y = {"Amy": amy_token_vectors, "Joan": joan_token_vectors}
    A = {"career": career_token_vectors, "office": office_token_vectors}
    B = {"family": family_token_vectors, "home": home_token_vectors}

    result = CEAT().evaluate((X, Y), (A, B), n_samples=10_000, random_seed=42)
    print(result["effect_size"])  # CES
    print(result["p_value"])      # two-sided random-effects p-value

For every one of n_samples iterations, CEAT keeps all stimuli, independently
chooses one contextual occurrence per stimulus, computes a WEAT effect size,
and pools the sample effect sizes using a DerSimonian--Laird random-effects
model. A stimulus is sampled without replacement if it has at least n_samples
occurrences; otherwise it is sampled with replacement. X and Y must have equal
numbers of stimuli.

The result includes effect_size (the CES), standard_error,
between_context_variance, and a two-sided p_value. ceat_score remains an alias
for effect_size. Set return_details=True to receive per-sample effect sizes,
within-sample variances, random-effects weights, and selected indices.
