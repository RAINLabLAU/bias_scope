"""
--------------------------------------------------------------
Psycholinguistic Norms Example

Aggregates word-level psycholinguistic ratings over generated
completions using a norms lexicon.

For each completion, the metric averages the scores of covered
lexicon words on each selected dimension, then averages those
completion-level values across the dataset.

This example:
  1. Defines generated completions
  2. Provides a small psycholinguistic norms lexicon
  3. Computes aggregated scores for valence, arousal, and dominance

NOTE: This is a lexicon-based metric. It does not require a model.
--------------------------------------------------------------
"""

from bias_scope.generated_text_based.psycholinguistic_norms import (
    PsycholinguisticNorms,
)


# --- Example generated completions ---
completions = [
    [
        "The nurse was kind calm and caring.",
        "The criminal was hostile dangerous and cruel.",
    ],
    [
        "The leader seemed confident strong and smart.",
        "The child felt afraid and helpless.",
    ],
]


# --- Example psycholinguistic norms lexicon ---
norms_lexicon = {
    "kind": {"valence": 8.2, "arousal": 3.1, "dominance": 6.0},
    "calm": {"valence": 7.5, "arousal": 2.4, "dominance": 5.3},
    "caring": {"valence": 7.9, "arousal": 3.2, "dominance": 5.8},
    "criminal": {"valence": 2.1, "arousal": 5.8, "dominance": 3.2},
    "hostile": {"valence": 2.3, "arousal": 6.4, "dominance": 4.1},
    "dangerous": {"valence": 2.0, "arousal": 6.8, "dominance": 4.8},
    "cruel": {"valence": 1.9, "arousal": 5.9, "dominance": 4.4},
    "leader": {"valence": 6.8, "arousal": 4.7, "dominance": 7.4},
    "confident": {"valence": 7.4, "arousal": 4.5, "dominance": 7.1},
    "strong": {"valence": 6.9, "arousal": 5.1, "dominance": 7.6},
    "smart": {"valence": 7.3, "arousal": 4.2, "dominance": 6.5},
    "afraid": {"valence": 2.6, "arousal": 5.7, "dominance": 2.4},
    "helpless": {"valence": 2.4, "arousal": 4.9, "dominance": 1.9},
}


# --- Evaluate psycholinguistic norms ---
metric = PsycholinguisticNorms()

pn_scores = metric.evaluate(
    completions=completions,
    norms_lexicon=norms_lexicon,
)

print("Psycholinguistic Norms Example")
for key, value in pn_scores.items():
    print(f"{key}: {value:.4f}")
print()


# --- Evaluate with detailed statistics ---
detailed_result = metric.evaluate(
    completions=completions,
    norms_lexicon=norms_lexicon,
    return_details=True,
)

print("Detailed statistics:")
for key, value in detailed_result.items():
    print(f"{key}: {value:.4f}")

print()
print("Interpretation:")
print("  Higher pn::valence -> more positive affective wording on average")
print("  Higher pn::arousal -> more emotionally activating language on average")
print("  Higher pn::dominance -> more control/power-oriented language on average")
