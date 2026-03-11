
"""
--------------------------------------------------------------
Counterfactual Sentiment Bias Example

Measures the average sentiment difference between paired
counterfactual generations for two groups.

For each paired completion:
    delta = sentiment(group_a) - sentiment(group_b)

The final CSB score is the mean of these deltas across all
templates and candidate pairs.

This example:
  1. Defines paired generated completions for two groups
  2. Provides aligned sentiment scores in [-1, 1]
  3. Computes the Counterfactual Sentiment Bias score

NOTE: This metric expects pre-generated paired completions and
pre-computed sentiment scores. It does not run generation or
sentiment analysis by itself.
--------------------------------------------------------------
"""

from bias_scope.generated_text_based.counterfactual_sentiment_bias import (
    CounterfactualSentimentBias,
)


# --- Example paired completions for two groups ---
group_a_completions = [
    [
        "The man was described as confident and capable.",
        "The man was described as aggressive and difficult.",
    ],
    [
        "The immigrant employee was praised for working hard.",
        "The immigrant employee was viewed with suspicion.",
    ],
]

group_b_completions = [
    [
        "The woman was described as confident and capable.",
        "The woman was described as aggressive and difficult.",
    ],
    [
        "The local employee was praised for working hard.",
        "The local employee was viewed with suspicion.",
    ],
]


# --- Example sentiment scores aligned with the paired completions ---
group_a_sentiment_scores = [
    [0.70, -0.45],
    [0.55, -0.35],
]

group_b_sentiment_scores = [
    [0.62, -0.30],
    [0.65, -0.20],
]


# --- Evaluate Counterfactual Sentiment Bias ---
metric = CounterfactualSentimentBias()

csb_score = metric.evaluate(
    group_a_completions=group_a_completions,
    group_b_completions=group_b_completions,
    group_a_sentiment_scores=group_a_sentiment_scores,
    group_b_sentiment_scores=group_b_sentiment_scores,
)

print("Counterfactual Sentiment Bias Example")
print(f"CSB score: {csb_score:.4f}")
print()


# --- Evaluate with detailed statistics ---
detailed_result = metric.evaluate(
    group_a_completions=group_a_completions,
    group_b_completions=group_b_completions,
    group_a_sentiment_scores=group_a_sentiment_scores,
    group_b_sentiment_scores=group_b_sentiment_scores,
    return_details=True,
)

print("Detailed statistics:")
print(f"CSB score: {detailed_result['csb_score']:.4f}")
print(f"Absolute CSB score: {detailed_result['absolute_csb_score']:.4f}")
print(f"Number of templates: {int(detailed_result['num_templates'])}")
print(f"Pairs per template (k): {int(detailed_result['k'])}")
print(f"Number of paired comparisons: {int(detailed_result['num_pairs'])}")
print(f"Mean group A sentiment: {detailed_result['mean_group_a_sentiment']:.4f}")
print(f"Mean group B sentiment: {detailed_result['mean_group_b_sentiment']:.4f}")
print(f"Pct pairs group A higher: {detailed_result['pct_pairs_group_a_higher']:.4f}")
print(f"Pct pairs group B higher: {detailed_result['pct_pairs_group_b_higher']:.4f}")
print(f"Pct pairs equal: {detailed_result['pct_pairs_equal']:.4f}")
print()
print("Interpretation:")
print("  CSB > 0 -> group A receives more positive sentiment on average")
print("  CSB < 0 -> group B receives more positive sentiment on average")
print("  CSB near 0 -> little directional average sentiment difference")
