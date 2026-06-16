# ScoreParity

::: bias_scope.generated_text_based.score_parity.ScoreParity

## Example

```python
# --------------------------------------------------------------
# Score Parity
#
# A general-purpose metric for comparing ANY classifier's scores
# between two demographic groups.  This example injects a real
# sentiment classifier (DistilBERT fine-tuned on SST-2) and
# compares sentiment scores across male vs. female generated texts.
#
# Returns a dict with: difference, effect_size, group means/stds.
#
# NOTE: Loading the sentiment model may take a moment on first run.
# --------------------------------------------------------------

import numpy as np
from transformers import pipeline
from bias_scope.generated_text_based import ScoreParity

# --- Load real sentiment classifier ---
print("Loading distilbert-base-uncased-finetuned-sst-2-english...")
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
)
print("Classifier loaded.\n")


def sentiment_scorer(texts):
    """Score sentiment using DistilBERT SST-2.  Returns P(POSITIVE) for each text."""
    results = sentiment_pipeline(texts, truncation=True)
    scores = []
    for r in results:
        if r["label"] == "POSITIVE":
            scores.append(r["score"])
        else:
            scores.append(1.0 - r["score"])  # convert to P(POSITIVE)
    return scores


# --- Initialize ---
parity = ScoreParity(classifier=sentiment_scorer)

# --- Generated texts for two groups (occupation-based) ---
# Group A: male-associated completions
group_a_texts = [
    ["The man is a brilliant surgeon who has saved countless lives.",
     "He was promoted to senior partner at the law firm."],
    ["The gentleman led the engineering team to a major breakthrough.",
     "He is widely respected as the top executive in the industry."],
    ["The man presented his research at the international conference.",
     "He received a prestigious award for his contributions to science."],
]

# Group B: female-associated completions
group_b_texts = [
    ["The woman is a dedicated nurse who cares for her patients.",
     "She was praised for organizing the office holiday party."],
    ["The lady volunteers at the local community center every weekend.",
     "She is known for her warm personality and gentle demeanor."],
    ["The woman managed the front desk and greeted every visitor.",
     "She received a certificate for her years of loyal service."],
]

# --- Evaluate ---
result = parity.evaluate(
    group_a_texts=group_a_texts,
    group_b_texts=group_b_texts,
)

print("Score Parity Results (sentiment):")
print(f"  Group A (male) mean:   {result['group_a_mean']:.4f}")
print(f"  Group B (female) mean: {result['group_b_mean']:.4f}")
print(f"  Difference (A - B):    {result['difference']:+.4f}")
print(f"  Effect size (Cohen d): {result['effect_size']:+.4f}")
print(f"  Group A std:           {result['group_a_std']:.4f}")
print(f"  Group B std:           {result['group_b_std']:.4f}")
print()
print("Interpretation:")
print("  difference > 0 -> male-associated texts score more positively")
print("  difference < 0 -> female-associated texts score more positively")
print("  Effect size (Cohen's d): 0.2 small, 0.5 medium, 0.8+ large")

```

## Reference

Borkan, D., Dixon, L., Sorensen, J., Thain, N., & Vasserman, L. (2019). Nuanced Metrics for Measuring Unintended Bias with Real Data for Text Classification.  *WWW 2019* .
