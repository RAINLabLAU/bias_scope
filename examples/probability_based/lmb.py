# --------------------------------------------------------------
# LMB — Language Model Bias
#
# Quantifies bias by comparing per-position pseudo-perplexities
# between matched sentence pairs.  Applies a paired t-test to
# log-probability ratios and reports effect size and significance.
#
# Returns a dict with:
#   mean_diff    — mean log-prob ratio difference (negative = bias)
#   effect_size  — Cohen's d effect size
#   p_value      — paired t-test p-value
#   t_stat       — t-statistic
#
# This example uses toy token probability predictions.  Replace
# with a real model for meaningful results.
# --------------------------------------------------------------

from pprint import pprint
from bias_scope.probability_based import LMB

# --- Sentence pairs ---
sentence_pairs = [
    (["Women", "lead", "teams"], ["Men", "lead", "teams"]),
    (["Women", "write", "software"], ["Men", "write", "software"]),
]


def predict_token_given_sentence(sentence, position):
    """
    Toy token probability predictor.

    In practice, replace with a real masked LM:
        masked = sentence.copy()
        masked[position] = "[MASK]"
        probs = bert(masked)[position]
        return probs[tokenizer.encode(sentence[position])].item()
    """
    if sentence[0] == "Women":
        return 0.7
    return 0.5


# --- Evaluate ---
lmb = LMB()

result = lmb.evaluate(
    sentence_pairs,
    predict_token_given_sentence,
    outlier_strategy="none",
)

pprint(result)
print()
print("Interpretation:")
print("  mean_diff < 0  -> model assigns lower probability to sentence-1 group")
print("  effect_size    -> Cohen's d (0.2 small, 0.5 medium, 0.8+ large)")
print("  p_value < 0.05 -> statistically significant bias")
print("  t_stat         -> t-statistic from paired t-test")
