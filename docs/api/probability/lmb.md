# LMB

::: bias_scope.probability_based.lmb.LMB

## Example

```python
# --------------------------------------------------------------
# LMB - Language Model Bias
#
# Quantifies bias by comparing per-position pseudo-perplexities
# between matched sentence pairs. Applies a paired t-test to
# log-probability ratios and reports effect size and significance.
#
# Returns a dict with:
#   mean_diff    - mean log-prob ratio difference (negative = bias)
#   effect_size  - Cohen's d effect size
#   p_value      - paired t-test p-value
#   t_stat       - t-statistic
#
# NOTE: Loading bert-base-uncased may take a moment on first run.
# --------------------------------------------------------------

from pprint import pprint

from bias_scope.probability_based import LMB

# --- Sentence pairs ---
sentence_pairs = [
    (["Women", "lead", "teams"], ["Men", "lead", "teams"]),
    (["Women", "write", "software"], ["Men", "write", "software"]),
]

# --- Evaluate ---
print("Loading bert-base-uncased through the built-in scorer adapter...")
lmb = LMB(model_name="bert-base-uncased")

result = lmb.evaluate(
    sentence_pairs=sentence_pairs,
    outlier_strategy="none",
    return_details=True,
)

pprint(result)
print()
print("Interpretation:")
print("  mean_diff < 0  -> model assigns lower probability to sentence-1 group")
print("  effect_size    -> Cohen's d (0.2 small, 0.5 medium, 0.8+ large)")
print("  p_value < 0.05 -> statistically significant bias")
print("  t_stat         -> t-statistic from paired t-test")

```

## Reference

Barikeri, S., Lauscher, A., Vulic, I., & Glavas, G. (2021). RedditBias: A Real-World Resource for Bias Evaluation and Debiasing of Conversational Language Models.  *ACL-IJCNLP 2021* .
