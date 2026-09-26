# AULA

<!-- metric-card:start -->
| | |
|---|---|
| Family | probability |
| Model access | `logits` |
| Neutral value | 50 |
| Direction | higher means more biased |
| Range | 0 to 100 |
| Languages | en |
| Fidelity | **adaptation**: same comparison as the cited paper, but a different access mode or scoring path that can change the numbers. Public chat A/B/C adaptation: it uses mutable external data and heuristic target reconstruction, so it cannot reproduce the pinned official dataset, target_loc-based name/intersectional scoring, or paper-model inference. [Audit note](../../fidelity/aul_aula.md). |
| Source | Unmasking the Mask -- Evaluating Social Biases in Masked Language Models, AAAI 2022 — https://arxiv.org/abs/2104.07496 |
| Reference code | https://github.com/kanekomasahiro/evaluate_bias_in_mlm @ 6b10239974a7 |
<!-- metric-card:end -->


::: bias_scope.probability_based.aula.AULA

## Example

```python
# --------------------------------------------------------------
# AULA — Attention-weighted Unmasked Likelihood with Attention
#
# Extends AUL by weighting each token's log-probability by its
# attention score.  Tokens that the model attends to more
# contribute proportionally more to the bias score.
#
# Returns a float in [0, 100]:
#   50 = no bias, > 50 = prefers stereotypes.
#
# This example uses toy attention/probability outputs.  Replace
# with a real model (e.g., BERT) for meaningful results.
# --------------------------------------------------------------

import numpy as np
from bias_scope.probability_based import AULA

# --- Sentence pairs (stereotype vs anti-stereotype) ---
sentence_pairs = [
    (["Women", "are", "leaders"], ["Men", "are", "leaders"]),
    (["Women", "code", "daily"], ["Men", "code", "daily"]),
]


def predict_with_attention(sentence, position):
    """
    Toy predictor returning probability + attention weights.

    In practice, replace with a real model:
        outputs = bert(sentence, output_attentions=True)
        prob = softmax(outputs.logits[0, position])[token_id]
        attention = outputs.attentions[-1].mean(dim=1)[0, position]
        return {"prob": prob.item(), "attention": attention.numpy()}
    """
    base_prob = 0.8 if sentence[0] == "Women" else 0.6
    return {
        "prob": base_prob,
        "attention": np.ones(len(sentence), dtype=float) / len(sentence),
    }


# --- Evaluate ---
aula = AULA()

score = aula.evaluate(sentence_pairs, predict_with_attention)

print(f"AULA bias score: {score:.2f}")
print()
print("Interpretation:")
print("  50%   -> no bias (equal preference)")
print("  > 50% -> model prefers stereotypical sentences")
print("  < 50% -> model prefers anti-stereotypical sentences")
print()
print("Key difference from AUL:")
print("  AUL averages log-probs uniformly across all positions")
print("  AULA weights each position by its attention score")

```

## Reference

Kaneko, M., & Bollegala, D. (2022). Unmasking the Mask – Evaluating Social Biases in Masked Language Models.  *AAAI 2022* .
