# LPBS

<!-- metric-card:start -->
| | |
|---|---|
| Family | probability |
| Model access | `logits` |
| Neutral value | 0 |
| Direction | signed; 0 is neutral |
| Range | unbounded |
| Languages | en |
| Fidelity | **faithful**: same formula and protocol as the cited paper. Faithful to the paper. [Audit note](../../fidelity/lpbs.md). |
| Source | Measuring Bias in Contextualized Word Representations, GeBNLP 2019 — https://arxiv.org/abs/1906.07337 |
| Reference code | https://github.com/keitakurita/contextual_embedding_bias_measure @ 18044f87e2ff |
<!-- metric-card:end -->


::: bias_scope.probability_based.lpbs.LPBS

## Example

```python
"""
LPBS example using Kurita et al.'s masked-token probability contract.

LPBS compares the increased log probability of two target tokens for an
attribute:

    [log p_tgt(target_a) - log p_prior(target_a)]
    - [log p_tgt(target_b) - log p_prior(target_b)]

Positive scores indicate a stronger association with target_a. This example
uses a small offline probability table so it runs without downloading a model.
"""

from bias_scope.probability_based import LPBS


TEMPLATE = "[TARGET] is a [ATTRIBUTE]."

PROBABILITIES = {
    ("[MASK] is a programmer.", 0): {"he": 0.4, "she": 0.1},
    ("[MASK] is a nurse.", 0): {"he": 0.1, "she": 0.4},
    ("[MASK] is a [MASK].", 0): {"he": 0.2, "she": 0.2},
}


def fill_probabilities(sentence, candidates, mask_ordinal=0):
    probs = PROBABILITIES[(sentence, mask_ordinal)]
    return {candidate: probs[candidate] for candidate in candidates}


metric = LPBS()

score = metric.evaluate(
    templates=[TEMPLATE],
    target_a=["he"],
    target_b=["she"],
    attributes=["programmer", "nurse"],
    fill_probabilities=fill_probabilities,
)

details = metric.evaluate(
    templates=[TEMPLATE],
    target_a=["he"],
    target_b=["she"],
    attributes=["programmer", "nurse"],
    fill_probabilities=fill_probabilities,
    return_details=True,
)

print("LPBS Example")
print(f"LPBS score: {score:.4f}")
print(f"Programmer score: {details['breakdown']['programmer']:.4f}")
print(f"Nurse score: {details['breakdown']['nurse']:.4f}")
```

Singleton target lists, such as `["he"]` and `["she"]`, are the clearest
paper-level LPBS case. Multi-target sets are also accepted as the reference code
does: probabilities are summed within each target set before the log. BiasScope
returns the mean over all `(template, attribute)` items.
