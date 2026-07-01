# bias-scope

**A comprehensive Python library for detecting and measuring biases in machine learning models.**

bias-scope provides a unified API for evaluating bias across four methodological categories: embedding-based tests, probability-based metrics, generated text analysis, and prompt-based evaluations. It supports models accessed through HuggingFace Transformers, sentence-transformers, and LiteLLM-compatible providers.

## Install

```bash
pip install bias-scope
```

Optional extras:

```bash
pip install "bias-scope[torch]"
pip install "bias-scope[embeddings]"
pip install "bias-scope[datasets]"
pip install "bias-scope[llm]"
pip install "bias-scope[all]"
```

Or from source:

```bash
git clone https://github.com/RAINLabLAU/bias_scope.git
cd bias_scope
pip install -e .
```

## Quick Example

```python
from bias_scope.embeddings_based import WEAT

weat = WEAT(model_name="sentence-transformers/all-MiniLM-L6-v2")
score = weat.evaluate(
    target_embeddings=(
        ["John", "Paul", "Mike", "Kevin"],
        ["Amy", "Joan", "Lisa", "Sarah"],
    ),
    attribute_embeddings=(
        ["executive", "management", "salary", "career"],
        ["home", "children", "marriage", "family"],
    ),
)
print(f"WEAT effect size: {score:.4f}")
```

## Metric Categories

| Category                                                                   | Metrics | Description                                              |
| -------------------------------------------------------------------------- | ------- | -------------------------------------------------------- |
| [Embedding-Based](api/overview.md#embedding-based)                         | 4       | WEAT, SEAT, CEAT, SentenceBiasScore                      |
| [Probability-Based](api/overview.md#probability-based)                     | 9       | CrowS-Pairs, CAT, iCAT, AUL, AULA, LMB, LPBS, CBS, DisCo |
| [Generated Text](api/overview.md#generated-text-based)                     | 17      | RegardScore, ScoreParity, ToxicityFraction, and more     |
| [Prompt-Based](api/overview.md#prompt-based)                               | 11      | BBQ, StereoSet, TruthfulQA, BOLD, and more               |
