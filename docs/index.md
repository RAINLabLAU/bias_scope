**A Python library for measuring bias in language models, with an optional conversational agent.**

BiasScope puts bias metrics from four methodological families behind one consistent
interface: embedding-based tests, probability-based metrics, generated-text analysis, and
prompt-based benchmarks. It works with sentence encoders, masked language models, causal
language models and chat APIs, and it tells you which metrics a given model can actually
support and how closely each implementation follows the paper it cites.

## Install

```bash
pip install bias-scope
```

Heavier dependencies are optional extras:

```bash
pip install "bias-scope[torch]"      # local Hugging Face models
pip install "bias-scope[llm]"        # chat APIs through LiteLLM
pip install "bias-scope[agent]"      # the conversational agent
pip install "bias-scope[all]"        # everything
```

See [Installation](getting-started/installation.md) for every extra, or install from source:

```bash
git clone https://github.com/RAINLabLAU/bias_scope.git
cd bias_scope
pip install -e .
```

## Quick example

```python
import numpy as np
from bias_scope.embeddings_based import WEAT

# Rows from a static word-embedding table; use the paper's full word lists in a
# real experiment.
X = np.array([[1.0, 0.0], [0.9, 0.1]])
Y = np.array([[0.0, 1.0], [0.1, 0.9]])
A = np.array([[1.0, 0.0], [0.95, 0.05]])
B = np.array([[0.0, 1.0], [0.05, 0.95]])

details = WEAT().evaluate((X, Y), (A, B), return_details=True)
print(f"WEAT effect size: {details['effect_size']:.4f}")
```

## Three ways to use it

**One metric at a time.** Every metric has an `evaluate()` method. Start with the
[Quick Start](getting-started/quickstart.md) and the [API overview](api/overview.md).

**Many metrics against one model.** [`recommend_metrics`](framework/recommend.md) says which
metrics a model can run, [`BiasSuite`](framework/suite.md) runs them, and a
[`Report`](framework/results.md) shows every score with its confidence interval and
fidelity.

**In conversation.** The [agent](agent/index.md) works out how your model can be accessed,
proposes a plan, and runs it only after you confirm.

## Metric families

<!-- families:start -->
| Family | Metrics | Needs | Examples |
|---|---:|---|---|
| [Embedding-based](api/overview.md#embedding-based) | 4 | embeddings | WEAT, SEAT, CEAT, SentenceBiasScore |
| [Probability-based](api/overview.md#probability-based) | 11 | masked-token logits | CrowS-Pairs, CAT, iCAT, AUL, AULA, LPBS, CBS, DisCo |
| [Generated text](api/overview.md#generated-text-based) | 17 | completions | RegardScore, HONEST, MarkedPersons, EMT, and more |
| [Prompt-based](api/overview.md#prompt-based) | 24 | chat or completions | BBQ, StereoSet, WinoBias, DecodingTrust, TrustLLM, and more |
<!-- families:end -->

## What sets it apart

**Fidelity is stated, not implied.** Every metric declares whether it is `faithful` to its
paper, an `adaptation` that can change the numbers, or BiasScope's own `original`. The
label is set only after the paper and the authors' code have been read. See the
[fidelity index](fidelity/INDEX.md) and [Metric metadata](framework/metadata.md).

**Access is derived, not guessed.** A chat API cannot return the logits a probability
metric needs, so it is not offered one.

**A metric that cannot run is skipped with a reason, never scored as zero.** And there is no
composite score, because metrics with different units do not average.

**Every result carries its protocol**: model, dtype, seed and resources, so a number can be
reproduced.
