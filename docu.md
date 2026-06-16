# bias-scope MkDocs Documentation — Claude Code Instructions

## Your Task

Build a complete MkDocs documentation site with the Material theme for the `bias-scope` Python library. The site must be fully functional, deployable to Read the Docs, and include a full API reference page for every metric in the library.

---

## Step 1 — Install Dependencies

Run these commands in the repo root:

```bash
pip install "mkdocs>=1.6.1" mkdocs-material mkdocstrings[python] mkdocs-gen-files mkdocs-literate-nav
```

Then update `pyproject.toml` docs dependencies to:

```toml
[project.optional-dependencies]
docs = [
    "mkdocs>=1.6.1",
    "mkdocs-material>=9.0.0",
    "mkdocstrings[python]>=0.25.0",
    "mkdocs-gen-files>=0.5.0",
    "mkdocs-literate-nav>=0.6.0",
]
```

---

## Step 2 — Create `mkdocs.yml`

Create `mkdocs.yml` at the repo root with this exact content:

```yaml
site_name: bias-scope
site_description: A comprehensive Python library for detecting and measuring biases in machine learning models.
site_author: Chadi Helwe, Nancy Kalmiche, Elissa El Haber, Jason Greich
site_url: https://bias-scope.readthedocs.io
repo_url: https://github.com/YOUR_ORG/bias-scope
repo_name: YOUR_ORG/bias-scope
edit_uri: edit/main/docs/

theme:
  name: material
  logo: assets/logo.png
  favicon: assets/logo.png
  palette:
    - scheme: default
      primary: indigo
      accent: indigo
      toggle:
        icon: material/brightness-7
        name: Switch to dark mode
    - scheme: slate
      primary: indigo
      accent: indigo
      toggle:
        icon: material/brightness-4
        name: Switch to light mode
  features:
    - navigation.tabs
    - navigation.tabs.sticky
    - navigation.sections
    - navigation.expand
    - navigation.path
    - navigation.top
    - search.suggest
    - search.highlight
    - content.code.copy
    - content.code.annotate
    - toc.follow

plugins:
  - search
  - mkdocstrings:
      handlers:
        python:
          paths: [src]
          options:
            show_source: true
            show_root_heading: true
            show_symbol_type_heading: true
            show_symbol_type_toc: true
            docstring_style: google
            merge_init_into_class: true

markdown_extensions:
  - admonition
  - pymdownx.details
  - pymdownx.superfences
  - pymdownx.highlight:
      anchor_linenums: true
  - pymdownx.inlinehilite
  - pymdownx.snippets
  - pymdownx.tabbed:
      alternate_style: true
  - attr_list
  - md_in_html
  - tables
  - toc:
      permalink: true

nav:
  - Home: index.md
  - Getting Started:
    - Installation: getting-started/installation.md
    - Quick Start: getting-started/quickstart.md
  - API Reference:
    - Overview: api/overview.md
    - Embedding-Based:
      - WEAT: api/embeddings/weat.md
      - SEAT: api/embeddings/seat.md
      - CEAT: api/embeddings/ceat.md
      - SentenceBiasScore: api/embeddings/sentence_bias_score.md
    - Probability-Based:
      - CrowS-Pairs: api/probability/crows_pairs.md
      - CAT: api/probability/cat.md
      - iCAT: api/probability/icat.md
      - AUL: api/probability/aul.md
      - AULA: api/probability/aula.md
      - LMB: api/probability/lmb.md
      - LPBS: api/probability/lpbs.md
      - CBS: api/probability/cbs.md
      - DisCo: api/probability/disco.md
    - Generated Text:
      - RegardScore: api/generated_text/regard_score.md
      - ScoreParity: api/generated_text/score_parity.md
      - ToxicityFraction: api/generated_text/toxicity_fraction.md
      - ToxicityProbability: api/generated_text/toxicity_probability.md
      - SocialGroupSubstitution: api/generated_text/social_group_substitution.md
      - CoOccurrenceBiasScore: api/generated_text/cooccurrence_bias_score.md
      - DemographicRepresentation: api/generated_text/demographic_representation.md
      - StereotypicalAssociations: api/generated_text/stereotypical_associations.md
      - MarkedPersons: api/generated_text/marked_persons.md
      - EMT: api/generated_text/emt.md
      - FGB: api/generated_text/fgb.md
      - PGB: api/generated_text/pgb.md
      - GenderPolarity: api/generated_text/gender_polarity.md
      - HONEST: api/generated_text/honest.md
      - PsycholinguisticNorms: api/generated_text/psycholinguistic_norms.md
      - CounterfactualSentimentBias: api/generated_text/counterfactual_sentiment_bias.md
      - PerspectiveAPIClient: api/generated_text/perspective_api.md
    - Prompt-Based:
      - BBQ: api/prompts/bbq.md
      - StereoSet: api/prompts/stereoset.md
      - TruthfulQA: api/prompts/truthfulqa.md
      - BOLD: api/prompts/bold.md
      - RealToxicityPrompts: api/prompts/realtoxicityprompts.md
      - CounterfactualFairness: api/prompts/counterfactual_fairness.md
      - DemographicRepresentationBias: api/prompts/demographic_representation_bias.md
      - AnalogicalReasoningBias: api/prompts/analogical_reasoning_bias.md
      - TofNof: api/prompts/tof_nof.md
      - OpinionConsistencyAcrossPersonas: api/prompts/opinion_consistency_across_personas.md
      - UnQover: api/prompts/unqover.md
  - Authors: authors.md
```

---

## Step 3 — Create the `docs/` Folder Structure

Create every file listed below. The exact content for each file follows.

```
docs/
  index.md
  authors.md
  assets/
    logo.png              ← copy from assets/logo.png in repo root
  getting-started/
    installation.md
    quickstart.md
  api/
    overview.md
    embeddings/
      weat.md
      seat.md
      ceat.md
      sentence_bias_score.md
    probability/
      crows_pairs.md
      cat.md
      icat.md
      aul.md
      aula.md
      lmb.md
      lpbs.md
      cbs.md
      disco.md
    generated_text/
      regard_score.md
      score_parity.md
      toxicity_fraction.md
      toxicity_probability.md
      social_group_substitution.md
      cooccurrence_bias_score.md
      demographic_representation.md
      stereotypical_associations.md
      marked_persons.md
      emt.md
      fgb.md
      pgb.md
      gender_polarity.md
      honest.md
      psycholinguistic_norms.md
      counterfactual_sentiment_bias.md
      perspective_api.md
    prompts/
      bbq.md
      stereoset.md
      truthfulqa.md
      bold.md
      realtoxicityprompts.md
      counterfactual_fairness.md
      demographic_representation_bias.md
      analogical_reasoning_bias.md
      tof_nof.md
      opinion_consistency_across_personas.md
      unqover.md
```

**Copy the logo:**

```bash
cp assets/logo.png docs/assets/logo.png
```

---

## Step 4 — File Contents

### `docs/index.md`

```markdown
# bias-scope

**A comprehensive Python library for detecting and measuring biases in machine learning models.**

bias-scope provides a unified API for evaluating bias across four methodological categories: embedding-based tests, probability-based metrics, generated text analysis, and prompt-based evaluations. It supports virtually any model accessible via HuggingFace Transformers or LiteLLM-compatible APIs.

## Install

```bash
pip install bias-scope
```

Or from source:

```bash
pip install git+https://github.com/YOUR_ORG/bias-scope.git
```

## Quick Example

```python
from sentence_transformers import SentenceTransformer
from bias_scope.embeddings_based import WEAT

model = SentenceTransformer("all-MiniLM-L6-v2")

male_emb   = model.encode(["John", "Paul", "Mike", "Kevin"])
female_emb = model.encode(["Amy", "Joan", "Lisa", "Sarah"])
career_emb = model.encode(["executive", "management", "salary", "career"])
family_emb = model.encode(["home", "children", "marriage", "family"])

weat = WEAT()
score = weat.evaluate(
    target_embeddings=(male_emb, female_emb),
    attribute_embeddings=(career_emb, family_emb),
)
print(f"WEAT effect size: {score:.4f}")
```

## Metric Categories

| Category                                                                   | Metrics | Description                                              |
| -------------------------------------------------------------------------- | ------- | -------------------------------------------------------- |
| [Embedding-Based](https://claude.ai/chat/api/overview.md#embedding-based)     | 4       | WEAT, SEAT, CEAT, SentenceBiasScore                      |
| [Probability-Based](https://claude.ai/chat/api/overview.md#probability-based) | 9       | CrowS-Pairs, CAT, iCAT, AUL, AULA, LMB, LPBS, CBS, DisCo |
| [Generated Text](https://claude.ai/chat/api/overview.md#generated-text-based) | 17      | RegardScore, ScoreParity, ToxicityFraction, and more     |
| [Prompt-Based](https://claude.ai/chat/api/overview.md#prompt-based)           | 11      | BBQ, StereoSet, TruthfulQA, BOLD, and more               |

```

---

### `docs/getting-started/installation.md`

```markdown
# Installation

## Requirements

- Python >= 3.10
- pip

## Install from PyPI

```bash
pip install bias-scope
```

## Install from Source

```bash
git clone https://github.com/YOUR_ORG/bias-scope.git
cd bias-scope
pip install -e .
```

## Install with Docs Dependencies

```bash
pip install "bias-scope[docs]"
```

## Dependencies

| Package               | Version  | Purpose                                   |
| --------------------- | -------- | ----------------------------------------- |
| numpy                 | >=2.1.3  | Numerical operations                      |
| torch                 | >=2.5.1  | Tensor operations for probability metrics |
| transformers          | >=4.31.0 | HuggingFace model loading                 |
| datasets              | >=5.0.0  | HuggingFace dataset loading               |
| litellm               | >=1.0.0  | LLM API access for prompt-based metrics   |
| sentence-transformers | >=2.0.0  | Sentence encoding                         |
| requests              | >=2.28.0 | HTTP requests (Perspective API)           |

```

---

### `docs/getting-started/quickstart.md`

```markdown
# Quick Start

This page shows one complete example for each metric category.

## Embedding-Based: WEAT

```python
from sentence_transformers import SentenceTransformer
from bias_scope.embeddings_based import WEAT

model = SentenceTransformer("all-MiniLM-L6-v2")

male_names   = ["John", "Paul", "Mike", "Kevin", "Steve", "Greg", "Jeff", "Bill"]
female_names = ["Amy",  "Joan", "Lisa", "Sarah", "Diana", "Kate", "Ann",  "Donna"]
career_words = ["executive", "management", "professional", "corporation",
                "salary", "office", "business", "career"]
family_words = ["home", "parents", "children", "family",
                "cousins", "marriage", "wedding", "relatives"]

male_emb   = model.encode(male_names)
female_emb = model.encode(female_names)
career_emb = model.encode(career_words)
family_emb = model.encode(family_words)

weat = WEAT()
score = weat.evaluate(
    target_embeddings=(male_emb, female_emb),
    attribute_embeddings=(career_emb, family_emb),
)

print(f"WEAT effect size: {score:.4f}")
# > 0  -> male names associate more with career words
# < 0  -> male names associate more with family words
# ~0   -> no differential association (no bias)
```

## Probability-Based: CrowS-Pairs

```python
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from bias_scope.probability_based import CrowSPairs

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
bert = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
bert.eval()

def predict_masked_token(sentence, mask_position):
    text = " ".join(sentence)
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"][0]
    mask_positions = (input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
    if len(mask_positions) == 0:
        return 0.5
    mask_idx = mask_positions[0].item()
    with torch.no_grad():
        logits = bert(**inputs).logits[0, mask_idx]
    return torch.softmax(logits, dim=-1).max().item()

pairs = [
    (["Women", "are", "bad", "at", "math"],
     ["Men",   "are", "bad", "at", "math"]),
    (["He", "is", "a", "nurse"],
     ["She", "is", "a", "nurse"]),
]

crows = CrowSPairs()
score = crows.evaluate(pairs, predict_masked_token)
print(f"CrowS-Pairs bias score: {score:.2%}")
# > 50% -> model prefers stereotype sentences
```

## Generated Text: RegardScore

```python
from bias_scope.generated_text_based import RegardScore

regard = RegardScore(model_name="siebert/sentiment-roberta-large-english")

group_a_texts = [
    ["The man worked as a software engineer.",
     "The man worked as a senior executive."],
]
group_b_texts = [
    ["The woman worked as a nurse.",
     "The woman worked as a school teacher."],
]

result = regard.evaluate(group_a_texts, group_b_texts)
print(f"Regard difference: {result['regard_difference']:.4f}")
```

## Prompt-Based: BBQ

```python
from bias_scope.prompts_based import BBQMetric

metric = BBQMetric(
    model_name="openrouter/meta-llama/llama-3.1-8b-instruct",
    api_key="your-api-key-here",
)

result = metric.evaluate(num_samples=20, subset="Age")
print(f"Bias score: {result['bias_score']:.2%}")
print(f"Accuracy:   {result['accuracy']:.2%}")
```

## Supported LLM Providers (Prompt-Based Metrics)

Prompt-based metrics use [LiteLLM](https://github.com/BerriAI/litellm). Pass any of the following as `model_name`:

| Provider      | Example model string                                |
| ------------- | --------------------------------------------------- |
| OpenAI        | `openai/gpt-4o`                                   |
| Anthropic     | `anthropic/claude-3-5-sonnet-20241022`            |
| Google Gemini | `gemini/gemini-1.5-flash`                         |
| OpenRouter    | `openrouter/meta-llama/llama-3.1-8b-instruct`     |
| HuggingFace   | `huggingface/meta-llama/Meta-Llama-3-8B-Instruct` |

```

---

### `docs/api/overview.md`

```markdown
# API Overview

Every metric in bias-scope inherits from a base class and implements a single `evaluate()` method.

## Base Classes

```python
from bias_scope.base import (
    BiasMetric,          # abstract base
    EmbeddingMetric,     # embedding-based metrics
    ProbabilityMetric,   # probability-based metrics
    GeneratedTextMetric, # generated text metrics
    PromptBasedMetric,   # prompt-based metrics
)
```

All metrics follow the same pattern:

```python
metric = MyMetric(...)        # initialize
result = metric.evaluate(...) # evaluate — returns float or dict
print(metric.category)        # "embedding" | "probability" | "generated_text" | "prompt_based"
```

---

## Embedding-Based

Operate on embedding arrays. No model API required.

| Class                 | Import                                                        |
| --------------------- | ------------------------------------------------------------- |
| `WEAT`              | `from bias_scope.embeddings_based import WEAT`              |
| `SEAT`              | `from bias_scope.embeddings_based import SEAT`              |
| `CEAT`              | `from bias_scope.embeddings_based import CEAT`              |
| `SentenceBiasScore` | `from bias_scope.embeddings_based import SentenceBiasScore` |

---

## Probability-Based

Use masked language model token probabilities. Require a model and a prediction function.

| Class           | Import                                                   |
| --------------- | -------------------------------------------------------- |
| `CrowSPairs`  | `from bias_scope.probability_based import CrowSPairs`  |
| `CAT`         | `from bias_scope.probability_based import CAT`         |
| `ICAT`        | `from bias_scope.probability_based import ICAT`        |
| `AUL`         | `from bias_scope.probability_based import AUL`         |
| `AULA`        | `from bias_scope.probability_based import AULA`        |
| `LMB`         | `from bias_scope.probability_based import LMB`         |
| `LPBS`        | `from bias_scope.probability_based import LPBS`        |
| `CBS`         | `from bias_scope.probability_based import CBS`         |
| `DisCoMetric` | `from bias_scope.probability_based import DisCoMetric` |

---

## Generated Text-Based

Analyze text already generated by a model. You provide the generations.

| Class                           | Import                                                                      |
| ------------------------------- | --------------------------------------------------------------------------- |
| `RegardScore`                 | `from bias_scope.generated_text_based import RegardScore`                 |
| `ScoreParity`                 | `from bias_scope.generated_text_based import ScoreParity`                 |
| `ToxicityFraction`            | `from bias_scope.generated_text_based import ToxicityFraction`            |
| `ToxicityProbability`         | `from bias_scope.generated_text_based import ToxicityProbability`         |
| `SocialGroupSubstitution`     | `from bias_scope.generated_text_based import SocialGroupSubstitution`     |
| `CoOccurrenceBiasScore`       | `from bias_scope.generated_text_based import CoOccurrenceBiasScore`       |
| `DemographicRepresentation`   | `from bias_scope.generated_text_based import DemographicRepresentation`   |
| `StereotypicalAssociations`   | `from bias_scope.generated_text_based import StereotypicalAssociations`   |
| `MarkedPersons`               | `from bias_scope.generated_text_based import MarkedPersons`               |
| `EMT`                         | `from bias_scope.generated_text_based import EMT`                         |
| `FGB`                         | `from bias_scope.generated_text_based import FGB`                         |
| `PGB`                         | `from bias_scope.generated_text_based import PGB`                         |
| `GenderPolarity`              | `from bias_scope.generated_text_based import GenderPolarity`              |
| `HONEST`                      | `from bias_scope.generated_text_based import HONEST`                      |
| `PsycholinguisticNorms`       | `from bias_scope.generated_text_based import PsycholinguisticNorms`       |
| `CounterfactualSentimentBias` | `from bias_scope.generated_text_based import CounterfactualSentimentBias` |
| `PerspectiveAPIClient`        | `from bias_scope.generated_text_based import PerspectiveAPIClient`        |

---

## Prompt-Based

Send prompts to a live LLM via API. Require a model name and API key.

| Class                                | Import                                                                    |
| ------------------------------------ | ------------------------------------------------------------------------- |
| `BBQMetric`                        | `from bias_scope.prompts_based import BBQMetric`                        |
| `StereoSetMetric`                  | `from bias_scope.prompts_based import StereoSetMetric`                  |
| `TruthfulQA`                       | `from bias_scope.prompts_based import TruthfulQA`                       |
| `BOLD`                             | `from bias_scope.prompts_based import BOLD`                             |
| `RealToxicityPrompts`              | `from bias_scope.prompts_based import RealToxicityPrompts`              |
| `CounterfactualFairness`           | `from bias_scope.prompts_based import CounterfactualFairness`           |
| `DemographicRepresentationBias`    | `from bias_scope.prompts_based import DemographicRepresentationBias`    |
| `AnalogicalReasoningBias`          | `from bias_scope.prompts_based import AnalogicalReasoningBias`          |
| `TofNof`                           | `from bias_scope.prompts_based import TofNof`                           |
| `OpinionConsistencyAcrossPersonas` | `from bias_scope.prompts_based import OpinionConsistencyAcrossPersonas` |
| `UnQoverMetric`                    | `from bias_scope.prompts_based import UnQoverMetric`                    |

```

---

### Individual API Reference Pages — Template

**For every metric page** listed in the nav, create the file using the following template. Replace the placeholders for each metric.

Every metric page has three sections:

1. Auto-generated API docs pulled from the class docstring using `mkdocstrings`
2. A hand-written example (copy from `examples/` folder in the repo)
3. The original paper reference

**Template:**

```markdown
# {MetricName}

::: bias_scope.{module_path}.{ClassName}

## Example

```python
{paste full content of examples/{category}/{filename}.py here}
```

## Reference

{AuthorLastName}, {Initials}., et al. ({Year}). {Paper Title}. {Venue}.

```

---

### All Metric Pages — Exact Content

Create each file below. For each one, the `:::` directive pulls docstrings automatically. The example is copied from the corresponding file in `examples/`.

---

#### `docs/api/embeddings/weat.md`
```markdown
# WEAT

::: bias_scope.embeddings_based.weat.WEAT

## Example

```python
# copy full content of examples/embeddings_based/weat.py
```

## Reference

Caliskan, A., Bryson, J. J., & Narayanan, A. (2017). Semantics derived automatically from language corpora contain human-like biases.  *Science* , 356(6334), 183–186.

```

#### `docs/api/embeddings/seat.md`
```markdown
# SEAT

::: bias_scope.embeddings_based.seat.SEAT

## Example

```python
# copy full content of examples/embeddings_based/seat.py
```

## Reference

May, C., Wang, A., Bordia, S., Bowman, S. R., & Rudinger, R. (2019). On measuring social biases in sentence encoders.  *NAACL-HLT 2019* .

```

#### `docs/api/embeddings/ceat.md`
```markdown
# CEAT

::: bias_scope.embeddings_based.ceat.CEAT

## Example

```python
# copy full content of examples/embeddings_based/ceat.py
```

## Reference

Guo, W., & Caliskan, A. (2021). Detecting Emergent Intersectional Biases: Contextualized Word Embeddings Contain a Distribution of Human-like Biases.  *AIES 2021* .

```

#### `docs/api/embeddings/sentence_bias_score.md`
```markdown
# SentenceBiasScore

::: bias_scope.embeddings_based.sentence_bias_score.SentenceBiasScore

## Example

```python
# copy full content of examples/embeddings_based/sentence_bias_score.py
```

## Reference

Dolci, M., Azzalini, D., & Tanelli, M. (2023). Sentence-level bias detection in transformer models.

```

#### `docs/api/probability/crows_pairs.md`
```markdown
# CrowS-Pairs

::: bias_scope.probability_based.crows_pairs.CrowSPairs

## Example

```python
# copy full content of examples/probability_based/crows_pairs.py
```

## Reference

Nangia, N., Ying, C., Goodman, A., & Bowman, S. R. (2020). CrowS-Pairs: A Challenge Dataset for Measuring Social Biases in Masked Language Models.  *EMNLP 2020* .

```

#### `docs/api/probability/cat.md`
```markdown
# CAT

::: bias_scope.probability_based.cat.CAT

## Example

```python
# copy full content of examples/probability_based/cat.py
```

## Reference

Nadeem, M., Bethke, A., & Reddy, S. (2021). StereoSet: Measuring stereotypical bias in pretrained language models.  *ACL 2021* .

```

#### `docs/api/probability/icat.md`
```markdown
# iCAT

::: bias_scope.probability_based.icat.ICAT

## Example

```python
# copy full content of examples/probability_based/icat.py
```

## Reference

Nadeem, M., Bethke, A., & Reddy, S. (2021). StereoSet: Measuring stereotypical bias in pretrained language models.  *ACL 2021* .

```

#### `docs/api/probability/aul.md`
```markdown
# AUL

::: bias_scope.probability_based.aul.AUL

## Example

```python
# copy full content of examples/probability_based/aul.py
```

## Reference

Kaneko, M., & Bollegala, D. (2022). Unmasking the Mask – Evaluating Social Biases in Masked Language Models.  *AAAI 2022* .

```

#### `docs/api/probability/aula.md`
```markdown
# AULA

::: bias_scope.probability_based.aula.AULA

## Example

```python
# copy full content of examples/probability_based/aula.py
```

## Reference

Kaneko, M., & Bollegala, D. (2022). Unmasking the Mask – Evaluating Social Biases in Masked Language Models.  *AAAI 2022* .

```

#### `docs/api/probability/lmb.md`
```markdown
# LMB

::: bias_scope.probability_based.lmb.LMB

## Example

```python
# copy full content of examples/probability_based/lmb.py
```

## Reference

Barikeri, S., Lauscher, A., Vulić, I., & Glavaš, G. (2021). RedditBias: A Real-World Resource for Bias Evaluation and Debiasing of Conversational Language Models.  *ACL-IJCNLP 2021* .

```

#### `docs/api/probability/lpbs.md`
```markdown
# LPBS

::: bias_scope.probability_based.lpbs.LPBS

## Example

```python
# copy full content of examples/probability_based/lpbs.py
```

```

#### `docs/api/probability/cbs.md`
```markdown
# CBS

::: bias_scope.probability_based.cbs.CBS

## Example

```python
# copy full content of examples/probability_based/cbs.py
```

```

#### `docs/api/probability/disco.md`
```markdown
# DisCo

::: bias_scope.probability_based.disco.DisCoMetric

## Example

```python
# copy full content of examples/probability_based/disco.py
```

```

#### `docs/api/generated_text/regard_score.md`
```markdown
# RegardScore

::: bias_scope.generated_text_based.regard_score.RegardScore

## Example

```python
# copy full content of examples/generated_text_based/regard_score.py
```

## Reference

Sheng, E., Chang, K. W., Natarajan, P., & Peng, N. (2019). The Woman Worked as a Babysitter: On Biases in Language Generation.  *EMNLP 2019* .

```

#### `docs/api/generated_text/score_parity.md`
```markdown
# ScoreParity

::: bias_scope.generated_text_based.score_parity.ScoreParity

## Example

```python
# copy full content of examples/generated_text_based/score_parity.py
```

## Reference

Borkan, D., Dixon, L., Sorensen, J., Thain, N., & Vasserman, L. (2019). Nuanced Metrics for Measuring Unintended Bias with Real Data for Text Classification.  *WWW 2019* .

```

#### For all remaining generated text pages, follow the same pattern:
- `toxicity_fraction.md` → `bias_scope.generated_text_based.toxicity_fraction.ToxicityFraction`
- `toxicity_probability.md` → `bias_scope.generated_text_based.toxicity_probability.ToxicityProbability`
- `social_group_substitution.md` → `bias_scope.generated_text_based.social_group_substitution.SocialGroupSubstitution`
- `cooccurrence_bias_score.md` → `bias_scope.generated_text_based.cooccurrence_bias_score.CoOccurrenceBiasScore`
- `demographic_representation.md` → `bias_scope.generated_text_based.demographic_representation.DemographicRepresentation`
- `stereotypical_associations.md` → `bias_scope.generated_text_based.stereotypical_associations.StereotypicalAssociations`
- `marked_persons.md` → `bias_scope.generated_text_based.marked_persons.MarkedPersons`
- `emt.md` → `bias_scope.generated_text_based.emt.EMT`
- `fgb.md` → `bias_scope.generated_text_based.fgb.FGB`
- `pgb.md` → `bias_scope.generated_text_based.pgb.PGB`
- `gender_polarity.md` → `bias_scope.generated_text_based.gender_polarity.GenderPolarity`
- `honest.md` → `bias_scope.generated_text_based.honest.HONEST`
- `psycholinguistic_norms.md` → `bias_scope.generated_text_based.psycholinguistic_norms.PsycholinguisticNorms`
- `counterfactual_sentiment_bias.md` → `bias_scope.generated_text_based.counterfactual_sentiment_bias.CounterfactualSentimentBias`
- `perspective_api.md` → `bias_scope.generated_text_based.perspective_api.PerspectiveAPIClient`

#### For all prompt-based pages, same pattern:
- `bbq.md` → `bias_scope.prompts_based.bbq.BBQMetric`
- `stereoset.md` → `bias_scope.prompts_based.stereoset.StereoSetMetric`
- `truthfulqa.md` → `bias_scope.prompts_based.truthfulqa.TruthfulQA`
- `bold.md` → `bias_scope.prompts_based.bold.BOLD`
- `realtoxicityprompts.md` → `bias_scope.prompts_based.realtoxicityprompts.RealToxicityPrompts`
- `counterfactual_fairness.md` → `bias_scope.prompts_based.counterfactual_fairness.CounterfactualFairness`
- `demographic_representation_bias.md` → `bias_scope.prompts_based.demographic_representation_bias.DemographicRepresentationBias`
- `analogical_reasoning_bias.md` → `bias_scope.prompts_based.analogical_reasoning_bias.AnalogicalReasoningBias`
- `tof_nof.md` → `bias_scope.prompts_based.tof_nof.TofNof`
- `opinion_consistency_across_personas.md` → `bias_scope.prompts_based.opinion_consistency_across_personas.OpinionConsistencyAcrossPersonas`
- `unqover.md` → `bias_scope.prompts_based.unqover.UnQoverMetric`

For each, paste the full corresponding example from `examples/prompts_based/`.

---

### `docs/authors.md`

```markdown
# Authors

bias-scope is developed and maintained by:

| Name | Role |
|------|------|
| Chadi Helwe | Lead Developer |
| Nancy Kalmiche | Contributor |
| Elissa El Haber | Contributor |
| Jason Greich | Contributor |

## Citation

```bibtex
@software{biasscope2024,
  title   = {bias-scope: A Comprehensive Library for Bias Detection in Language Models},
  author  = {Helwe, Chadi and Kalmiche, Nancy and El Haber, Elissa and Greich, Jason},
  year    = {2024},
  url     = {https://github.com/YOUR_ORG/bias-scope}
}
```

When using bias-scope, please also cite the original papers for the specific metrics you use.

```

---

## Step 5 — Add Read the Docs Config

Create `.readthedocs.yaml` at the repo root:

```yaml
version: 2

build:
  os: ubuntu-22.04
  tools:
    python: "3.11"

mkdocs:
  configuration: mkdocs.yml

python:
  install:
    - method: pip
      path: .
      extra_requirements:
        - docs
```

---

## Step 6 — Test Locally

```bash
mkdocs serve
```

Open `http://127.0.0.1:8000`. Fix any errors before committing.

To build the static site:

```bash
mkdocs build
```

Output goes to `site/` (already in `.gitignore`).

---

## Step 7 — Final Checklist Before Handing Off to DR

* [ ] `mkdocs serve` runs without errors
* [ ] All 4 category tabs appear in the top nav
* [ ] Every metric has its own page with auto-rendered docstring
* [ ] Every metric page has a working code example
* [ ] Dark/light mode toggle works
* [ ] Logo appears in header
* [ ] `.readthedocs.yaml` is at repo root
* [ ] `pyproject.toml` docs dependencies are updated

```

```
