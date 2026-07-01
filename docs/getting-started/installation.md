# Installation

## Requirements

- Python >= 3.10
- pip

## Install from PyPI

```bash
pip install bias-scope
```

## Install Optional Extras

```bash
pip install "bias-scope[torch]"
pip install "bias-scope[embeddings]"
pip install "bias-scope[datasets]"
pip install "bias-scope[llm]"
pip install "bias-scope[all]"
```

## Install from Source

```bash
git clone https://github.com/RAINLabLAU/bias_scope.git
cd bias-scope
pip install -e .
```

## Install with Docs Dependencies

```bash
pip install "bias-scope[docs]"
```

## Dependencies

### Core

| Package  | Version  | Purpose                         |
| -------- | -------- | ------------------------------- |
| numpy    | >=2.1.3  | Numerical operations            |
| requests | >=2.28.0 | HTTP requests (Perspective API) |

### Optional Extras

| Extra        | Packages                                                   | Purpose                                    |
| ------------ | ---------------------------------------------------------- | ------------------------------------------ |
| `torch`      | torch, transformers                                        | Probability metrics, `BertPLLScorer`, RegardScore |
| `embeddings` | sentence-transformers                                      | `bias_scope.embeddings_based.embed` helper |
| `datasets`   | datasets                                                   | Prompt-based dataset loaders               |
| `llm`        | litellm                                                    | Prompt-based model calls                   |
| `all`        | torch, transformers, sentence-transformers, datasets, litellm | Full feature set                        |
