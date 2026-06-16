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
git clone https://github.com/RAINLabLAU/bias_scope.git
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
