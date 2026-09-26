# Backends

A backend is a model plus an honest statement of what can be asked of it. It is what
`recommend_metrics` and `BiasSuite` read to decide which metrics can run.

**Access is derived from the backend, never guessed.** A chat API genuinely cannot return
the masked-token logits a probability metric needs, so no metric that needs them is
offered for it.

| Backend | Use it for | Access |
|---|---|---|
| `HuggingFaceBackend` | Local encoders and causal LMs | `embeddings`, plus `logits` for a masked LM, or `completions` for a causal LM |
| `LiteLLMBackend` | Hosted chat APIs, OpenRouter, local servers | `completions`, `chat` |
| `StubBackend` | Tests, and exercising the framework offline | Whatever you say |

```python
from bias_scope import load_model

encoder = load_model("bert-base-uncased", backend="hf", kind="encoder", dtype="fp32")
causal = load_model("gpt2", backend="hf", kind="causal")
api = load_model("openrouter/meta-llama/llama-3.1-8b-instruct", backend="litellm")

print(encoder.access, causal.access, api.access)
```

## HuggingFaceBackend

`HuggingFaceBackend(model_id, kind="causal", dtype="bf16", device=None)` needs the
`torch` extra.

- `kind="causal"` gives `embeddings` and `completions`. Embeddings come from the model's
  hidden states, pooled over tokens.
- `kind="encoder"` gives `embeddings` and, when the checkpoint really has a masked-LM
  head, `logits`.
- **`logits` is checked, not assumed.** A checkpoint whose config names a masked-LM
  architecture but ships no head weights (the sentence-transformers `all-mpnet-base-v2`
  is one) would score with a randomly initialized head and still report a number. The
  backend reads the weights as well as the config and withholds `logits` if the head is
  missing, so those metrics are not offered. Building an encoder backend may therefore
  read the checkpoint's config from the Hub.
- **`dtype` is part of the result.** `bf16` (the default), `fp16` and `fp32` are
  accepted, and the value is recorded in the protocol of every result, because precision
  can move a bias score. Quantized models are not used for validation runs.
- **`device` is not auto-detected.** Omit it and the model stays on CPU. Pass
  `device="cuda"` to use a GPU.

## LiteLLMBackend

`LiteLLMBackend(model_id, api_key=None, api_base=None)` needs the `llm` extra. Any model
string [LiteLLM](https://github.com/BerriAI/litellm) accepts works, for example
`openai/gpt-4o`, `anthropic/claude-3-5-sonnet-20241022`, or
`openrouter/meta-llama/llama-3.1-8b-instruct`.

It provides `completions` and `chat` but **not** `logits`: most chat APIs expose at best
top-k log-probabilities, which is not the distribution a masked-LM metric needs, so
claiming `logits` would let metrics run on data they cannot use. API credentials come from
the provider's standard environment variable, or `api_key`.

## Generation cache

`cached_generate(backend, prompts, decoding, cache_dir)` stores completions as JSON lines
under `cache_dir`, keyed by a hash of the model, its dtype, the decoding settings and the
prompts. Changing any of them is a cache miss, because silently reusing generations from
a different protocol would be worse than having no cache. Re-running an evaluation then
reuses the text instead of generating it again, and two runs of the same protocol score
the same generations. Without a `cache_dir` it simply generates. The agent's
generation-based datasets cache under `cache/generations/`.

## API reference

::: bias_scope.backends.Backend

::: bias_scope.backends.HuggingFaceBackend

::: bias_scope.backends.LiteLLMBackend

::: bias_scope.backends.StubBackend

::: bias_scope.backends.load_model

::: bias_scope.backends.cached_generate
