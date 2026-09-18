"""Model backends (PLAN.md 5.4).

Two backends only:

``HuggingFaceBackend``  encoders and causal LMs — embeddings, logits, completions
``LiteLLMBackend``      chat APIs — completions, chat

vLLM stays in `scripts/` for speed; it is not a library backend.

**Access is derived from the backend, never guessed.** A backend declares what
it can actually do, and `recommend_metrics` and `BiasSuite` use that to decide
which metrics can run. A metric needing `logits` will not be offered for a chat
API, because the chat API genuinely cannot provide them.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Architecture suffixes that carry a masked-language-model head. A checkpoint
# without one still loads through AutoModelForMaskedLM - transformers newly
# initializes the missing head and only warns - so every masked-LM metric would
# return a number computed from random weights (REVIEW_LATER RL-058).
_MASKED_LM_ARCHITECTURES = ("ForMaskedLM", "ForPreTraining")


def _has_masked_lm_head(model_id: str) -> Optional[bool]:
    """True/False from the checkpoint's config, or None if it cannot be read.

    None means "unknown" (offline, a local directory without a config, a repo
    that lists no architectures), not "no": staying optimistic there preserves
    the previous behaviour rather than silently disabling half the library.
    """
    try:
        from transformers import AutoConfig

        architectures = AutoConfig.from_pretrained(model_id).architectures
    except Exception:
        return None
    if not architectures:
        return None
    return any(name.endswith(_MASKED_LM_ARCHITECTURES) for name in architectures)



class Backend(ABC):
    """A model, and an honest statement of what can be asked of it."""

    #: Which access modes this backend provides. Subset of
    #: `bias_scope.metadata.ACCESS_MODES`.
    access: Tuple[str, ...] = ()

    #: Identifier recorded in every result's protocol block.
    model_id: str = ""

    #: Numeric precision, recorded in the protocol. AWQ-INT4 moved BBQ by 17
    #: points in the v0.1 reproduction, so this is not cosmetic.
    dtype: Optional[str] = None

    def supports(self, required: Sequence[str]) -> bool:
        """True when every required access mode is available here."""
        return set(required) <= set(self.access)

    @abstractmethod
    def generate(self, prompts: Sequence[str], **decoding: Any) -> List[str]:
        """One completion per prompt."""

    def protocol_fields(self) -> Dict[str, Any]:
        """The backend's contribution to a result's protocol block."""
        return {"model_id": self.model_id, "dtype": self.dtype}


class HuggingFaceBackend(Backend):
    """
    Local transformers models.

    An encoder gives `embeddings` and `logits`; a causal LM adds `completions`.
    Nothing here provides `chat` — an instruction-tuned model loaded through
    transformers is still being asked for raw completions, and pretending
    otherwise would let chat-only metrics run on the wrong access path.
    """

    def __init__(
        self,
        model_id: str,
        kind: str = "causal",
        dtype: str = "bf16",
        device: Optional[str] = None,
    ):
        """
        Args:
            model_id (str): Hugging Face model identifier.
            kind (str): "causal" or "encoder".
            dtype (str): "bf16" or "fp32". PLAN.md Section 1 forbids quantized
                models in validation runs.
            device (str, optional): Passed to transformers.

        Raises:
            ValueError: If `kind` or `dtype` is not recognised.
        """
        if kind not in ("causal", "encoder"):
            raise ValueError(f"kind must be 'causal' or 'encoder', got {kind!r}")
        if dtype not in ("bf16", "fp32", "fp16"):
            raise ValueError(f"dtype must be 'bf16', 'fp32' or 'fp16', got {dtype!r}")

        self.model_id = model_id
        self.kind = kind
        self.dtype = dtype
        self.device = device
        # `logits` here means *masked-token* logits, which is the only kind
        # anything in this library consumes: every metric declaring `logits`
        # scores through a masked-LM scorer (scorers.py's BertPLLScorer and
        # WordPieceBertScorer build AutoModelForMaskedLM; cbs.py and
        # topk_fill_divergence.py load it directly; LPBS and DisCoMetric take a
        # caller-supplied masked-token predictor). A causal LM has next-token
        # logits and no masked-token prediction, so advertising `logits` for it
        # got all 11 probability metrics recommended and every one of them
        # failed with "Unrecognized configuration class" (REVIEW_LATER RL-057).
        # If a causal-logits metric is ever added, split this into two access
        # modes rather than widening this one back.
        # An encoder offers `logits` only if its checkpoint actually has a
        # masked-LM head; see _has_masked_lm_head. `lm_head_verified` records
        # whether that could be established, so an unreadable config is
        # distinguishable from a confirmed head.
        self.lm_head_verified = False
        if kind == "causal":
            self.access = ("embeddings", "completions")
        else:
            has_head = _has_masked_lm_head(model_id)
            self.lm_head_verified = has_head is True
            self.access = ("embeddings",) if has_head is False else ("embeddings", "logits")
        self._model = None
        self._tokenizer = None

    def _load(self):
        """Load lazily, so constructing a backend costs nothing."""
        if self._model is not None:
            return self._tokenizer, self._model

        try:
            import torch
            import transformers
        except ImportError as exc:  # pragma: no cover - needs the extra missing
            raise ImportError(
                "HuggingFaceBackend requires torch and transformers. Install "
                "bias-scope[torch]."
            ) from exc

        torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
                       "fp32": torch.float32}[self.dtype]
        loader = (
            transformers.AutoModelForCausalLM
            if self.kind == "causal"
            else transformers.AutoModel
        )
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_id)
        self._model = loader.from_pretrained(self.model_id, dtype=torch_dtype)
        self._model.eval()
        if self.device:
            self._model.to(self.device)
        return self._tokenizer, self._model

    def generate(self, prompts: Sequence[str], **decoding: Any) -> List[str]:
        """Greedy or sampled continuations, one per prompt."""
        if self.kind != "causal":
            raise ValueError(
                f"{self.model_id} was loaded as an encoder, which cannot generate. "
                "Use kind='causal'."
            )
        import torch

        tokenizer, model = self._load()
        max_new_tokens = decoding.pop("max_new_tokens", 20)
        outputs: List[str] = []
        for prompt in prompts:
            encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                generated = model.generate(
                    **encoded, max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.eos_token_id, **decoding,
                )
            text = tokenizer.decode(
                generated[0][encoded["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )
            outputs.append(text)
        return outputs


class LiteLLMBackend(Backend):
    """
    Chat APIs through LiteLLM.

    Provides `completions` and `chat`. **Not** `logits`: most chat APIs expose
    at best top-k token logprobs, which is not the same as the full
    distribution a masked-LM metric needs, so claiming `logits` here would let
    metrics run on data they cannot actually use.
    """

    access = ("completions", "chat")

    def __init__(
        self,
        model_id: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ):
        self.model_id = model_id
        self.api_key = api_key
        self.api_base = api_base
        self.dtype = None

    def generate(self, prompts: Sequence[str], **decoding: Any) -> List[str]:
        """One chat completion per prompt."""
        try:
            from litellm import completion
        except ImportError as exc:  # pragma: no cover - needs the extra missing
            raise ImportError(
                "LiteLLMBackend requires litellm. Install bias-scope[llm]."
            ) from exc

        outputs: List[str] = []
        for prompt in prompts:
            response = completion(
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
                api_key=self.api_key,
                api_base=self.api_base,
                **decoding,
            )
            outputs.append(response.choices[0].message.content or "")
        return outputs


class StubBackend(Backend):
    """
    Canned answers, for tests and for exercising the framework offline.

    Declares whatever access the caller says, so a test can check that
    `recommend_metrics` and `BiasSuite` really do respect access constraints
    rather than offering everything.
    """

    def __init__(
        self,
        answers: Sequence[str] = ("stub answer",),
        access: Sequence[str] = ("completions", "chat"),
        model_id: str = "stub/model",
    ):
        if not answers:
            raise ValueError("answers must contain at least one response")
        self.answers = list(answers)
        self.access = tuple(access)
        self.model_id = model_id
        self.dtype = None
        self.calls: List[str] = []

    def generate(self, prompts: Sequence[str], **decoding: Any) -> List[str]:
        outputs = []
        for i, prompt in enumerate(prompts):
            self.calls.append(prompt)
            outputs.append(self.answers[min(i, len(self.answers) - 1)])
        return outputs


def load_model(
    model_id: str,
    backend: str = "hf",
    **kwargs: Any,
) -> Backend:
    """
    Build a backend.

    Args:
        model_id (str): Model identifier.
        backend (str): "hf" or "litellm".
        **kwargs: Passed to the backend constructor.

    Returns:
        Backend

    Examples:
        >>> load_model("stub/model", backend="stub").access
        ('completions', 'chat')
    """
    if backend == "hf":
        return HuggingFaceBackend(model_id, **kwargs)
    if backend == "litellm":
        return LiteLLMBackend(model_id, **kwargs)
    if backend == "stub":
        return StubBackend(model_id=model_id, **kwargs)
    raise ValueError(f"backend must be 'hf', 'litellm' or 'stub', got {backend!r}")


# ── Generation cache (PLAN.md 5.4) ────────────────────────────────────────
def cached_generate(
    backend: Backend,
    prompts: Sequence[str],
    decoding: Optional[Dict[str, Any]] = None,
    cache_dir: Optional[Path] = None,
) -> List[str]:
    """
    Generate, reusing a cache keyed by the protocol hash.

    Section 1 requires generations to be cacheable so that metrics can be
    recomputed without regenerating. The key covers the model, the decoding
    parameters and the prompts, so any change to any of them is a cache miss —
    silently reusing generations from a different protocol would be worse than
    no cache.

    Stored as JSON lines under `cache_dir`. No database.

    Returns:
        list of str: One completion per prompt.
    """
    from bias_scope.utils import protocol_hash

    decoding = dict(decoding or {})
    if cache_dir is None:
        return backend.generate(prompts, **decoding)

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = protocol_hash(
        {
            "model_id": backend.model_id,
            "dtype": backend.dtype,
            "decoding": decoding,
            "prompts_sha": hashlib.sha256(
                "\n".join(prompts).encode("utf-8")
            ).hexdigest(),
        }
    )
    path = cache_dir / f"{key}.jsonl"

    if path.exists():
        records = [json.loads(line) for line in path.read_text().splitlines() if line]
        if len(records) == len(prompts):
            cached_generate.hits += 1
            return [r["completion"] for r in records]

    cached_generate.misses += 1
    completions = backend.generate(prompts, **decoding)
    path.write_text(
        "\n".join(
            json.dumps({"prompt": p, "completion": c})
            for p, c in zip(prompts, completions)
        )
        + "\n"
    )
    return completions


#: Counters so a test can assert a second run made no generation calls, which
#: PLAN.md 5.4 lists as an acceptance check.
cached_generate.hits = 0
cached_generate.misses = 0


__all__ = [
    "Backend",
    "HuggingFaceBackend",
    "LiteLLMBackend",
    "StubBackend",
    "load_model",
    "cached_generate",
]
