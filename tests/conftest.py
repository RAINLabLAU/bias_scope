"""Shared fixtures for the BiasScope test suite.

Four fixtures, all CPU-only and all runnable without an API key:

``tiny_encoder``     tiny random BERT             masked-LM path
``tiny_causal``      ``sshleifer/tiny-gpt2``      causal-LM path
``tiny_embeddings``  seeded random vectors        embedding path, no network
``stub_chat``        canned-answer chat backend   prompt path, no network

The two model fixtures download once into the Hugging Face cache under
``~/.cache`` and are session-scoped. When transformers is not installed or the
download fails (offline), they skip with a message naming the model and the
extra to install rather than failing the suite.
"""

from typing import Callable, Dict, List, Sequence
from unittest.mock import MagicMock

import numpy as np
import pytest

# PLAN.md Section 1 names `prajjwal1/bert-tiny` first, but that repo ships no
# tokenizer.json and transformers 5.x cannot convert its slow tokenizer
# ("You need to have sentencepiece or tiktoken installed"), with or without
# use_fast=False. `hf-internal-testing/tiny-random-*` is the other family the
# plan sanctions, loads cleanly, and is smaller.
TINY_ENCODER_ID = "hf-internal-testing/tiny-random-BertForMaskedLM"
TINY_CAUSAL_ID = "sshleifer/tiny-gpt2"

# Seed and shape for tiny_embeddings. Fixed so a golden test can rely on them.
EMBEDDING_SEED = 42
EMBEDDING_DIM = 16
EMBEDDING_SET_SIZE = 8


def _load_tiny_model(model_id: str, loader_name: str):
    """Load a tiny HF model, or skip the test with an actionable message."""
    try:
        import transformers
    except ImportError:  # pragma: no cover - exercised only without the extra
        pytest.skip(
            f"{model_id} needs transformers; install bias-scope[torch] to run this test."
        )

    loader = getattr(transformers, loader_name)
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
        model = loader.from_pretrained(model_id)
    except (OSError, ValueError) as exc:  # pragma: no cover - environment-only
        # OSError: offline or the repo moved. ValueError: the tokenizer cannot
        # be built in this transformers version. Both are environment problems,
        # not test failures, but the cause is named so they are not mysterious.
        pytest.skip(f"could not load {model_id}: {type(exc).__name__}: {exc}")

    model.eval()
    return tokenizer, model


@pytest.fixture(scope="session")
def tiny_encoder():
    """(tokenizer, model) for TINY_ENCODER_ID, in eval mode."""
    return _load_tiny_model(TINY_ENCODER_ID, "AutoModelForMaskedLM")


@pytest.fixture(scope="session")
def tiny_causal():
    """(tokenizer, model) for TINY_CAUSAL_ID, in eval mode."""
    return _load_tiny_model(TINY_CAUSAL_ID, "AutoModelForCausalLM")


@pytest.fixture
def tiny_embeddings() -> Dict[str, np.ndarray]:
    """
    Four seeded embedding sets shaped like WEAT/SEAT input.

    Returns ``{"target_a", "target_b", "attribute_a", "attribute_b"}``, each
    ``(8, 16)``. Drawn from ``np.random.default_rng(42)``, so the values are
    identical on every machine and every run, and independent of any
    ``seed_everything`` call the test under test may make.
    """
    rng = np.random.default_rng(EMBEDDING_SEED)
    return {
        name: rng.standard_normal((EMBEDDING_SET_SIZE, EMBEDDING_DIM))
        for name in ("target_a", "target_b", "attribute_a", "attribute_b")
    }


class StubChat:
    """
    Canned-answer stand-in for ``litellm.completion``.

    Call it like ``completion(model=..., messages=[...])``; it returns a
    response shaped like litellm's (``response.choices[0].message.content``)
    and records the call. Answers are consumed in order and the last one
    repeats once the queue is exhausted, so a test does not have to know how
    many calls a metric makes.

    Wire it into a metric with the module-level patch the suite already uses::

        with patch("bias_scope.prompts_based.bbq.completion", stub_chat):
            ...
    """

    def __init__(self, answers: Sequence[str]):
        if not answers:
            raise ValueError("answers must contain at least one response")
        self._answers = list(answers)
        self._index = 0
        self.calls: List[dict] = []

    @property
    def call_count(self) -> int:
        """Number of completions requested so far."""
        return len(self.calls)

    @property
    def prompts(self) -> List[str]:
        """The content of the last message of each recorded call."""
        return [call["messages"][-1]["content"] for call in self.calls]

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        answer = self._answers[min(self._index, len(self._answers) - 1)]
        self._index += 1

        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message = MagicMock()
        response.choices[0].message.content = answer
        return response


@pytest.fixture
def stub_chat() -> Callable[..., StubChat]:
    """
    Factory for a :class:`StubChat` backend.

    ``stub_chat(["A", "B"])`` returns a callable that answers "A" then "B",
    repeating "B" thereafter, and exposes ``.call_count`` and ``.prompts``.
    """
    return StubChat
