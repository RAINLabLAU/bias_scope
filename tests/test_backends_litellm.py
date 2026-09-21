"""LiteLLMBackend.generate must speak the chat API's parameter names.

The dataset providers pass Hugging Face decoding names (`max_new_tokens`,
`do_sample`, `top_k`); a chat API takes `max_tokens`, `temperature`, `top_p`.
Passing the HF names straight through made every generation-based provider
fail on an API-served target (2026-09-21). Mocked: no network.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

pytest.importorskip("litellm")

from bias_scope.backends import LiteLLMBackend  # noqa: E402


def _response(text):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


def test_hf_decoding_names_are_translated_to_chat_api_names():
    backend = LiteLLMBackend("openrouter/some/model")
    with patch("litellm.completion", return_value=_response("out")) as completion:
        backend.generate(["p"], max_new_tokens=30, do_sample=False, top_k=50)
    kwargs = completion.call_args.kwargs
    assert kwargs["max_tokens"] == 30
    assert kwargs["temperature"] == 0.0          # greedy
    assert "max_new_tokens" not in kwargs and "do_sample" not in kwargs and "top_k" not in kwargs


def test_sampling_keeps_temperature_and_top_p():
    backend = LiteLLMBackend("openrouter/some/model")
    with patch("litellm.completion", return_value=_response("out")) as completion:
        backend.generate(["p"], max_new_tokens=20, do_sample=True, top_p=0.9, temperature=1.0)
    kwargs = completion.call_args.kwargs
    assert kwargs["top_p"] == 0.9 and kwargs["temperature"] == 1.0 and kwargs["max_tokens"] == 20


def test_one_call_per_prompt_and_a_none_reply_is_an_empty_string():
    backend = LiteLLMBackend("openrouter/some/model")
    with patch("litellm.completion", side_effect=[_response("a"), _response(None)]) as completion:
        out = backend.generate(["p1", "p2"])
    assert out == ["a", ""] and completion.call_count == 2
