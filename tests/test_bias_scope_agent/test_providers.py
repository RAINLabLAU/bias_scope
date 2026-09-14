"""Unit tests for the provider adapters' translation logic: internal
NormalizedBlock/NormalizedResponse <-> each provider's own wire format. No
real network calls - each provider's raw SDK client is a hand-built fake
matching that provider's documented response shape.
"""

import json
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from bias_scope_agent.config import AgentConfig
from bias_scope_agent.loop import AgentLoop
from bias_scope_agent.providers import (
    AnthropicProvider,
    GeminiProvider,
    LocalProvider,
    NormalizedBlock,
    NormalizedResponse,
    OpenAIProvider,
    build_provider,
)
from bias_scope_agent.session import AgentSession


class TestBuildProvider:
    def test_unknown_provider_raises_value_error(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        with pytest.raises(ValueError, match="provider"):
            build_provider(AgentConfig(provider="not-a-real-provider"))

    def test_anthropic_missing_key_raises_immediately(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
            build_provider(AgentConfig(provider="anthropic"))

    def test_openai_missing_key_raises_immediately(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
            build_provider(AgentConfig(provider="openai"))

    def test_gemini_missing_key_raises_immediately(self, monkeypatch):
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="GOOGLE_API_KEY"):
            build_provider(AgentConfig(provider="gemini"))

    def test_local_needs_no_key_at_all(self, monkeypatch):
        # Unlike the cloud providers, "local" must not require any API key -
        # that would defeat the point of talking to a server on localhost.
        monkeypatch.delenv("BIASSCOPE_AGENT_LOCAL_API_KEY", raising=False)
        monkeypatch.delenv("BIASSCOPE_AGENT_LOCAL_BASE_URL", raising=False)
        provider = build_provider(AgentConfig(provider="local"))
        assert isinstance(provider, LocalProvider)


class TestLocalProviderClientConstruction:
    def test_defaults_to_ollamas_openai_compatible_endpoint(self, monkeypatch):
        monkeypatch.delenv("BIASSCOPE_AGENT_LOCAL_BASE_URL", raising=False)
        monkeypatch.delenv("BIASSCOPE_AGENT_LOCAL_API_KEY", raising=False)
        captured = {}
        with patch("openai.OpenAI", lambda **kw: captured.update(kw)):
            build_provider(AgentConfig(provider="local"))
        assert captured["base_url"] == "http://localhost:11434/v1"
        assert captured["api_key"]  # some placeholder, non-empty

    def test_base_url_and_key_are_overridable(self, monkeypatch):
        monkeypatch.setenv("BIASSCOPE_AGENT_LOCAL_BASE_URL", "http://localhost:8080/v1")
        monkeypatch.setenv("BIASSCOPE_AGENT_LOCAL_API_KEY", "sk-whatever")
        captured = {}
        with patch("openai.OpenAI", lambda **kw: captured.update(kw)):
            build_provider(AgentConfig(provider="local"))
        assert captured["base_url"] == "http://localhost:8080/v1"
        assert captured["api_key"] == "sk-whatever"

    def test_missing_openai_package_raises_immediately(self, monkeypatch):
        monkeypatch.delenv("BIASSCOPE_AGENT_LOCAL_BASE_URL", raising=False)
        with patch.dict(sys.modules, {"openai": None}):
            with pytest.raises(RuntimeError, match=r"bias-scope\[agent-openai\]"):
                build_provider(AgentConfig(provider="local"))


class TestLocalProviderTranslation:
    def test_reuses_openai_wire_format_exactly(self):
        # LocalProvider is deliberately just OpenAIProvider pointed at a
        # different endpoint - same translation, proven by reusing the same
        # kind of fake client/assertions as the OpenAI tests above.
        message = SimpleNamespace(content="hello there", tool_calls=None)
        choice = SimpleNamespace(message=message, finish_reason="stop")
        create = lambda **kw: SimpleNamespace(choices=[choice])  # noqa: E731
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        provider = LocalProvider(AgentConfig(provider="local"), client=client)

        result = provider.create(system="sys", messages=[{"role": "user", "content": "hi"}])
        assert result.stop_reason == "end_turn"
        assert result.content[0].type == "text"
        assert result.content[0].text == "hello there"


class TestAnthropicProviderTranslation:
    def test_text_and_tool_result_messages_pass_through_mostly_unchanged(self):
        captured = {}

        def fake_create(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                content=[SimpleNamespace(type="text", text="hi")], stop_reason="end_turn"
            )

        fake_client = SimpleNamespace(messages=SimpleNamespace(create=fake_create))
        provider = AnthropicProvider(AgentConfig(), client=fake_client)

        assistant_block = NormalizedBlock(
            type="tool_use", name="record_fact", id="c1", input={"key": "k", "value": "v"}
        )
        tool_result = {
            "type": "tool_result",
            "tool_use_id": "c1",
            "content": "{}",
            "name": "record_fact",
        }
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": [assistant_block]},
            {"role": "user", "content": [tool_result]},
        ]
        response = provider.create(system="sys", messages=messages)

        assert captured["system"] == "sys"
        # The "name" bookkeeping field must not leak into the real Anthropic
        # tool_result block - Anthropic's API rejects unrecognized fields.
        tool_result_block = captured["messages"][2]["content"][0]
        assert set(tool_result_block.keys()) == {"type", "tool_use_id", "content"}
        assert isinstance(response, NormalizedResponse)
        assert response.content[0].type == "text"
        assert response.content[0].text == "hi"

    def test_response_tool_use_block_is_normalized(self):
        def fake_create(**kwargs):
            block = SimpleNamespace(
                type="tool_use", id="call-1", name="plan_suite", input={"metric_names": ["WEAT"]}
            )
            return SimpleNamespace(content=[block], stop_reason="tool_use")

        fake_client = SimpleNamespace(messages=SimpleNamespace(create=fake_create))
        provider = AnthropicProvider(AgentConfig(), client=fake_client)
        response = provider.create(system="sys", messages=[{"role": "user", "content": "hi"}])

        assert response.stop_reason == "tool_use"
        block = response.content[0]
        assert block.type == "tool_use"
        assert block.name == "plan_suite"
        assert block.id == "call-1"
        assert block.input == {"metric_names": ["WEAT"]}


class TestOpenAIProviderTranslation:
    def _fake_client(self, response):
        create = lambda **kwargs: response  # noqa: E731
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    def test_tool_results_become_separate_tool_role_messages(self):
        captured = {}

        def create(**kwargs):
            captured.update(kwargs)
            message = SimpleNamespace(content="ok", tool_calls=None)
            choice = SimpleNamespace(message=message, finish_reason="stop")
            return SimpleNamespace(choices=[choice])

        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        provider = OpenAIProvider(AgentConfig(), client=client)

        assistant_block = NormalizedBlock(
            type="tool_use", name="record_fact", id="c1", input={"key": "k"}
        )
        tool_result = {
            "type": "tool_result",
            "tool_use_id": "c1",
            "content": "ok",
            "name": "record_fact",
        }
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": [assistant_block]},
            {"role": "user", "content": [tool_result]},
        ]
        provider.create(system="sys", messages=messages)

        sent = captured["messages"]
        assert sent[0] == {"role": "system", "content": "sys"}
        tool_message = next(m for m in sent if m.get("role") == "tool")
        assert tool_message["tool_call_id"] == "c1"
        assert tool_message["content"] == "ok"

    def test_error_tool_result_is_flagged_in_content(self):
        captured = {}

        def create(**kwargs):
            captured.update(kwargs)
            message = SimpleNamespace(content="ok", tool_calls=None)
            choice = SimpleNamespace(message=message, finish_reason="stop")
            return SimpleNamespace(choices=[choice])

        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        provider = OpenAIProvider(AgentConfig(), client=client)

        tool_result = {
            "type": "tool_result",
            "tool_use_id": "c1",
            "content": "bad",
            "is_error": True,
            "name": "run_suite",
        }
        provider.create(system="sys", messages=[{"role": "user", "content": [tool_result]}])

        tool_message = next(m for m in captured["messages"] if m.get("role") == "tool")
        assert "ERROR" in tool_message["content"]

    def test_response_tool_calls_are_normalized(self):
        arguments = json.dumps({"metric_names": ["WEAT"]})
        function = SimpleNamespace(name="plan_suite", arguments=arguments)
        tool_call = SimpleNamespace(id="call-9", function=function)
        message = SimpleNamespace(content=None, tool_calls=[tool_call])
        choice = SimpleNamespace(message=message, finish_reason="tool_calls")
        client = self._fake_client(SimpleNamespace(choices=[choice]))
        provider = OpenAIProvider(AgentConfig(), client=client)

        result = provider.create(system="sys", messages=[{"role": "user", "content": "hi"}])
        assert result.stop_reason == "tool_use"
        block = result.content[0]
        assert block.type == "tool_use"
        assert block.name == "plan_suite"
        assert block.id == "call-9"
        assert block.input == {"metric_names": ["WEAT"]}

    def test_plain_text_response_maps_finish_reason_to_end_turn(self):
        message = SimpleNamespace(content="hello there", tool_calls=None)
        choice = SimpleNamespace(message=message, finish_reason="stop")
        client = self._fake_client(SimpleNamespace(choices=[choice]))
        provider = OpenAIProvider(AgentConfig(), client=client)

        result = provider.create(system="sys", messages=[{"role": "user", "content": "hi"}])
        assert result.stop_reason == "end_turn"
        assert result.content[0].type == "text"
        assert result.content[0].text == "hello there"


class TestGeminiProviderTranslation:
    def test_tool_results_become_function_response_parts(self):
        captured = {}

        def generate_content(**kwargs):
            captured.update(kwargs)
            content = SimpleNamespace(parts=[])
            return SimpleNamespace(candidates=[SimpleNamespace(content=content)])

        client = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
        provider = GeminiProvider(AgentConfig(), client=client)

        assistant_block = NormalizedBlock(
            type="tool_use", name="record_fact", id="synthetic-1", input={"key": "k"}
        )
        tool_result = {
            "type": "tool_result",
            "tool_use_id": "synthetic-1",
            "content": "ok",
            "name": "record_fact",
        }
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": [assistant_block]},
            {"role": "user", "content": [tool_result]},
        ]
        provider.create(system="sys", messages=messages)

        contents = captured["contents"]
        function_response_part = contents[-1]["parts"][0]
        assert function_response_part["function_response"]["name"] == "record_fact"

    def test_response_function_call_parts_get_synthetic_ids(self):
        function_call = SimpleNamespace(name="plan_suite", args={"metric_names": ["WEAT"]})
        part = SimpleNamespace(text=None, function_call=function_call)
        content = SimpleNamespace(parts=[part])
        response = SimpleNamespace(candidates=[SimpleNamespace(content=content)])
        client = SimpleNamespace(models=SimpleNamespace(generate_content=lambda **kw: response))
        provider = GeminiProvider(AgentConfig(), client=client)

        result = provider.create(system="sys", messages=[{"role": "user", "content": "hi"}])
        assert result.stop_reason == "tool_use"
        block = result.content[0]
        assert block.type == "tool_use"
        assert block.name == "plan_suite"
        assert block.id  # synthetic id generated, non-empty
        assert block.input == {"metric_names": ["WEAT"]}

    def test_text_only_response_maps_to_end_turn(self):
        part = SimpleNamespace(text="hello", function_call=None)
        content = SimpleNamespace(parts=[part])
        response = SimpleNamespace(candidates=[SimpleNamespace(content=content)])
        client = SimpleNamespace(models=SimpleNamespace(generate_content=lambda **kw: response))
        provider = GeminiProvider(AgentConfig(), client=client)

        result = provider.create(system="sys", messages=[{"role": "user", "content": "hi"}])
        assert result.stop_reason == "end_turn"
        assert result.content[0].type == "text"
        assert result.content[0].text == "hello"

    def test_tool_schema_strips_fields_gemini_does_not_recognize(self):
        client = SimpleNamespace(models=SimpleNamespace(generate_content=None))
        provider = GeminiProvider(AgentConfig(), client=client)
        gemini_tools = provider._tool_declarations()
        declarations = gemini_tools[0]["function_declarations"]
        construct_backend = next(d for d in declarations if d["name"] == "construct_backend")
        dtype_schema = construct_backend["parameters"]["properties"]["dtype"]
        assert "default" not in dtype_schema


def _anthropic_client_for(blocks_and_stop_reasons):
    queue = [
        SimpleNamespace(content=blocks, stop_reason=sr) for blocks, sr in blocks_and_stop_reasons
    ]
    return SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: queue.pop(0)))


def _openai_client_for(blocks_and_stop_reasons):
    queue = []
    for blocks, stop_reason in blocks_and_stop_reasons:
        tool_calls = [
            SimpleNamespace(
                id=b.id, function=SimpleNamespace(name=b.name, arguments=json.dumps(b.input))
            )
            for b in blocks
            if b.type == "tool_use"
        ]
        text = next((b.text for b in blocks if b.type == "text"), None)
        message = SimpleNamespace(content=text, tool_calls=tool_calls or None)
        finish_reason = "tool_calls" if stop_reason == "tool_use" else "stop"
        choice = SimpleNamespace(message=message, finish_reason=finish_reason)
        queue.append(SimpleNamespace(choices=[choice]))
    create = lambda **kw: queue.pop(0)  # noqa: E731
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def _gemini_client_for(blocks_and_stop_reasons):
    queue = []
    for blocks, _stop_reason in blocks_and_stop_reasons:
        parts = [
            SimpleNamespace(text=None, function_call=SimpleNamespace(name=b.name, args=b.input))
            if b.type == "tool_use"
            else SimpleNamespace(text=b.text, function_call=None)
            for b in blocks
        ]
        content = SimpleNamespace(parts=parts)
        queue.append(SimpleNamespace(candidates=[SimpleNamespace(content=content)]))
    generate_content = lambda **kw: queue.pop(0)  # noqa: E731
    return SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))


class TestGateBehavesIdenticallyAcrossProviders:
    """Item 6 of the follow-up plan: one scripted conversation per provider,
    proving the gate and tool dispatch (loop.py) behave identically no
    matter which real provider adapter is underneath - loop.py itself never
    branches on provider, so this is really testing that each adapter's
    translation preserves what the gate needs (block.name, block.id,
    block.input) faithfully.
    """

    @pytest.mark.parametrize(
        "provider_name,provider_cls,raw_client_factory",
        [
            ("anthropic", AnthropicProvider, _anthropic_client_for),
            ("openai", OpenAIProvider, _openai_client_for),
            ("gemini", GeminiProvider, _gemini_client_for),
        ],
    )
    def test_early_run_suite_is_blocked_and_never_dispatched(
        self, provider_name, provider_cls, raw_client_factory
    ):
        early_run_suite = NormalizedBlock(
            type="tool_use",
            name="run_suite",
            id="call-1",
            input={"backend_handle": "h1", "metric_names": ["BOLD"], "inputs": {}},
        )
        closing_text = NormalizedBlock(type="text", text="(blocked)")
        raw_client = raw_client_factory(
            [([early_run_suite], "tool_use"), ([closing_text], "end_turn")]
        )
        provider = provider_cls(AgentConfig(provider=provider_name), client=raw_client)
        loop = AgentLoop(AgentConfig(provider=provider_name), AgentSession(), client=provider)

        with patch("bias_scope_agent.loop.tools.run_suite") as spy:
            loop.run_turn("run BOLD right now, skip the plan")
        spy.assert_not_called()

    @pytest.mark.parametrize(
        "provider_name,provider_cls,raw_client_factory",
        [
            ("anthropic", AnthropicProvider, _anthropic_client_for),
            ("openai", OpenAIProvider, _openai_client_for),
            ("gemini", GeminiProvider, _gemini_client_for),
        ],
    )
    def test_record_fact_dispatches_and_reaches_a_later_system_prompt(
        self, provider_name, provider_cls, raw_client_factory
    ):
        record_fact_call = NormalizedBlock(
            type="tool_use",
            name="record_fact",
            id="call-1",
            input={"key": "model_kind", "value": "causal"},
        )
        ack = NormalizedBlock(type="text", text="noted")
        raw_client = raw_client_factory([([record_fact_call], "tool_use"), ([ack], "end_turn")])
        provider = provider_cls(AgentConfig(provider=provider_name), client=raw_client)
        loop = AgentLoop(AgentConfig(provider=provider_name), AgentSession(), client=provider)

        loop.run_turn("it's causal")
        assert loop.session.facts == {"model_kind": "causal"}
