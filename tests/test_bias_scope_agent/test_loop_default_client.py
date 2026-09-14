"""AgentLoop's default-client construction (client=None) should fail fast
and legibly, at __init__ time, rather than several frames deep inside the
anthropic SDK on the first real turn. Confirmed by manual reproduction
(Item 3 of the follow-up plan): with no ANTHROPIC_API_KEY, the SDK's own
failure is a bare TypeError raised from deep inside `_base_client.py`,
first surfacing on the first `messages.create` call, not at construction.
"""

import sys
from unittest.mock import patch

import pytest

from bias_scope_agent.config import AgentConfig
from bias_scope_agent.loop import AgentLoop
from bias_scope_agent.session import AgentSession


class TestDefaultClientMissingApiKey:
    def test_raises_immediately_naming_the_env_var(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
            AgentLoop(AgentConfig(), AgentSession())

    def test_does_not_require_a_turn_to_surface(self, monkeypatch):
        # The failure must happen at construction, not on the first run_turn.
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(RuntimeError):
            loop = AgentLoop(AgentConfig(), AgentSession())
            loop.run_turn("hello")  # pragma: no cover - never reached


class TestDefaultClientMissingPackage:
    def test_raises_immediately_naming_the_extra(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-placeholder")
        with patch.dict(sys.modules, {"anthropic": None}):
            with pytest.raises(RuntimeError, match=r"bias-scope\[agent\]"):
                AgentLoop(AgentConfig(), AgentSession())


class TestExplicitClientBypassesBothChecks:
    def test_explicit_client_never_touches_env_or_import(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        fake_client = object()
        loop = AgentLoop(AgentConfig(), AgentSession(), client=fake_client)
        assert loop.client is fake_client
