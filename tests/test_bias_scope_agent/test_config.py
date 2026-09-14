import pytest

from bias_scope_agent.config import AgentConfig, load_config


class TestLoadConfig:
    def test_defaults_with_empty_env(self):
        config = load_config(env={})
        assert config == AgentConfig()

    def test_reads_model_from_env(self):
        config = load_config(env={"BIASSCOPE_AGENT_MODEL": "claude-opus-5"})
        assert config.model == "claude-opus-5"

    def test_reads_max_tokens_from_env(self):
        config = load_config(env={"BIASSCOPE_AGENT_MAX_TOKENS": "512"})
        assert config.max_tokens == 512

    def test_bad_max_tokens_raises_value_error(self):
        with pytest.raises(ValueError):
            load_config(env={"BIASSCOPE_AGENT_MAX_TOKENS": "not-a-number"})

    @pytest.mark.parametrize("value", ["0", "false", "False", "no"])
    def test_inspect_live_disabled_by_falsy_values(self, value):
        config = load_config(env={"BIASSCOPE_AGENT_INSPECT_LIVE": value})
        assert config.inspect_model_live is False

    @pytest.mark.parametrize("value", ["1", "true", "True", "yes"])
    def test_inspect_live_enabled_by_truthy_values(self, value):
        config = load_config(env={"BIASSCOPE_AGENT_INSPECT_LIVE": value})
        assert config.inspect_model_live is True

    def test_defaults_to_real_process_environment(self, monkeypatch):
        monkeypatch.setenv("BIASSCOPE_AGENT_MODEL", "from-real-env")
        config = load_config()
        assert config.model == "from-real-env"

    def test_no_api_key_field_exists(self):
        config = load_config(env={})
        assert not hasattr(config, "api_key")
