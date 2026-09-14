import json
from unittest.mock import patch

from bias_scope_agent.introspection import inspect_model, metrics_needing_data


class TestMetricsNeedingData:
    def test_weat_needs_target_and_attribute_embeddings(self):
        result = metrics_needing_data(["WEAT"])
        assert result["WEAT"] == ["target_embeddings", "attribute_embeddings"]

    def test_bbq_needs_nothing_because_it_loads_its_own_dataset(self):
        result = metrics_needing_data(["BBQMetric"])
        assert result["BBQMetric"] == []

    def test_reports_both_metrics_requested(self):
        result = metrics_needing_data(["WEAT", "BBQMetric"])
        assert set(result.keys()) == {"WEAT", "BBQMetric"}

    def test_unknown_metric_name_is_reported_as_needing_data_conservatively(self):
        # A metric that failed to import or doesn't exist can't be introspected;
        # treat it as "needs data" rather than silently claiming it needs nothing.
        result = metrics_needing_data(["NotARealMetric"])
        assert result["NotARealMetric"] == ["<metric class not importable>"]


class TestInspectModelLocalPath:
    def test_reads_local_config_json_causal(self, tmp_path):
        config = {"architectures": ["GPT2LMHeadModel"], "model_type": "gpt2"}
        (tmp_path / "config.json").write_text(json.dumps(config))

        result = inspect_model(str(tmp_path), live=True)

        assert result["guessed_source"] == "local_path"
        assert result["guessed_kind"] == "causal"
        assert result["confidence"] == "high"

    def test_reads_local_config_json_encoder(self, tmp_path):
        config = {"architectures": ["BertForMaskedLM"], "model_type": "bert"}
        (tmp_path / "config.json").write_text(json.dumps(config))

        result = inspect_model(str(tmp_path), live=True)

        assert result["guessed_kind"] == "encoder"

    def test_missing_config_json_degrades_to_low_confidence(self, tmp_path):
        result = inspect_model(str(tmp_path), live=True)
        assert result["guessed_kind"] is None
        assert result["confidence"] == "low"
        assert result["notes"]


class TestInspectModelHfHub:
    def test_hf_hub_id_uses_hf_hub_download_when_live(self):
        config = {"architectures": ["GPT2LMHeadModel"], "model_type": "gpt2"}
        with patch("bias_scope_agent.introspection._download_hf_config", return_value=config):
            result = inspect_model("some-org/some-model", live=True)
        assert result["guessed_source"] == "hf_hub"
        assert result["guessed_kind"] == "causal"
        assert result["confidence"] == "high"

    def test_hf_hub_failure_degrades_gracefully(self):
        with patch(
            "bias_scope_agent.introspection._download_hf_config",
            side_effect=RuntimeError("network unreachable"),
        ):
            result = inspect_model("some-org/some-model", live=True)
        assert result["guessed_kind"] is None
        assert result["confidence"] == "low"
        assert any("network unreachable" in note for note in result["notes"])

    def test_live_false_never_calls_hf_hub_download(self):
        with patch(
            "bias_scope_agent.introspection._download_hf_config",
            side_effect=AssertionError("must not be called when live=False"),
        ):
            result = inspect_model("some-org/some-model", live=False)
        assert result["confidence"] == "low"
        assert result["guessed_kind"] is None


class TestInspectModelApiEndpoint:
    def test_url_like_identifier_is_treated_as_api_endpoint(self):
        result = inspect_model("https://api.example.com/v1", live=False)
        assert result["guessed_source"] == "api_endpoint"

    def test_api_base_argument_forces_api_endpoint_classification(self):
        result = inspect_model("my-model", api_base="https://api.example.com/v1", live=False)
        assert result["guessed_source"] == "api_endpoint"

    def test_live_probe_success_reports_chat_formatted(self):
        fake_response = {"choices": [{"message": {"content": "hi"}}]}
        target = "bias_scope_agent.introspection._probe_completion_api"
        with patch(target, return_value=fake_response):
            result = inspect_model(
                "my-model", api_base="https://api.example.com/v1", live=True
            )
        assert result["chat_formatted"] is True

    def test_live_probe_failure_degrades_gracefully(self):
        with patch(
            "bias_scope_agent.introspection._probe_completion_api",
            side_effect=RuntimeError("401 unauthorized"),
        ):
            result = inspect_model(
                "my-model", api_base="https://api.example.com/v1", live=True
            )
        assert result["confidence"] == "low"
        assert any("401 unauthorized" in note for note in result["notes"])

    def test_live_false_never_probes_the_api(self):
        with patch(
            "bias_scope_agent.introspection._probe_completion_api",
            side_effect=AssertionError("must not be called when live=False"),
        ):
            result = inspect_model(
                "my-model", api_base="https://api.example.com/v1", live=False
            )
        assert result["confidence"] == "low"


class TestInspectModelReturnShape:
    def test_return_shape_has_all_expected_keys(self, tmp_path):
        result = inspect_model(str(tmp_path), live=False)
        expected_keys = {
            "identifier", "guessed_source", "guessed_kind", "has_lm_head",
            "chat_formatted", "supports_logprobs", "confidence", "notes", "raw",
        }
        assert expected_keys <= set(result.keys())

    def test_never_raises_on_unexpected_input(self):
        # inspect_model must always return a best-guess dict, never raise.
        result = inspect_model("", live=True)
        assert isinstance(result, dict)
