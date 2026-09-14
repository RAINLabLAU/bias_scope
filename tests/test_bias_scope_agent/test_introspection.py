import json
import sys
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


class TestInspectModelSeq2SeqDetection:
    """HuggingFaceBackend only supports kind='causal' or 'encoder' - a
    seq2seq/encoder-decoder model (T5, BART, ...) is not usable there at
    all. inspect_model should say so explicitly rather than leave it as an
    unexplained low-confidence guess."""

    def test_t5_architecture_flagged_as_unsupported(self, tmp_path):
        config = {"architectures": ["T5ForConditionalGeneration"], "model_type": "t5"}
        (tmp_path / "config.json").write_text(json.dumps(config))

        result = inspect_model(str(tmp_path), live=True)

        assert result["guessed_kind"] is None
        notes_lower = [note.lower() for note in result["notes"]]
        assert any("seq2seq" in note or "encoder-decoder" in note for note in notes_lower)

    def test_bart_model_type_flagged_as_unsupported(self, tmp_path):
        config = {"architectures": ["BartModel"], "model_type": "bart"}
        (tmp_path / "config.json").write_text(json.dumps(config))

        result = inspect_model(str(tmp_path), live=True)

        assert result["guessed_kind"] is None
        assert any("bart" in note.lower() or "seq2seq" in note.lower() for note in result["notes"])


class TestInspectModelModelTypeFallback:
    """When the architecture-string heuristic can't classify (e.g. a config
    that only names model_type, not architectures), fall back to a lookup
    of well-known model_type values, widening high-confidence coverage."""

    def test_llama_model_type_classified_as_causal(self, tmp_path):
        config = {"model_type": "llama"}
        (tmp_path / "config.json").write_text(json.dumps(config))

        result = inspect_model(str(tmp_path), live=True)

        assert result["guessed_kind"] == "causal"
        assert result["confidence"] == "high"

    def test_roberta_model_type_classified_as_encoder(self, tmp_path):
        config = {"model_type": "roberta"}
        (tmp_path / "config.json").write_text(json.dumps(config))

        result = inspect_model(str(tmp_path), live=True)

        assert result["guessed_kind"] == "encoder"
        assert result["confidence"] == "high"

    def test_unknown_model_type_with_no_architectures_stays_low_confidence(self, tmp_path):
        config = {"model_type": "some-brand-new-architecture-nobody-has-heard-of"}
        (tmp_path / "config.json").write_text(json.dumps(config))

        result = inspect_model(str(tmp_path), live=True)

        assert result["guessed_kind"] is None
        assert result["confidence"] == "low"


class TestInspectModelLiteLLMHint:
    """A plain model string like "gpt-4o-mini" is never a real HF Hub repo
    id - without this, inspect_model would only ever waste a network call
    and come back with an unhelpful low-confidence guess for it."""

    def test_known_litellm_model_string_classified_without_hf_hub_attempt(self):
        with patch(
            "bias_scope_agent.introspection._download_hf_config",
            side_effect=AssertionError("must not attempt an HF Hub lookup for a litellm model id"),
        ):
            result = inspect_model("gpt-4o-mini", live=True)

        assert result["guessed_source"] == "litellm_model_id"
        assert result["confidence"] == "high"

    def test_works_even_when_live_is_false(self):
        # This is a free, offline lookup against litellm's bundled model
        # list - it should not be gated behind the live-inspection flag.
        result = inspect_model("gpt-4o-mini", live=False)
        assert result["guessed_source"] == "litellm_model_id"
        assert result["confidence"] == "high"

    def test_typo_gets_a_did_you_mean_hint(self):
        result = inspect_model("gpt4o-mini", live=False)
        assert result["guessed_source"] == "hf_hub"
        assert any("gpt-4o-mini" in note for note in result["notes"])

    def test_litellm_not_installed_degrades_gracefully(self):
        with patch.dict(sys.modules, {"litellm": None}):
            result = inspect_model("gpt-4o-mini", live=False)
        # No exact-match hint available without litellm; falls through to
        # the ordinary hf_hub path rather than raising.
        assert result["guessed_source"] == "hf_hub"
        assert isinstance(result, dict)
