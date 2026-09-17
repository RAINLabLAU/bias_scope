import numpy as np
import pytest

from bias_scope.backends import StubBackend
from bias_scope_agent import tools
from bias_scope_agent.session import AgentSession

# WEAT accepts raw embedding arrays directly (no live embedding model needed),
# per its own evaluate() docstring example - the only offline-friendly metric
# among the ones exercised here.
_WEAT_INPUTS = {
    "target_embeddings": (
        np.array([[1.0, 0.0], [0.9, 0.1], [0.8, 0.2]]),
        np.array([[0.0, 1.0], [0.1, 0.9], [0.2, 0.8]]),
    ),
    "attribute_embeddings": (
        np.array([[1.0, 0.0], [0.95, 0.05], [0.9, 0.1]]),
        np.array([[0.0, 1.0], [0.05, 0.95], [0.1, 0.9]]),
    ),
}


def make_session_with_backend(access=("embeddings",), model_id="stub/model"):
    session = AgentSession()
    backend = StubBackend(access=access, model_id=model_id)
    handle = session.backends.register(backend)
    return session, handle


class TestConstructBackend:
    def test_huggingface_requires_backend_kind(self):
        session = AgentSession()
        with pytest.raises(ValueError, match="backend_kind"):
            tools.construct_backend(session, kind="huggingface", model_id="sshleifer/tiny-gpt2")

    def test_huggingface_rejects_bad_backend_kind(self):
        session = AgentSession()
        with pytest.raises(ValueError, match="backend_kind"):
            tools.construct_backend(
                session, kind="huggingface", model_id="sshleifer/tiny-gpt2", backend_kind="bogus"
            )

    def test_huggingface_returns_a_resolvable_handle(self):
        session = AgentSession()
        handle = tools.construct_backend(
            session, kind="huggingface", model_id="sshleifer/tiny-gpt2", backend_kind="causal"
        )
        backend = session.backends.get(handle)
        assert backend.access == ("embeddings", "logits", "completions")

    def test_litellm_does_not_require_backend_kind(self):
        session = AgentSession()
        handle = tools.construct_backend(session, kind="litellm", model_id="gpt-4o-mini")
        backend = session.backends.get(handle)
        assert backend.access == ("completions", "chat")

    def test_unknown_kind_raises_value_error(self):
        session = AgentSession()
        with pytest.raises(ValueError):
            tools.construct_backend(session, kind="bogus", model_id="x")


class TestRecommendMetricsTool:
    def test_uses_backend_access_never_a_typed_argument(self):
        session, handle = make_session_with_backend(access=("embeddings",))
        result = tools.recommend_metrics_tool(session, handle)
        assert result
        for row in result:
            assert set(row.keys()) == {"metric", "family", "fidelity", "access", "reason"}
            assert set(row["access"]) <= {"embeddings"}

    def test_unknown_backend_handle_raises_key_error(self):
        session = AgentSession()
        with pytest.raises(KeyError):
            tools.recommend_metrics_tool(session, "no-such-handle")


class TestExplainExclusionsTool:
    def test_is_a_direct_passthrough(self):
        session, handle = make_session_with_backend(access=("embeddings",))
        result = tools.explain_exclusions_tool(session, handle)
        assert isinstance(result, dict)
        assert result  # some metrics are excluded when access is embeddings-only


class TestPlanSuite:
    def test_reports_plan_and_plan_id(self):
        session, handle = make_session_with_backend(access=("embeddings",))
        result = tools.plan_suite(session, handle, metric_names=["WEAT"])
        assert result["plan_id"]
        assert result["plan"] == [
            {
                "metric": "WEAT",
                "family": "embedding",
                "fidelity": "faithful",
                "access": ["embeddings"],
            }
        ]

    def test_needs_data_only_lists_metrics_missing_required_params(self):
        session, handle = make_session_with_backend(access=("chat", "completions"))
        result = tools.plan_suite(session, handle, metric_names=["BBQMetric"])
        assert "BBQMetric" not in result["needs_data"]

    def test_needs_data_reports_weat_requirements(self):
        session, handle = make_session_with_backend(access=("embeddings",))
        result = tools.plan_suite(session, handle, metric_names=["WEAT"])
        assert set(result["needs_data"]["WEAT"]) == {"target_embeddings", "attribute_embeddings"}

    def test_unknown_metric_name_raises_value_error(self):
        session, handle = make_session_with_backend(access=("embeddings",))
        with pytest.raises(ValueError):
            tools.plan_suite(session, handle, metric_names=["NotAMetric"])

    def test_records_a_plan_on_the_session(self):
        session, handle = make_session_with_backend(access=("embeddings",))
        result = tools.plan_suite(session, handle, metric_names=["WEAT"])
        with pytest.raises(Exception):  # GateError: same turn, not yet advanced
            session.confirm_plan(result["plan_id"])


class TestRequestMissingInputs:
    def test_returns_structured_message_and_touches_no_library_state(self):
        session = AgentSession()
        result = tools.request_missing_inputs(session, ["WEAT"])
        assert result == {"awaiting_input_for": ["WEAT"]}


class TestConfirmPlan:
    def test_delegates_to_session_confirm_plan(self):
        session, handle = make_session_with_backend(access=("embeddings",))
        plan = tools.plan_suite(session, handle, metric_names=["WEAT"])
        session.advance_turn()
        result = tools.confirm_plan(session, plan["plan_id"])
        assert result == {"confirmed": True, "plan_id": plan["plan_id"]}


class TestRunSuite:
    def test_has_no_gate_logic_and_can_be_called_directly(self):
        session, handle = make_session_with_backend(access=("embeddings",))
        report_handle = tools.run_suite(
            session, handle, metric_names=["WEAT"], inputs={"WEAT": dict(_WEAT_INPUTS)}
        )
        report = session.reports.get(report_handle)
        assert report.model_id == "stub/model"
        assert report.scores()["WEAT"]

    def test_metric_omitted_from_inputs_is_skipped_with_needs_data(self):
        session, handle = make_session_with_backend(access=("embeddings",))
        report_handle = tools.run_suite(session, handle, metric_names=["WEAT"], inputs={})
        report = session.reports.get(report_handle)
        assert "WEAT" in report.skipped


class TestSummarizeReport:
    def _run_weat(self, session, handle):
        return tools.run_suite(
            session, handle, metric_names=["WEAT"], inputs={"WEAT": dict(_WEAT_INPUTS)}
        )

    def test_chat_format_includes_fidelity_badge_for_every_score(self):
        session, handle = make_session_with_backend(access=("embeddings",))
        report_handle = self._run_weat(session, handle)
        summary = tools.summarize_report(session, report_handle, format="chat")
        assert "WEAT" in summary
        # WEAT's fidelity badge (per bias_scope's own FIDELITY_BADGE table)
        from bias_scope.metadata import list_metrics
        from bias_scope.report import FIDELITY_BADGE

        badge = FIDELITY_BADGE[list_metrics()["WEAT"].fidelity]
        assert badge in summary

    def test_markdown_format_is_passthrough(self):
        session, handle = make_session_with_backend(access=("embeddings",))
        report_handle = self._run_weat(session, handle)
        summary = tools.summarize_report(session, report_handle, format="markdown")
        assert "WEAT" in summary


class TestRecordFact:
    def test_stores_fact_on_session(self):
        session = AgentSession()
        result = tools.record_fact(session, "model_kind", "causal")
        assert result == {"model_kind": "causal"}
        assert session.facts["model_kind"] == "causal"


class TestRunSuiteInputsShape:
    """`inputs` is keyed by metric name; a flat dict of parameters is the
    natural wrong guess, and `BiasSuite.run()` answers it by skipping the
    metric with a reason that reads like the user forgot to supply data.
    A live gemma4:12b run made exactly this mistake and reported a skip as
    the result (REVIEW_LATER RL-049). Rejecting it gives the agent an error
    it can act on instead.
    """

    def test_a_flat_inputs_dict_is_rejected_not_silently_skipped(self):
        session = AgentSession()
        handle = tools.construct_backend(
            session, kind="litellm", model_id="stub/model"
        )
        with pytest.raises(ValueError, match="keyed by metric name"):
            tools.run_suite(
                session,
                handle,
                metric_names=["CrowSPairs"],
                inputs={"sentence_pairs": [["he", "she"]]},
            )

    def test_the_error_names_the_offending_key_and_the_expected_one(self):
        session = AgentSession()
        handle = tools.construct_backend(
            session, kind="litellm", model_id="stub/model"
        )
        with pytest.raises(ValueError) as excinfo:
            tools.run_suite(
                session, handle, metric_names=["BOLD"], inputs={"prompts": []}
            )
        message = str(excinfo.value)
        assert "prompts" in message
        assert "BOLD" in message

    def test_an_empty_inputs_dict_is_still_allowed(self):
        """Every metric skipping for want of data is a legitimate outcome."""
        session = AgentSession()
        handle = tools.construct_backend(
            session, kind="litellm", model_id="stub/model"
        )
        assert tools.run_suite(session, handle, metric_names=["BOLD"], inputs={})
