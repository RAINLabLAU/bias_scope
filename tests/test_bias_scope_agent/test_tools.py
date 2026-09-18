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

    def test_needs_data_lists_a_constructor_requirement_the_dataset_does_not_cover(self):
        # BBQ ships its own dataset, so this used to assert BBQMetric was
        # absent from needs_data entirely. Its constructor still requires a
        # model_name with no default, and an agent told "needs nothing" calls
        # run_suite with nothing and has the metric skipped (RL-051).
        session, handle = make_session_with_backend(access=("chat", "completions"))
        result = tools.plan_suite(session, handle, metric_names=["BBQMetric"])
        assert result["needs_data"]["BBQMetric"] == ["__init__.model_name"]

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

    def test_metric_omitted_from_inputs_is_refused_naming_what_it_needs(self):
        # This asserted, until RL-052, that the metric was quietly skipped and
        # a report handle returned. BiasSuite still does exactly that - the
        # change is at the agent boundary, where a handle to an empty report
        # was read back by a live model as a score (see
        # TestRunSuiteRefusesToProduceAnEmptyResult). The skip reason is not
        # lost; it moves into an error the agent can act on.
        session, handle = make_session_with_backend(access=("embeddings",))
        with pytest.raises(ValueError) as excinfo:
            tools.run_suite(session, handle, metric_names=["WEAT"], inputs={})
        assert "WEAT" in str(excinfo.value)
        assert "target_embeddings" in str(excinfo.value)


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

    def test_an_empty_inputs_dict_is_refused_rather_than_returning_an_empty_report(self):
        """The shape check passes; the completeness check is what refuses it.

        This test previously asserted the opposite, on the reasoning that
        "every metric skipping for want of data is a legitimate outcome". It
        is - for BiasSuite. For a tool call it is not: nothing ran, so there is
        no result, and a handle to that reads like one (RL-052).
        """
        session = AgentSession()
        handle = tools.construct_backend(session, kind="litellm", model_id="stub/model")
        with pytest.raises(ValueError, match="missing required inputs"):
            tools.run_suite(session, handle, metric_names=["BOLD"], inputs={})


class TestRunSuiteRefusesToProduceAnEmptyResult:
    """RL-052: a run_suite call that cannot produce a score fails loudly.

    Observed, in the 2026-09-18 live runs: a real agent called run_suite for
    CrowSPairs supplying only `sentence_pairs`. CrowSPairs needs its model at
    construction time, so BiasSuite skipped it, and run_suite returned a handle
    to a report containing nothing but the skip reason. summarize_report then
    renders a "result" whose entire content is an error message.

    Nothing here can stop a model from writing a false sentence, and none of
    these tests claims to. What they remove is the silent-skip opportunity: a
    call that would produce nothing now comes back as a correctable tool error
    (loop.py's _CAUGHT_TOOL_ERRORS hands ValueError to the agent to fix and
    retry in the same turn) rather than as a handle that reads like a result.

    Honest scope note, because the first draft of this docstring got it wrong:
    the run that motivated these guards reported a *correct* score (0.40 on 20
    pairs, confirmed by computing CrowSPairs directly). The skip was real and
    is worth preventing; the fabrication it was first thought to have caused
    was an artifact of the recording script, not of the agent. See
    REVIEW_LATER.md RL-052 for the full sequence.
    """

    def test_missing_constructor_input_is_rejected_before_anything_runs(self):
        session, handle = make_session_with_backend(access=("embeddings", "logits"))
        with pytest.raises(ValueError, match=r"__init__\.model_name"):
            tools.run_suite(
                session,
                handle,
                metric_names=["CrowSPairs"],
                inputs={"CrowSPairs": {"sentence_pairs": [["a b", "c d"]]}},
            )

    def test_the_error_names_the_metric_and_every_missing_parameter(self):
        session, handle = make_session_with_backend(access=("embeddings", "logits"))
        with pytest.raises(ValueError) as excinfo:
            tools.run_suite(
                session, handle, metric_names=["CrowSPairs"], inputs={"CrowSPairs": {}}
            )
        message = str(excinfo.value)
        assert "CrowSPairs" in message
        assert "sentence_pairs" in message
        assert "__init__.model_name" in message

    def test_a_complete_call_is_not_rejected(self):
        session, handle = make_session_with_backend(access=("embeddings",))
        # WEAT needs only evaluate() arrays; supplying them must not raise.
        pairs = [[[1.0, 0.0], [0.9, 0.1]], [[0.0, 1.0], [0.1, 0.9]]]
        tools.run_suite(
            session,
            handle,
            metric_names=["WEAT"],
            inputs={"WEAT": {"target_embeddings": pairs, "attribute_embeddings": pairs}},
        )

    def test_a_report_in_which_nothing_ran_is_an_error_not_a_result(self, monkeypatch):
        """Backstop for whatever the input check cannot foresee.

        A metric can still fail inside BiasSuite for reasons no signature
        inspection predicts. If *every* planned metric was skipped, there is
        no result to summarize, and handing back a handle invites exactly the
        fabrication above.
        """
        session, handle = make_session_with_backend(access=("embeddings",))
        monkeypatch.setattr(tools, "_check_required_inputs", lambda *args, **kwargs: None)
        with pytest.raises(ValueError, match="no metric produced a score"):
            tools.run_suite(
                session, handle, metric_names=["WEAT"], inputs={"WEAT": {"bogus_param": 1}}
            )
