import pytest

from bias_scope_agent.session import AgentSession, GateError


def _plan_and_advance(
    session, backend_handle="backend-1", metrics=("WEAT",), axis="gender", language="en"
):
    plan_id = session.record_plan(backend_handle, metrics, axis, language)
    session.advance_turn()
    return plan_id


class TestCheckRunGate:
    def test_raises_before_any_plan_suite_call(self):
        session = AgentSession()
        with pytest.raises(GateError):
            session.check_run_gate("backend-1", ["WEAT"], "gender", "en")

    def test_raises_if_plan_was_never_confirmed(self):
        session = AgentSession()
        _plan_and_advance(session)
        with pytest.raises(GateError):
            session.check_run_gate("backend-1", ["WEAT"], "gender", "en")

    def test_passes_once_confirmed_after_a_real_turn_boundary(self):
        session = AgentSession()
        plan_id = _plan_and_advance(session)
        session.confirm_plan(plan_id)
        session.check_run_gate("backend-1", ["WEAT"], "gender", "en")  # does not raise

    def test_still_raises_if_metric_set_differs_from_confirmed_plan(self):
        session = AgentSession()
        plan_id = _plan_and_advance(session, metrics=("WEAT",))
        session.confirm_plan(plan_id)
        with pytest.raises(GateError):
            session.check_run_gate("backend-1", ["WEAT", "SEAT"], "gender", "en")

    def test_still_raises_if_backend_handle_differs(self):
        session = AgentSession()
        plan_id = _plan_and_advance(session, backend_handle="backend-1")
        session.confirm_plan(plan_id)
        with pytest.raises(GateError):
            session.check_run_gate("backend-2", ["WEAT"], "gender", "en")

    def test_metric_order_does_not_matter(self):
        session = AgentSession()
        plan_id = session.record_plan("backend-1", ["WEAT", "SEAT"], "gender", "en")
        session.advance_turn()
        session.confirm_plan(plan_id)
        session.check_run_gate("backend-1", ["SEAT", "WEAT"], "gender", "en")  # does not raise


class TestConfirmPlan:
    def test_raises_on_unknown_plan_id(self):
        session = AgentSession()
        with pytest.raises(GateError):
            session.confirm_plan("no-such-plan")

    def test_raises_if_confirmed_in_the_same_turn_it_was_recorded(self):
        session = AgentSession()
        plan_id = session.record_plan("backend-1", ["WEAT"], "gender", "en")
        with pytest.raises(GateError):
            session.confirm_plan(plan_id)  # no advance_turn() in between

    def test_succeeds_after_a_turn_boundary(self):
        session = AgentSession()
        plan_id = _plan_and_advance(session)
        session.confirm_plan(plan_id)  # does not raise


class TestSessionFacts:
    def test_facts_start_empty_and_are_a_plain_dict(self):
        session = AgentSession()
        assert session.facts == {}
        session.facts["model_kind"] = "causal"
        assert session.facts["model_kind"] == "causal"


class TestSessionRegistries:
    def test_backends_and_reports_are_independent_handle_registries(self):
        session = AgentSession()
        backend_handle = session.backends.register("fake-backend")
        report_handle = session.reports.register("fake-report")
        assert session.backends.get(backend_handle) == "fake-backend"
        assert session.reports.get(report_handle) == "fake-report"
