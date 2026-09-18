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


class TestRunningFewerMetricsThanWereApproved:
    """RL-059: the gate matched the metric set exactly, which blocked a
    multi-dataset evaluation from ever running.

    Observed live: the agent planned five metrics across three datasets, the
    user confirmed, and then each `run_suite` call - necessarily one per
    dataset, since a prepared handle covers only its own metrics - was refused
    because {CrowSPairs, AUL, AULA} is not equal to the approved set of five.
    The agent diagnosed it correctly and asked for three fresh confirmations,
    which is the right behaviour and also a dead end: the same split recurs
    every time.

    Running a *subset* of an approved plan is not an escalation - the user
    approved strictly more than what runs - so the gate now accepts it. A
    metric that was never approved is still refused, which is the property
    that matters.
    """

    def _confirmed_plan(self, metrics):
        session = AgentSession()
        plan_id = session.record_plan("backend-1", metrics, "gender", "en")
        session.advance_turn()
        session.confirm_plan(plan_id)
        return session

    def test_a_subset_of_the_approved_metrics_is_allowed(self):
        session = self._confirmed_plan(["CrowSPairs", "AUL", "AULA", "WEAT", "SEAT"])
        session.check_run_gate("backend-1", ["CrowSPairs", "AUL", "AULA"], "gender", "en")
        session.check_run_gate("backend-1", ["WEAT"], "gender", "en")
        session.check_run_gate("backend-1", ["SEAT"], "gender", "en")

    def test_the_full_approved_set_is_still_allowed(self):
        session = self._confirmed_plan(["WEAT", "SEAT"])
        session.check_run_gate("backend-1", ["SEAT", "WEAT"], "gender", "en")

    def test_a_metric_that_was_never_approved_is_still_refused(self):
        session = self._confirmed_plan(["WEAT", "SEAT"])
        with pytest.raises(GateError):
            session.check_run_gate("backend-1", ["WEAT", "CrowSPairs"], "gender", "en")

    def test_a_different_backend_is_still_refused(self):
        session = self._confirmed_plan(["WEAT"])
        with pytest.raises(GateError):
            session.check_run_gate("backend-2", ["WEAT"], "gender", "en")

    def test_a_different_axis_is_still_refused(self):
        session = self._confirmed_plan(["WEAT"])
        with pytest.raises(GateError):
            session.check_run_gate("backend-1", ["WEAT"], "race", "en")

    def test_an_unconfirmed_plan_still_blocks_a_subset(self):
        session = AgentSession()
        session.record_plan("backend-1", ["WEAT", "SEAT"], "gender", "en")
        with pytest.raises(GateError):
            session.check_run_gate("backend-1", ["WEAT"], "gender", "en")

    def test_an_empty_metric_set_is_refused_rather_than_trivially_a_subset(self):
        session = self._confirmed_plan(["WEAT"])
        with pytest.raises(GateError):
            session.check_run_gate("backend-1", [], "gender", "en")
