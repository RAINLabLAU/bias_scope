"""Per-conversation state, including the application-level confirm-before-run gate.

`run_suite` must never execute before the user has seen a `plan_suite`
result and explicitly confirmed it, across a real turn boundary. That
guarantee is enforced here, structurally, rather than only prompted: the
tool dispatcher (see loop.py) calls `check_run_gate` before invoking the
real `run_suite` function, and refuses to invoke it on a GateError.

Scope note: this gate is structural (a plan was shown, a turn boundary
passed, confirm_plan was explicitly called) - it cannot itself judge whether
the user's reply was actually affirmative. That judgment stays with the
single agent LLM, per this project's "no second reviewer LLM" design (see
REVIEW_LATER.md RL-040).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple
from uuid import uuid4

from bias_scope.backends import Backend
from bias_scope.report import Report
from bias_scope_agent.registry import HandleRegistry


class BiasScopeAgentError(RuntimeError):
    """Base class for bias_scope_agent-specific errors."""


class GateError(BiasScopeAgentError):
    """Raised when a tool call is attempted before its required gate is satisfied."""


@dataclass
class PlanRecord:
    plan_id: str
    backend_handle: str
    metric_names: Tuple[str, ...]
    axis: str
    language: str
    turn: int
    confirmed: bool = False
    confirmed_turn: Optional[int] = None


class AgentSession:
    """State for one conversation: handle registries, facts, and plan gating."""

    def __init__(self) -> None:
        self.backends: HandleRegistry[Backend] = HandleRegistry()
        self.reports: HandleRegistry[Report] = HandleRegistry()
        self.facts: Dict[str, str] = {}
        self.turn: int = 0
        self._plans: Dict[str, PlanRecord] = {}

    def advance_turn(self) -> None:
        """Call exactly once per new human message."""
        self.turn += 1

    def record_plan(
        self, backend_handle: str, metric_names: Sequence[str], axis: str, language: str
    ) -> str:
        plan_id = uuid4().hex
        self._plans[plan_id] = PlanRecord(
            plan_id=plan_id,
            backend_handle=backend_handle,
            metric_names=tuple(sorted(metric_names)),
            axis=axis,
            language=language,
            turn=self.turn,
        )
        return plan_id

    def confirm_plan(self, plan_id: str) -> None:
        record = self._plans.get(plan_id)
        if record is None:
            raise GateError(f"unknown plan_id {plan_id!r}; call plan_suite first")
        if self.turn <= record.turn:
            raise GateError(
                "cannot confirm a plan in the same turn it was produced - "
                "show the plan to the user and wait for their next message"
            )
        record.confirmed = True
        record.confirmed_turn = self.turn

    def check_run_gate(
        self, backend_handle: str, metric_names: Sequence[str], axis: str, language: str
    ) -> None:
        key = (backend_handle, tuple(sorted(metric_names)), axis, language)
        for record in self._plans.values():
            record_key = (record.backend_handle, record.metric_names, record.axis, record.language)
            if record_key == key and record.confirmed:
                return
        raise GateError(
            "run_suite is blocked: no confirmed plan matches this backend, metric set, "
            "axis and language. Call plan_suite, show its result to the user, wait for "
            "their reply, then call confirm_plan."
        )
