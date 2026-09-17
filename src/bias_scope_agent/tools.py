"""Thin tool wrappers the agent LLM calls. No metric-selection logic lives
here - that stays in bias_scope. Every function takes `session` explicitly
as its first argument so it is directly unit-testable; the agent loop binds
`session` via functools.partial when building its dispatch table, and the
LLM-facing tool schema never mentions it.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from bias_scope.backends import HuggingFaceBackend, LiteLLMBackend
from bias_scope.recommend import explain_exclusions, recommend_metrics
from bias_scope.report import FIDELITY_BADGE, Report, to_html, to_markdown
from bias_scope.suite import BiasSuite
from bias_scope_agent.introspection import metrics_needing_data
from bias_scope_agent.session import AgentSession

_HUGGINGFACE_KINDS = ("causal", "encoder")


def construct_backend(
    session: AgentSession,
    kind: str,
    model_id: str,
    backend_kind: Optional[str] = None,
    dtype: str = "bf16",
    device: Optional[str] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
) -> str:
    """Build a backend for the user's target model. Returns a backend_handle."""
    if kind == "huggingface":
        if backend_kind not in _HUGGINGFACE_KINDS:
            raise ValueError(
                "backend_kind must be 'causal' or 'encoder' when kind='huggingface', "
                f"got {backend_kind!r}"
            )
        backend = HuggingFaceBackend(model_id, kind=backend_kind, dtype=dtype, device=device)
    elif kind == "litellm":
        backend = LiteLLMBackend(model_id, api_key=api_key, api_base=api_base)
    else:
        raise ValueError(f"kind must be 'huggingface' or 'litellm', got {kind!r}")
    return session.backends.register(backend)


def recommend_metrics_tool(
    session: AgentSession,
    backend_handle: str,
    axis: Optional[str] = None,
    language: str = "en",
    family: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Metrics legal for this backend's access. Access always comes from the
    looked-up Backend, never from an argument the agent could get wrong."""
    backend = session.backends.get(backend_handle)
    recommendations = recommend_metrics(backend.access, axis=axis, language=language, family=family)
    return [
        {
            "metric": rec.metric,
            "family": rec.info.family,
            "fidelity": rec.fidelity,
            "access": list(rec.info.access),
            "reason": rec.reason,
        }
        for rec in recommendations
    ]


def explain_exclusions_tool(
    session: AgentSession, backend_handle: str, language: str = "en"
) -> Dict[str, str]:
    """Direct passthrough to bias_scope.explain_exclusions."""
    backend = session.backends.get(backend_handle)
    return explain_exclusions(backend.access, language=language)


def plan_suite(
    session: AgentSession,
    backend_handle: str,
    metric_names: Sequence[str],
    axis: str = "gender",
    language: str = "en",
) -> Dict[str, Any]:
    """Dry-run plan for the requested metrics, plus which of them will need
    caller-supplied data before run_suite can actually run them. Never calls
    .run() - this cannot use run()'s own NEEDS_DATA skip semantics, since
    that would require executing every metric that already has data."""
    backend = session.backends.get(backend_handle)
    suite = BiasSuite(backend, axis=axis, language=language, metrics=metric_names)
    plan = suite.plan()

    needs = metrics_needing_data([name for name, _ in plan])
    needs_data = {name: params for name, params in needs.items() if params}

    plan_id = session.record_plan(backend_handle, [name for name, _ in plan], axis, language)
    return {
        "plan_id": plan_id,
        "plan": [
            {
                "metric": name,
                "family": info.family,
                "fidelity": info.fidelity,
                "access": list(info.access),
            }
            for name, info in plan
        ],
        "needs_data": needs_data,
    }


def request_missing_inputs(session: AgentSession, metric_names: Sequence[str]) -> Dict[str, Any]:
    """Agent-side only - no bias_scope import, no fabricated data. Just
    surfaces a structured message and lets the agent's turn end so the user
    can reply with the real data."""
    return {"awaiting_input_for": list(metric_names)}


def confirm_plan(session: AgentSession, plan_id: str) -> Dict[str, Any]:
    """Marks a plan confirmed. Raises GateError (via session) if called too
    early, e.g. in the same turn the plan was produced."""
    session.confirm_plan(plan_id)
    return {"confirmed": True, "plan_id": plan_id}


def _check_inputs_shape(metric_names: Sequence[str], inputs: Dict[str, Any]) -> None:
    """`inputs` is keyed by metric name, one entry per metric.

    A flat dict of parameters is the natural wrong guess, and it reaches
    `BiasSuite.run()` as "no data for this metric": the metric is skipped with
    a reason that reads like the user's omission rather than a malformed call.
    A live run made exactly this mistake (RL-049), so reject it here, where the
    agent gets a ValueError back as a tool error it can correct.
    """
    unexpected = [key for key in inputs if key not in set(metric_names)]
    if not unexpected:
        return
    example = metric_names[0] if metric_names else "<MetricName>"
    raise ValueError(
        f"inputs must be keyed by metric name: got top-level key(s) "
        f"{unexpected}, which are not in metric_names {list(metric_names)}. "
        f"Put each metric's parameters under its own name, with any "
        f'constructor arguments under "__init__", e.g. '
        f'{{"{example}": {{"__init__": {{"model_name": "..."}}, "<param>": ...}}}}.'
    )


def run_suite(
    session: AgentSession,
    backend_handle: str,
    metric_names: Sequence[str],
    inputs: Dict[str, Dict[str, Any]],
    axis: str = "gender",
    language: str = "en",
    seed: int = 42,
) -> str:
    """Runs the suite and returns a report_handle. No gate logic here - the
    confirm-before-run gate is enforced by the tool dispatcher (loop.py),
    not this function, so this stays directly unit-testable."""
    _check_inputs_shape(metric_names, inputs)
    backend = session.backends.get(backend_handle)
    suite = BiasSuite(backend, axis=axis, language=language, metrics=metric_names)
    report = suite.run(seed=seed, inputs=inputs)
    return session.reports.register(report)


def _chat_summary(report: Report) -> str:
    lines = [f"Bias report for {report.model_id}"]
    for family, results in report.by_family().items():
        lines.append(f"\n{family}:")
        for result in results:
            badge = FIDELITY_BADGE[result.info.fidelity]
            lines.append(f"  [{badge}] {result.metric}: {result.score:.4g}")
    if report.skipped:
        lines.append("\nskipped:")
        for name, reason in report.skipped.items():
            lines.append(f"  {name}: {reason}")
    return "\n".join(lines)


def summarize_report(session: AgentSession, report_handle: str, format: str = "chat") -> str:
    """"chat" is built from by_family()/fidelity_counts()/skipped, with every
    score shown next to its fidelity badge. "markdown"/"html" are passthrough
    to bias_scope's own renderers."""
    report = session.reports.get(report_handle)
    if format == "chat":
        return _chat_summary(report)
    if format == "markdown":
        return to_markdown(report)
    if format == "html":
        return to_html(report)
    raise ValueError(f"format must be 'chat', 'markdown' or 'html', got {format!r}")


def record_fact(session: AgentSession, key: str, value: str) -> Dict[str, str]:
    """Agent-side session memory: once a clarification is answered, store it
    so render_system_prompt can re-inject it and it is never re-asked."""
    session.facts[key] = value
    return {key: value}
