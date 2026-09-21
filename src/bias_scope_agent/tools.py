"""Thin tool wrappers the agent LLM calls. No metric-selection logic lives
here - that stays in bias_scope. Every function takes `session` explicitly
as its first argument so it is directly unit-testable; the agent loop binds
`session` via functools.partial when building its dispatch table, and the
LLM-facing tool schema never mentions it.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence

from bias_scope.backends import HuggingFaceBackend, LiteLLMBackend
from bias_scope.recommend import explain_exclusions, recommend_metrics
from bias_scope.report import FIDELITY_BADGE, Report, to_html, to_markdown
from bias_scope.suite import BiasSuite
from bias_scope_agent.datasets import available_datasets, build_inputs
from bias_scope_agent.introspection import _UNIMPORTABLE, metrics_needing_data
from bias_scope_agent.session import AgentSession

_HUGGINGFACE_KINDS = ("causal", "encoder")

# How metrics_needing_data spells a constructor argument, and where it goes in
# run_suite's `inputs`.
_INIT_PREFIX = "__init__."


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


def _check_required_inputs(metric_names: Sequence[str], inputs: Dict[str, Any]) -> None:
    """Every parameter `plan_suite` said was needed must actually be present.

    `BiasSuite.run` skips a metric it cannot construct or call, and records the
    reason; it does not fail. That is right for a library - one bad metric
    should not lose the other nine - but through the agent it means a call
    missing one argument comes back as a handle to an empty report, which reads
    like a result. In the 2026-09-18 live run it was then reported to the user
    as a score that did not exist (REVIEW_LATER.md RL-052).

    Raising here instead puts the failure where the agent can fix it: loop.py
    returns a ValueError to the model as a correctable tool error, naming the
    parameter, in the same turn.

    Note this checks only what a signature can show. A metric that accepts an
    alternative to a named parameter (CrowSPairs takes either `model_name` or a
    ready-made scorer object) would be over-strict here - except that every
    such alternative in this library is a Python callable, which cannot cross a
    JSON tool-call boundary anyway.
    """
    needed = metrics_needing_data(metric_names)
    missing: Dict[str, List[str]] = {}
    for name in metric_names:
        required = needed.get(name, [])
        if not required or _UNIMPORTABLE in required:
            continue
        provided = inputs.get(name) or {}
        init_block = provided.get("__init__") or {}
        absent = []
        for param in required:
            if param.startswith(_INIT_PREFIX):
                if param[len(_INIT_PREFIX) :] not in init_block:
                    absent.append(param)
            elif param not in provided:
                absent.append(param)
        if absent:
            missing[name] = absent
    if not missing:
        return
    detail = "; ".join(f"{name} is missing {params}" for name, params in sorted(missing.items()))
    raise ValueError(
        f"run_suite is missing required inputs: {detail}. These are exactly the "
        f"parameters plan_suite listed under needs_data. A name written "
        f'"__init__.<param>" goes under that metric\'s "__init__" key, e.g. '
        f'{{"CrowSPairs": {{"__init__": {{"model_name": "bert-base-uncased"}}, '
        f'"sentence_pairs": [...]}}}}. Supply them and call run_suite again.'
    )


def list_datasets(session: AgentSession, metric_names: Optional[Sequence[str]] = None) -> Any:
    """Datasets the harness can load itself, and which metrics each one feeds."""
    del session  # stateless; takes session only to match the dispatch signature
    return available_datasets(metric_names)


def prepare_inputs(
    session: AgentSession,
    backend_handle: str,
    dataset: str,
    metric_names: Sequence[str],
    axis: str = "gender",
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Load a named dataset server-side and return a handle plus provenance.

    The data itself is deliberately not returned. That is the whole mechanism:
    if evaluation data never enters the transcript, the model cannot paraphrase,
    truncate or "correct" it, which two live runs showed it otherwise does
    (REVIEW_LATER.md RL-053).
    """
    backend = session.backends.get(backend_handle)
    inputs, provenance = build_inputs(backend, dataset, metric_names, axis=axis, limit=limit)
    return {"inputs_handle": session.inputs.register(inputs), "provenance": provenance}


def _resolve_input_handles(session: AgentSession, handles: Sequence[str]) -> Dict[str, Any]:
    """Merge several prepared-input handles into one inputs object.

    A metric set spanning several datasets yields one handle per dataset, and
    running them as separate calls would produce separate reports and separate
    partial summaries (REVIEW_LATER RL-059). Two handles claiming the same
    metric are refused rather than silently last-wins: they would disagree
    about what that metric scored.
    """
    merged: Dict[str, Any] = {}
    for handle in handles:
        block = deepcopy(session.inputs.get(handle))
        clash = sorted(set(block) & set(merged))
        if clash:
            raise ValueError(
                f"two prepared handles both supply {clash}; they would disagree about "
                f"what was scored. Prepare each metric's data once."
            )
        merged |= block
    return merged


def run_suite(
    session: AgentSession,
    backend_handle: str,
    metric_names: Sequence[str],
    inputs: Optional[Dict[str, Dict[str, Any]]] = None,
    inputs_handle: Optional[str] = None,
    inputs_handles: Optional[Sequence[str]] = None,
    axis: str = "gender",
    language: str = "en",
    seed: int = 42,
) -> str:
    """Runs the suite and returns a report_handle. No gate logic here - the
    confirm-before-run gate is enforced by the tool dispatcher (loop.py),
    not this function, so this stays directly unit-testable.

    Data arrives either by value (`inputs`) or by reference (`inputs_handle`
    from prepare_inputs). Prefer the handle: see datasets.py.
    """
    handles = list(inputs_handles or [])
    if inputs_handle is not None:
        handles.append(inputs_handle)
    if inputs is not None and handles:
        raise ValueError(
            "pass inputs or prepared handles, not both - they would disagree about "
            "what was scored. Prefer the handles, which cannot be corrupted in transit."
        )
    if handles:
        inputs = _resolve_input_handles(session, handles)
    if inputs is None:
        raise ValueError(
            "run_suite needs data: either inputs_handles (from prepare_inputs, "
            "preferred - pass every handle in one call so the evaluation is one "
            "report) or an inputs object keyed by metric name."
        )
    _check_inputs_shape(metric_names, inputs)
    _check_required_inputs(metric_names, inputs)
    backend = session.backends.get(backend_handle)
    suite = BiasSuite(backend, axis=axis, language=language, metrics=metric_names)
    report = suite.run(seed=seed, inputs=inputs)
    if not report.results:
        raise ValueError(
            f"no metric produced a score, so there is nothing to report. "
            f"Skipped: {report.skipped}. Do not describe this as a result - "
            f"fix the cause and call run_suite again, or tell the user it "
            f"could not be run."
        )
    return session.reports.register(report)


def _chat_summary(report: Report) -> str:
    lines = [f"Bias report for {report.model_id}"]
    for family, results in report.by_family().items():
        lines.append(f"\n{family}:")
        for result in results:
            badge = FIDELITY_BADGE[result.info.fidelity]
            # n is not decoration: a score computed on 19 items when 20 were
            # asked for reads identically without it, and that has happened.
            lines.append(f"  [{badge}] {result.metric}: {result.score:.4g} (n={result.n})")
            # A provider that substituted a resource says so in the protocol;
            # the badge above is the class's static fidelity and cannot.
            for resource in result.protocol.get("resources", []):
                if resource.get("deviation"):
                    lines.append(f"      deviation: {resource['deviation']}")
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
