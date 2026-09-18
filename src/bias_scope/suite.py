"""`BiasSuite` — run many metrics against one model (PLAN.md 5.4).

This is the "unified interface" the v0.1 reviewers found missing. It selects the
metrics a backend can actually run, executes each one, and returns a `Report`.

Two design commitments, both deliberate:

- **No composite score.** `Report` has no aggregate field and never will;
  PLAN.md lists one as a non-goal. Metrics with different units, ranges and
  neutral values do not average into anything meaningful.
- **A metric that fails is recorded as skipped, never as a zero.** A missing
  number is information; a fabricated one is a defect.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

from bias_scope.backends import Backend
from bias_scope.base import BiasScopeError
from bias_scope.metadata import list_metrics
from bias_scope.recommend import recommend_metrics
from bias_scope.report import Report
from bias_scope.result import make_protocol

#: Inputs a metric needs that the suite cannot invent. A metric is skipped with
#: this reason rather than run on made-up data.
NEEDS_DATA = (
    "requires caller-supplied stimuli (word lists, templates or a dataset); "
    "pass them via `inputs={metric: {...}}`"
)


def _missing_metric_message(name: str) -> str:
    """Tell "no such metric" apart from "metric exists, extra not installed".

    A metric whose optional dependency is absent never reaches `register()`,
    so it is missing from `list_metrics()` for a reason a user can fix. Calling
    that "unknown metric" sends them looking for a typo instead (RL-047).
    """
    from bias_scope.prompts_based import PROMPT_METRIC_NAMES

    if name in PROMPT_METRIC_NAMES:
        return (
            f"{name} is a known metric, but its optional dependencies are not "
            f"installed, so it is not registered. Install them with: "
            f'pip install "bias-scope[datasets]", "bias-scope[llm]" or '
            f'"bias-scope[all]".'
        )
    return f"unknown metric {name!r}"


class BiasSuite:
    """
    Run every applicable metric against one model.

    Example
    -------
    >>> from bias_scope.backends import StubBackend
    >>> suite = BiasSuite(StubBackend(access=("chat",)))
    >>> plan = suite.plan()
    >>> all(set(i.access) <= {"chat"} for _, i in plan)
    True
    """

    def __init__(
        self,
        backend: Backend,
        axis: str = "gender",
        language: str = "en",
        *,
        include_mismatch: bool = False,
        include_unaudited: bool = True,
        metrics: Optional[Sequence[str]] = None,
    ):
        """
        Args:
            backend (Backend): The model. Its `access` decides what can run.
            axis (str): Bias axis, recorded in the protocol.
            language (str): Language code; metrics without resources for it are
                not selected.
            include_mismatch (bool): Run metrics known to implement a different
                statistic than they cite. Default False.
            include_unaudited (bool): Run metrics whose sources have not been
                read. Default True; their status travels with every result.
            metrics (Sequence[str], optional): Run exactly these, skipping
                selection. Still checked against the backend's access, because
                asking for a metric the model cannot support is an error, not a
                preference.
        """
        self.backend = backend
        self.axis = axis
        self.language = language
        self.include_mismatch = include_mismatch
        self.include_unaudited = include_unaudited
        self.requested = list(metrics) if metrics else None

    def plan(self) -> List[tuple]:
        """
        The (name, MetricInfo) pairs this suite would run, without running them.

        Cheap, and worth calling first: it shows exactly what a given backend
        can and cannot support.
        """
        if self.requested is None:
            return [
                (r.metric, r.info)
                for r in recommend_metrics(
                    self.backend.access,
                    axis=self.axis,
                    language=self.language,
                    include_mismatch=self.include_mismatch,
                    include_unaudited=self.include_unaudited,
                )
            ]

        registry = list_metrics()
        selected = []
        for name in self.requested:
            info = registry.get(name)
            if info is None:
                raise ValueError(_missing_metric_message(name))
            if not self.backend.supports(info.access):
                missing = sorted(set(info.access) - set(self.backend.access))
                raise ValueError(
                    f"{name} needs {'+'.join(missing)}, which "
                    f"{self.backend.model_id} does not provide "
                    f"(it has {'+'.join(self.backend.access)})"
                )
            selected.append((name, info))
        return selected

    def run(
        self,
        seed: int = 42,
        inputs: Optional[Dict[str, Dict[str, Any]]] = None,
        *,
        on_error: str = "skip",
        decoding: Optional[Dict[str, Any]] = None,
    ) -> Report:
        """
        Run the planned metrics and collect a `Report`.

        Args:
            seed (int): Applied before each metric and recorded in the protocol.
            inputs (dict, optional): Per-metric keyword arguments, e.g.
                ``{"WEAT": {"target_embeddings": ..., "attribute_embeddings": ...}}``.
                A metric with no entry is **skipped**, not run on invented data.
            on_error (str): "skip" (default) records the failure and continues;
                "raise" propagates. Never substitutes a value.
            decoding (dict, optional): Recorded in the protocol.

        Returns:
            Report
        """
        if on_error not in ("skip", "raise"):
            raise ValueError(f"on_error must be 'skip' or 'raise', got {on_error!r}")

        inputs = inputs or {}
        registry = _metric_classes()
        results = []
        skipped: Dict[str, str] = {}

        for name, info in self.plan():
            cls = registry.get(name)
            if cls is None:
                skipped[name] = "class not importable in this environment"
                continue

            supplied = inputs.get(name)
            if supplied is None:
                skipped[name] = NEEDS_DATA
                continue

            # A shallow copy, because the pop below would otherwise strip
            # "__init__" out of the caller's own dict: running the same inputs
            # twice then constructs the metric differently the second time, and
            # a caller recording `inputs` finds its record altered after the
            # fact (REVIEW_LATER RL-054). The values are not copied - they may
            # be large arrays and are not modified here.
            kwargs = dict(supplied)

            try:
                metric = cls(**kwargs.pop("__init__", {}))
                results.append(
                    metric.run(
                        seed=seed,
                        protocol_kwargs={
                            **self.backend.protocol_fields(),
                            "decoding": decoding or {},
                        },
                        **kwargs,
                    )
                )
            except (BiasScopeError, ValueError, TypeError, ImportError) as exc:
                if on_error == "raise":
                    raise
                skipped[name] = f"{type(exc).__name__}: {exc}"

        return Report(
            model_id=self.backend.model_id,
            results=results,
            protocol=make_protocol(
                metric="BiasSuite",
                seed=seed,
                decoding=decoding or {},
                **self.backend.protocol_fields(),
            )
            | {"axis": self.axis, "language": self.language},
            skipped=skipped,
        )


def _metric_classes() -> Dict[str, Callable]:
    """Every importable metric class, by name."""
    import importlib

    from bias_scope.base import BiasMetric

    classes: Dict[str, Callable] = {}
    for module_name in (
        "bias_scope.embeddings_based",
        "bias_scope.probability_based",
        "bias_scope.generated_text_based",
        "bias_scope.prompts_based",
    ):
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        for name in list_metrics():
            candidate = getattr(module, name, None)
            if isinstance(candidate, type) and issubclass(candidate, BiasMetric):
                classes[name] = candidate
    return classes


__all__ = ["BiasSuite", "NEEDS_DATA"]
