"""Reports, comparison and cross-metric analysis (PLAN.md 5.4).

Markdown and HTML are built with `string.Template` and f-strings; no templating
library, per Section 1's preference for the standard library.

Every rendered value carries its confidence interval and its fidelity status.
That is not decoration: a number from a `mismatch` metric and a number from a
`faithful` one mean different things, and a report that prints them identically
invites the reader to average them.
"""

from __future__ import annotations

import html
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from bias_scope.result import BiasResult

#: Rendered next to every score. Deliberately blunt.
FIDELITY_BADGE = {
    "faithful": "faithful",
    "adaptation": "ADAPTATION",
    "original": "ORIGINAL",
    "mismatch": "MISMATCH",
    "unaudited": "UNAUDITED",
}

FAMILY_ORDER = ("embedding", "probability", "generated_text", "prompt")


@dataclass
class Report:
    """One model's results across many metrics."""

    model_id: str
    results: List[BiasResult]
    protocol: Dict[str, Any] = field(default_factory=dict)
    skipped: Dict[str, str] = field(default_factory=dict)

    def by_family(self) -> Dict[str, List[BiasResult]]:
        """Results grouped by metric family, in the plan's order."""
        grouped: Dict[str, List[BiasResult]] = {f: [] for f in FAMILY_ORDER}
        for result in self.results:
            grouped.setdefault(result.info.family, []).append(result)
        return {k: v for k, v in grouped.items() if v}

    def fidelity_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for result in self.results:
            counts[result.info.fidelity] = counts.get(result.info.fidelity, 0) + 1
        return counts

    def scores(self) -> Dict[str, float]:
        """metric name -> score, for `correlate`."""
        return {r.metric: r.score for r in self.results}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "protocol": self.protocol,
            "skipped": self.skipped,
            "results": [r.to_dict() for r in self.results],
        }

    def __len__(self) -> int:
        return len(self.results)


def save_json(report: Report, path: Path | str) -> Path:
    """Write a report as JSON. Round-trips through `bias_scope.result.from_dict`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2, default=str) + "\n")
    return path


def _interval(result: BiasResult) -> str:
    if result.ci is None:
        return "—"
    return f"[{result.ci[0]:.4g}, {result.ci[1]:.4g}]"


def to_markdown(report: Report) -> str:
    """Render a report as Markdown, one section per family."""
    lines = [
        f"# Bias report — `{report.model_id}`",
        "",
        "Every score carries its interval and its fidelity status. **Scores from "
        "different metrics are not comparable and must not be averaged** — the "
        "library computes no composite score by design.",
        "",
    ]

    counts = report.fidelity_counts()
    if counts:
        lines += [
            "## Fidelity of the metrics run",
            "",
            "| Status | Count |",
            "|---|---|",
            *[f"| {k} | {v} |" for k, v in sorted(counts.items())],
            "",
        ]
        if counts.get("mismatch"):
            lines += [
                f"> **{counts['mismatch']} metric(s) in this report are known "
                "MISMATCHES** — they do not implement the metric they cite. "
                "Their numbers should not be reported as that paper's metric.",
                "",
            ]

    for family, results in report.by_family().items():
        lines += [
            f"## {family}",
            "",
            "| Metric | Score | 95% CI | n | Normalised | Fidelity |",
            "|---|---:|---|---:|---:|---|",
        ]
        for r in sorted(results, key=lambda x: x.metric):
            lines.append(
                f"| {r.metric} | {r.score:.4g} | {_interval(r)} | {r.n} "
                f"| {r.normalized_deviation():+.3f} | {FIDELITY_BADGE[r.info.fidelity]} |"
            )
        lines.append("")

    if report.skipped:
        lines += ["## Not run", "", "| Metric | Reason |", "|---|---|"]
        lines += [f"| {k} | {v} |" for k, v in sorted(report.skipped.items())]
        lines.append("")

    if report.protocol:
        lines += ["## Protocol", "", "| Field | Value |", "|---|---|"]
        lines += [
            f"| {k} | `{v}` |"
            for k, v in report.protocol.items()
            if v not in (None, {}, [])
        ]
        lines.append("")

    return "\n".join(lines)


_HTML_STYLE = """
:root { color-scheme: light dark; }
body { font-family: system-ui, -apple-system, sans-serif; margin: 2rem auto;
       max-width: 60rem; line-height: 1.5; padding: 0 1rem; }
table { border-collapse: collapse; width: 100%; margin: 1rem 0;
        display: block; overflow-x: auto; }
th, td { border: 1px solid #8884; padding: 0.4rem 0.6rem; text-align: left; }
th { background: #8881; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
.badge { font-size: 0.8rem; padding: 0.1rem 0.4rem; border-radius: 0.3rem;
         border: 1px solid currentColor; }
.faithful { color: #2a7; }
.ADAPTATION { color: #b83; }
.ORIGINAL { color: #46c; }
.MISMATCH { color: #c33; font-weight: bold; }
.UNAUDITED { color: #888; }
.warn { border-left: 4px solid #c33; padding: 0.5rem 1rem; background: #c331; }
figure { margin: 1.5rem 0; }
"""


def to_html(
    report: Report,
    path: Optional[Path | str] = None,
    *,
    figures: bool = False,
) -> str:
    """
    Render a self-contained HTML report.

    No external URLs: the page must open offline, which PLAN.md Section 9 lists
    as an acceptance check. With `figures=True` the profile plot — and a forest
    plot for every metric that has a per-group breakdown — is rendered to
    **inline SVG**, so the page stays a single file with nothing to fetch.

    `figures=True` needs the `viz` extra. If matplotlib is missing the report is
    still produced, with a note saying the figures were skipped, because a
    missing plot is not a reason to lose the numbers.
    """
    esc = html.escape
    counts = report.fidelity_counts()

    parts = [
        "<!doctype html>",
        '<html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>Bias report — {esc(report.model_id)}</title>",
        f"<style>{_HTML_STYLE}</style></head><body>",
        f"<h1>Bias report — <code>{esc(report.model_id)}</code></h1>",
        "<p>Every score carries its interval and its fidelity status. "
        "<strong>Scores from different metrics are not comparable and must not "
        "be averaged</strong> — the library computes no composite score by "
        "design.</p>",
    ]

    if counts.get("mismatch"):
        parts.append(
            f'<p class="warn"><strong>{counts["mismatch"]} metric(s) in this '
            "report are known MISMATCHES</strong> — they do not implement the "
            "metric they cite.</p>"
        )

    if figures:
        parts += _html_figures(report)
    parts += _html_families(report)
    parts += _html_notes(report)
    parts += _html_pairs("Not run", sorted(report.skipped.items()),
                         ("Metric", "Reason"))
    parts += _html_pairs(
        "Protocol",
        [(k, v) for k, v in report.protocol.items() if v not in (None, {}, [])],
        ("Field", "Value"), code=True,
    )
    parts.append("</body></html>")
    document = "\n".join(parts)

    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(document)
    return document


def _html_figures(report: "Report") -> List[str]:
    """The profile plot, plus a forest plot per metric with a breakdown."""
    try:
        from bias_scope.viz import figure_to_svg, plot_forest, plot_profile
    except ImportError:
        return [
            '<p class="warn">Figures skipped: matplotlib is not installed '
            "(<code>pip install bias-scope[viz]</code>). The tables below are "
            "unaffected.</p>"
        ]

    import matplotlib.pyplot as plt

    parts: List[str] = []
    figure = plot_profile(report)
    parts.append(f"<figure>{figure_to_svg(figure)}</figure>")
    plt.close(figure)

    for result in sorted(report.results, key=lambda r: r.metric):
        if not result.breakdown:
            continue
        figure = plot_forest(result)
        parts.append(f"<figure>{figure_to_svg(figure)}</figure>")
        plt.close(figure)
    return parts


def _html_families(report: "Report") -> List[str]:
    """One table per metric family."""
    esc = html.escape
    parts: List[str] = []
    for family, results in report.by_family().items():
        parts.append(f"<h2>{esc(family)}</h2>")
        parts.append(
            "<table><thead><tr><th>Metric</th><th>Score</th><th>95% CI</th>"
            "<th>n</th><th>Normalised</th><th>Fidelity</th></tr></thead><tbody>"
        )
        for r in sorted(results, key=lambda x: x.metric):
            badge = FIDELITY_BADGE[r.info.fidelity]
            parts.append(
                f"<tr><td>{esc(r.metric)}</td>"
                f'<td class="num">{r.score:.4g}</td>'
                f"<td>{esc(_interval(r))}</td>"
                f'<td class="num">{r.n}</td>'
                f'<td class="num">{r.normalized_deviation():+.3f}</td>'
                f'<td><span class="badge {badge}">{badge}</span></td></tr>'
            )
        parts.append("</tbody></table>")
    return parts


def _html_notes(report: "Report") -> List[str]:
    """Deviation notes, so a caveat is never more than one screen from a score."""
    esc = html.escape
    return [
        f"<p><small><strong>{esc(r.metric)}</strong> — "
        f"{esc(r.info.deviation_note)}</small></p>"
        for r in sorted(report.results, key=lambda x: x.metric)
        if r.info.deviation_note
    ]


def _html_pairs(title, rows, headers, code: bool = False) -> List[str]:
    """A simple two-column table, omitted entirely when there is nothing to show."""
    if not rows:
        return []
    esc = html.escape
    parts = [
        f"<h2>{esc(title)}</h2><table><thead><tr>"
        f"<th>{esc(headers[0])}</th><th>{esc(headers[1])}</th>"
        "</tr></thead><tbody>"
    ]
    for key, value in rows:
        rendered = f"<code>{esc(str(value))}</code>" if code else esc(str(value))
        parts.append(f"<tr><td>{esc(str(key))}</td><td>{rendered}</td></tr>")
    parts.append("</tbody></table>")
    return parts


# ── Cross-report analysis ─────────────────────────────────────────────────
@dataclass
class Delta:
    """One metric's change between two reports."""

    metric: str
    a: float
    b: float
    delta: float
    ci_a: Optional[Tuple[float, float]]
    ci_b: Optional[Tuple[float, float]]
    fidelity: str

    @property
    def intervals_overlap(self) -> Optional[bool]:
        """True when the two 95% intervals overlap, or None if either is absent.

        Non-overlapping intervals are the criterion PLAN.md Section 6.2 uses
        for "distinguishable"; a delta whose intervals overlap should not be
        read as a real difference.
        """
        if self.ci_a is None or self.ci_b is None:
            return None
        return self.ci_a[0] <= self.ci_b[1] and self.ci_b[0] <= self.ci_a[1]


def compare(report_a: Report, report_b: Report) -> List[Delta]:
    """
    Paired per-metric deltas between two reports.

    Raises:
        ValueError: If the two reports do not cover the same metrics. Comparing
            different metric sets silently would produce a table that looks
            like a model comparison but is not one.
    """
    a_scores = {r.metric: r for r in report_a.results}
    b_scores = {r.metric: r for r in report_b.results}
    if set(a_scores) != set(b_scores):
        only_a = sorted(set(a_scores) - set(b_scores))
        only_b = sorted(set(b_scores) - set(a_scores))
        raise ValueError(
            "compare() needs the same metric set in both reports. "
            f"Only in {report_a.model_id}: {only_a}. "
            f"Only in {report_b.model_id}: {only_b}."
        )

    return [
        Delta(
            metric=name,
            a=a_scores[name].score,
            b=b_scores[name].score,
            delta=b_scores[name].score - a_scores[name].score,
            ci_a=a_scores[name].ci,
            ci_b=b_scores[name].ci,
            fidelity=a_scores[name].info.fidelity,
        )
        for name in sorted(a_scores)
    ]


def _rank(values: Sequence[float]) -> List[float]:
    """Ranks with ties averaged, for Spearman."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        average = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = average
        i = j + 1
    return ranks


def _pearson(x: Sequence[float], y: Sequence[float]) -> float:
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    dx = math.sqrt(sum((a - mx) ** 2 for a in x))
    dy = math.sqrt(sum((b - my) ** 2 for b in y))
    if dx == 0 or dy == 0:
        return float("nan")
    return num / (dx * dy)


def correlate(
    reports: Sequence[Report],
    method: str = "spearman",
) -> Dict[str, Dict[str, float]]:
    """
    Metric × metric correlation across models.

    Each model contributes one observation per metric, so this answers "do these
    two metrics rank models the same way?" — the cross-metric analysis the v0.1
    reviewers found missing.

    Args:
        reports (Sequence[Report]): One per model. At least three are needed for
            a correlation to mean anything; two always give ±1.
        method (str): "spearman" (default, rank-based) or "pearson".

    Returns:
        dict: Nested `matrix[metric_a][metric_b]`, over the metrics present in
        **every** report.

    Raises:
        ValueError: If fewer than three reports are given, or the reports share
            fewer than two metrics.
    """
    if method not in ("spearman", "pearson"):
        raise ValueError(f"method must be 'spearman' or 'pearson', got {method!r}")
    if len(reports) < 3:
        raise ValueError(
            f"correlate() needs at least 3 reports to be meaningful, got "
            f"{len(reports)}. With two models every correlation is exactly ±1."
        )

    shared = set(reports[0].scores())
    for report in reports[1:]:
        shared &= set(report.scores())
    names = sorted(shared)
    if len(names) < 2:
        raise ValueError(
            f"reports share only {len(names)} metric(s); need at least 2 to "
            "correlate"
        )

    columns = {n: [r.scores()[n] for r in reports] for n in names}
    if method == "spearman":
        columns = {n: _rank(v) for n, v in columns.items()}

    return {
        a: {b: _pearson(columns[a], columns[b]) for b in names} for a in names
    }


__all__ = [
    "Report",
    "Delta",
    "save_json",
    "to_markdown",
    "to_html",
    "compare",
    "correlate",
    "FIDELITY_BADGE",
]
