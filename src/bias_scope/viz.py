"""Plots (PLAN.md Section 9). Extra: `viz` (matplotlib).

Section 9's rules, which are constraints on what may be drawn, not style
preferences:

- **Never** a composite score, gauge, traffic light, or single-polygon radar.
  Each of those implies the metrics share a scale. They do not.
- **Every plotted value carries its CI.** A point estimate drawn alone claims a
  precision the metric does not have.
- **Every figure footer shows** the protocol hash, seed, dtype and resource
  versions, so a figure lifted into a paper still says how it was produced.
- **Fidelity appears as a badge**, so a `mismatch` bar cannot be read as a
  finding about the cited paper.

Each plot is a plain function taking a report and returning a `Figure`. No
plotting classes, no styling framework; shared helpers live in `_style()`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

if TYPE_CHECKING:  # matplotlib is an optional extra
    from matplotlib.figure import Figure

from bias_scope.report import FIDELITY_BADGE, Report

#: Colour per fidelity status. Chosen to be distinguishable in greyscale by
#: lightness as well as hue, since papers still get printed.
FIDELITY_COLOR = {
    "faithful": "#2a7f62",
    "adaptation": "#b8862f",
    "original": "#4466cc",
    "mismatch": "#cc3333",
    "unaudited": "#888888",
}


def _require_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")  # never require a display
        import matplotlib.pyplot as plt

        return plt
    except ImportError as exc:  # pragma: no cover - needs the extra missing
        raise ImportError(
            "Plotting requires matplotlib. Install bias-scope[viz]."
        ) from exc


def _style(ax, title: str, xlabel: str) -> None:
    """The small amount of shared styling the plots need."""
    ax.set_title(title, fontsize=11, loc="left")
    ax.set_xlabel(xlabel, fontsize=9)
    ax.tick_params(labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def protocol_footer(protocol: Dict[str, Any]) -> str:
    """
    The provenance line every figure carries.

    Section 9 requires the protocol hash, seed, dtype and resource versions on
    every figure. A figure that leaves its protocol behind cannot be checked.
    """
    bits = [
        f"protocol {protocol.get('hash', '?')}",
        f"seed {protocol.get('seed', '?')}",
        f"dtype {protocol.get('dtype') or 'n/a'}",
        f"bias-scope {protocol.get('library_version', '?')}",
    ]
    resources = protocol.get("resources") or []
    if resources:
        names = ", ".join(str(r.get("name", "?")) for r in resources)
        bits.append(f"resources: {names}")
    if protocol.get("judge_model"):
        bits.append(f"judge {protocol['judge_model']}")
    return " · ".join(bits)


def _add_footer(fig, protocol: Dict[str, Any]) -> None:
    fig.text(0.01, 0.01, protocol_footer(protocol), fontsize=6, color="#666")


def plot_profile(report: Report, figsize=(9, None)) -> "Figure":
    """
    One row per metric, grouped by family; x is the normalised deviation.

    Zero is neutral and positive means more stereotyped, whatever each metric's
    own direction. **This is a display transform, not an aggregate** — the rows
    are deliberately not summarised, and no total is drawn.

    Metrics with an interval get a whisker; metrics without one get a marker
    only, so the difference is visible rather than implied.
    """
    plt = _require_matplotlib()

    rows: List[tuple] = []
    for family, results in report.by_family().items():
        for result in sorted(results, key=lambda r: r.metric):
            rows.append((family, result))
    if not rows:
        raise ValueError("report contains no results to plot")

    height = figsize[1] or max(2.5, 0.32 * len(rows) + 1.6)
    fig, ax = plt.subplots(figsize=(figsize[0], height))

    labels = []
    for i, (family, result) in enumerate(rows):
        y = len(rows) - i - 1
        value = result.normalized_deviation()
        colour = FIDELITY_COLOR[result.info.fidelity]

        if result.ci is not None:
            low = result.ci[0] - result.score + value
            high = result.ci[1] - result.score + value
            ax.plot([low, high], [y, y], color=colour, linewidth=1.4, alpha=0.75,
                    solid_capstyle="butt")
            ax.plot([low, low, high, high], [y - 0.13, y + 0.13, y + 0.13, y - 0.13],
                    color=colour, linewidth=0.9, linestyle="none", marker="_")
        ax.plot([value], [y], marker="o", markersize=5, color=colour,
                markeredgecolor="white", markeredgewidth=0.6, zorder=3)
        labels.append(f"{result.metric}  [{family}]")

    ax.axvline(0.0, color="#333", linewidth=0.9, linestyle="--", alpha=0.6)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(list(reversed(labels)), fontsize=8)
    _style(ax, f"Bias profile — {report.model_id}",
           "normalised deviation (0 = neutral, + = more stereotyped)")

    handles = [
        plt.Line2D([], [], marker="o", linestyle="none", color=FIDELITY_COLOR[k],
                   label=FIDELITY_BADGE[k])
        for k in ("faithful", "adaptation", "original", "mismatch", "unaudited")
        if any(r.info.fidelity == k for _, r in rows)
    ]
    if handles:
        ax.legend(handles=handles, fontsize=7, loc="lower right", frameon=False)

    _add_footer(fig, report.protocol)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


def _dumbbell_interval(ax, result, y: float, color: str) -> None:
    """Draw one result's interval on the dumbbell row, on the deviation scale.

    `normalized_deviation` re-expresses a score as distance from neutral, so
    the interval is shifted by the same offset rather than drawn in the metric's
    own units — the shift `plot_profile` applies. A result with no interval gets
    no whisker; the bare marker then says "no interval", which is honest.
    """
    if result.ci is None:
        return
    offset = result.normalized_deviation() - result.score
    low, high = sorted(bound + offset for bound in result.ci)
    ax.plot([low, high], [y, y], color=color, linewidth=1.0, alpha=0.55,
            solid_capstyle="butt", zorder=2)


def plot_dumbbell(report_a: Report, report_b: Report, figsize=(9, None)) -> "Figure":
    """
    Two models on one axis: a marker each, joined by a line.

    Raises:
        ValueError: If the two reports cover different metric sets. A dumbbell
            with unmatched rows is a misleading picture, not a partial one.
    """
    plt = _require_matplotlib()
    from bias_scope.report import compare

    deltas = compare(report_a, report_b)  # raises on mismatched metric sets
    by_name = {r.metric: r for r in report_a.results}
    b_by_name = {r.metric: r for r in report_b.results}

    height = figsize[1] or max(2.5, 0.32 * len(deltas) + 1.6)
    fig, ax = plt.subplots(figsize=(figsize[0], height))

    for i, delta in enumerate(deltas):
        y = len(deltas) - i - 1
        result_a, result_b = by_name[delta.metric], b_by_name[delta.metric]
        va = result_a.normalized_deviation()
        vb = result_b.normalized_deviation()
        ax.plot([va, vb], [y, y], color="#bbb", linewidth=1.4, zorder=1)
        # Section 9: every plotted value carries its CI. On a dumbbell this is
        # the whole point — the gap between two markers reads as a difference,
        # and without the intervals there is nothing to say whether it is one.
        _dumbbell_interval(ax, result_a, y, "#4466cc")
        _dumbbell_interval(ax, result_b, y, "#b8862f")
        ax.plot([va], [y], marker="o", markersize=5, color="#4466cc", zorder=3)
        ax.plot([vb], [y], marker="D", markersize=4.5, color="#b8862f", zorder=3)

    ax.axvline(0.0, color="#333", linewidth=0.9, linestyle="--", alpha=0.6)
    ax.set_yticks(range(len(deltas)))
    ax.set_yticklabels([d.metric for d in reversed(deltas)], fontsize=8)
    _style(ax, f"{report_a.model_id}  vs  {report_b.model_id}",
           "normalised deviation (0 = neutral, + = more stereotyped)")
    ax.legend(
        handles=[
            plt.Line2D([], [], marker="o", linestyle="none", color="#4466cc",
                       label=report_a.model_id),
            plt.Line2D([], [], marker="D", linestyle="none", color="#b8862f",
                       label=report_b.model_id),
        ],
        fontsize=7, loc="lower right", frameon=False,
    )
    _add_footer(fig, report_a.protocol)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


def plot_agreement(
    matrix: Dict[str, Dict[str, float]],
    n_models: int,
    figsize=(7.5, 6.5),
) -> "Figure":
    """
    Metric × metric correlation heatmap.

    `n_models` is printed on the figure because a correlation over four models
    is not the same evidence as one over twenty, and a heatmap does not show
    its own sample size.
    """
    plt = _require_matplotlib()
    import numpy as np

    names = sorted(matrix)
    if len(names) < 2:
        raise ValueError("need at least 2 metrics to plot agreement")
    data = np.array([[matrix[a][b] for b in names] for a in names], dtype=float)

    fig, ax = plt.subplots(figsize=figsize)
    image = ax.imshow(data, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7)

    for i in range(len(names)):
        for j in range(len(names)):
            value = data[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center",
                        fontsize=5.5,
                        color="white" if abs(value) > 0.6 else "#222")

    ax.set_title(
        f"Metric agreement across {n_models} models"
        + ("  (n < 5: treat as indicative only)" if n_models < 5 else ""),
        fontsize=10, loc="left",
    )
    fig.colorbar(image, ax=ax, shrink=0.7, label="correlation")
    fig.tight_layout()
    return fig


def plot_envelope(
    runs: Sequence[Report],
    metric: str,
    variant_labels: Optional[Sequence[str]] = None,
    figsize=(8, 4),
) -> "Figure":
    """
    One metric's value across protocol variants, drawn as a band.

    This is the figure for PLAN.md 10.2's "same name, different numbers"
    finding: the spread a single metric shows across defensible protocol
    choices, next to its individual values.
    """
    plt = _require_matplotlib()

    values, intervals, labels = [], [], []
    for i, run in enumerate(runs):
        match = next((r for r in run.results if r.metric == metric), None)
        if match is None:
            raise ValueError(f"{metric} is not present in run {i} ({run.model_id})")
        values.append(match.score)
        intervals.append(match.ci)
        labels.append(
            variant_labels[i] if variant_labels else run.protocol.get("hash", str(i))
        )

    fig, ax = plt.subplots(figsize=figsize)
    xs = list(range(len(values)))
    ax.fill_between(xs, min(values), max(values), color="#4466cc", alpha=0.10,
                    label=f"spread across protocols ({max(values) - min(values):.4g})")
    for x, value, interval in zip(xs, values, intervals):
        if interval is not None:
            ax.plot([x, x], list(interval), color="#4466cc", linewidth=1.2, alpha=0.8)
        ax.plot([x], [value], marker="o", markersize=5, color="#4466cc")

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
    _style(ax, f"{metric} across protocol variants", "")
    ax.set_ylabel("score", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    _add_footer(fig, runs[0].protocol)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


def plot_forest(result, figsize=(7.5, None)) -> "Figure":
    """
    Per-group breakdown for one metric, with the overall score marked.

    Raises:
        ValueError: If the result carries no `breakdown`.
    """
    plt = _require_matplotlib()

    if not result.breakdown:
        raise ValueError(
            f"{result.metric} has no per-group breakdown to plot; "
            "plot_forest needs `BiasResult.breakdown`"
        )

    groups = sorted(result.breakdown)
    height = figsize[1] or max(2.2, 0.32 * len(groups) + 1.4)
    fig, ax = plt.subplots(figsize=(figsize[0], height))

    colour = FIDELITY_COLOR[result.info.fidelity]
    for i, group in enumerate(groups):
        y = len(groups) - i - 1
        ax.plot([result.breakdown[group]], [y], marker="o", markersize=5,
                color=colour, zorder=3)

    ax.axvline(result.score, color="#333", linewidth=1.0, linestyle="-",
               alpha=0.7, label=f"overall {result.score:.4g}")
    ax.axvline(result.info.neutral_value, color="#888", linewidth=0.9,
               linestyle="--", alpha=0.7,
               label=f"neutral {result.info.neutral_value:g}")
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels(list(reversed(groups)), fontsize=8)
    _style(ax, f"{result.metric} by group  [{FIDELITY_BADGE[result.info.fidelity]}]",
           "score")
    ax.legend(fontsize=7, frameon=False)
    _add_footer(fig, result.protocol)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


def figure_to_svg(fig) -> str:
    """Serialise a figure to inline SVG, for embedding in the HTML report."""
    import io

    buffer = io.StringIO()
    fig.savefig(buffer, format="svg", bbox_inches="tight")
    svg = buffer.getvalue()
    # Drop the XML declaration and DOCTYPE so the fragment can be inlined.
    start = svg.find("<svg")
    return svg[start:] if start >= 0 else svg


__all__ = [
    "plot_profile",
    "plot_dumbbell",
    "plot_agreement",
    "plot_envelope",
    "plot_forest",
    "figure_to_svg",
    "protocol_footer",
    "FIDELITY_COLOR",
]
