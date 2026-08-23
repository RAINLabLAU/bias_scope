"""Plots (PLAN.md Section 9), including its acceptance checks.

Assertions are on the **drawn data and artists**, not on pixels — a pixel
snapshot breaks on every matplotlib release and tells you nothing about whether
the figure is right.

Section 9's acceptance checks appear as named tests:

- the profile figure has one row per result and one CI whisker per item-level
  result (counted from the Axes artists)
- the footer text contains the protocol hash
- `plot_dumbbell` raises if the two reports have different metric sets
- the HTML report renders figures inline and opens offline (no external URLs)
"""

import pytest

from bias_scope.report import Report, correlate, to_html
from bias_scope.result import BiasResult, make_protocol

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from tests.test_framework import make_info, make_report, make_result  # noqa: E402

from bias_scope.viz import (  # noqa: E402
    FIDELITY_COLOR,
    figure_to_svg,
    plot_agreement,
    plot_dumbbell,
    plot_envelope,
    plot_forest,
    plot_profile,
    protocol_footer,
)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    import matplotlib.pyplot as plt

    plt.close("all")


def report_with_mixed_intervals():
    """Two metrics with intervals, one without."""
    return Report(
        model_id="test/model",
        results=[
            make_result("A", 0.6, ci=(0.5, 0.7), family="embedding"),
            make_result("B", 0.4, ci=(0.3, 0.5), family="probability"),
            BiasResult(
                metric="C", score=0.8, n=5, ci=None, ci_method="none",
                per_item=None, breakdown={}, details={},
                protocol=make_protocol("C", model_id="m", seed=42),
                info=make_info("C", family="prompt"),
            ),
        ],
        protocol=make_protocol("BiasSuite", model_id="test/model", seed=42,
                               dtype="bf16"),
    )


class TestProfile:
    def test_one_row_per_result(self):
        """PLAN.md Section 9 acceptance check."""
        report = report_with_mixed_intervals()
        ax = plot_profile(report).axes[0]
        assert len(ax.get_yticklabels()) == len(report.results)

    def test_one_ci_whisker_per_item_level_result(self):
        """PLAN.md Section 9 acceptance check.

        Two of the three results carry an interval, so exactly two horizontal
        whisker lines are drawn. A metric with no interval gets a marker only,
        so the difference is visible rather than implied.
        """
        ax = plot_profile(report_with_mixed_intervals()).axes[0]
        whiskers = [
            line for line in ax.get_lines()
            if line.get_linestyle() == "-" and len(line.get_xdata()) == 2
            and line.get_ydata()[0] == line.get_ydata()[1]
        ]
        assert len(whiskers) == 2

    def test_the_footer_contains_the_protocol_hash(self):
        """PLAN.md Section 9 acceptance check."""
        report = report_with_mixed_intervals()
        fig = plot_profile(report)
        footers = [t.get_text() for t in fig.texts]
        assert any(report.protocol["hash"] in f for f in footers)

    def test_the_footer_also_carries_seed_and_dtype(self):
        report = report_with_mixed_intervals()
        footer = " ".join(t.get_text() for t in plot_profile(report).texts)
        assert "seed 42" in footer
        assert "bf16" in footer

    def test_a_neutral_line_is_drawn_at_zero(self):
        ax = plot_profile(report_with_mixed_intervals()).axes[0]
        verticals = [line.get_xdata()[0] for line in ax.get_lines()
                     if line.get_linestyle() == "--"]
        assert 0.0 in verticals

    def test_fidelity_is_encoded_in_colour(self):
        report = Report(
            model_id="m",
            results=[make_result("Good", 0.5), make_result("Bad", 0.5,
                                                           fidelity="mismatch")],
            protocol=make_protocol("s", seed=42),
        )
        colours = {line.get_color() for line in plot_profile(report).axes[0].get_lines()}
        assert FIDELITY_COLOR["mismatch"] in colours

    def test_an_empty_report_raises_rather_than_drawing_nothing(self):
        empty = Report(model_id="m", results=[], protocol=make_protocol("s", seed=42))
        with pytest.raises(ValueError, match="no results"):
            plot_profile(empty)

    def test_no_composite_is_drawn(self):
        """Section 9 forbids a composite score, gauge or single-polygon radar."""
        ax = plot_profile(report_with_mixed_intervals()).axes[0]
        labels = [t.get_text().lower() for t in ax.get_yticklabels()]
        for banned in ("overall", "composite", "total", "average"):
            assert not any(banned in label for label in labels)


class TestDumbbell:
    def test_it_draws_two_markers_per_metric(self):
        a = make_report("a", {"M1": 0.6, "M2": 0.3})
        b = make_report("b", {"M1": 0.9, "M2": 0.1})
        ax = plot_dumbbell(a, b).axes[0]
        circles = [ln for ln in ax.get_lines() if ln.get_marker() == "o"]
        diamonds = [ln for ln in ax.get_lines() if ln.get_marker() == "D"]
        assert len(circles) == 2 and len(diamonds) == 2

    def test_every_plotted_value_carries_its_interval(self):
        """PLAN.md Section 9: "Every plotted value carries its CI."

        The dumbbell drew bare markers until 0.2.0, which is the most
        misleading case of all: two models whose intervals overlap completely
        read as a clear difference because the gap between the markers is the
        only thing on the row. One whisker per marker, so four for two metrics.
        """
        a = make_report("a", {"M1": 0.6, "M2": 0.3})
        b = make_report("b", {"M1": 0.9, "M2": 0.1})
        ax = plot_dumbbell(a, b).axes[0]
        whiskers = [ln for ln in ax.get_lines()
                    if ln.get_marker() in ("", "None", None)
                    and len(ln.get_xdata()) == 2
                    and ln.get_ydata()[0] == ln.get_ydata()[1]
                    and ln.get_linewidth() < 1.4]
        assert len(whiskers) == 4, (
            f"expected one CI whisker per marker, found {len(whiskers)}"
        )

    def test_a_result_without_an_interval_still_gets_its_marker(self):
        """A missing interval is not a reason to drop the point."""
        a = make_report("a", {"M1": 0.6})
        b = make_report("b", {"M1": 0.9})
        b.results[0] = BiasResult(
            metric="M1", score=0.9, n=5, ci=None, ci_method="none",
            per_item=None, breakdown={}, details={},
            protocol=make_protocol("M1", model_id="b", seed=42),
            info=b.results[0].info,
        )
        ax = plot_dumbbell(a, b).axes[0]
        assert len([ln for ln in ax.get_lines() if ln.get_marker() == "D"]) == 1

    def test_mismatched_metric_sets_raise(self):
        """PLAN.md Section 9 acceptance check."""
        a = make_report("a", {"M1": 0.6})
        b = make_report("b", {"M2": 0.6})
        with pytest.raises(ValueError, match="same metric set"):
            plot_dumbbell(a, b)

    def test_both_model_ids_are_in_the_title_or_legend(self):
        a = make_report("model-a", {"M": 0.6})
        b = make_report("model-b", {"M": 0.2})
        fig = plot_dumbbell(a, b)
        text = fig.axes[0].get_title(loc="left") + " ".join(
            t.get_text() for t in fig.axes[0].get_legend().get_texts()
        )
        assert "model-a" in text and "model-b" in text


class TestAgreement:
    def _matrix(self):
        reports = [make_report(f"m{i}", {"A": v, "B": v * 2, "C": 1 - v})
                   for i, v in enumerate([0.1, 0.3, 0.5, 0.7])]
        return correlate(reports), len(reports)

    def test_the_heatmap_is_square_over_the_shared_metrics(self):
        matrix, n = self._matrix()
        ax = plot_agreement(matrix, n).axes[0]
        assert len(ax.get_xticklabels()) == len(matrix)
        assert len(ax.get_yticklabels()) == len(matrix)

    def test_the_model_count_is_shown(self):
        """A correlation over four models is not the evidence twenty would be."""
        matrix, n = self._matrix()
        assert f"{n} models" in plot_agreement(matrix, n).axes[0].get_title(loc="left")

    def test_a_small_sample_is_called_out(self):
        matrix, _ = self._matrix()
        assert "indicative only" in plot_agreement(matrix, 3).axes[0].get_title(loc="left")
        assert "indicative only" not in plot_agreement(matrix, 12).axes[0].get_title(loc="left")

    def test_a_single_metric_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            plot_agreement({"A": {"A": 1.0}}, 5)


class TestEnvelope:
    def _runs(self):
        return [make_report(f"variant{i}", {"M": v}) for i, v in
                enumerate([0.4, 0.45, 0.6, 0.42])]

    def test_it_plots_one_point_per_variant(self):
        ax = plot_envelope(self._runs(), "M").axes[0]
        assert len(ax.get_xticklabels()) == 4

    def test_the_legend_states_the_spread(self):
        fig = plot_envelope(self._runs(), "M")
        legend = " ".join(t.get_text() for t in fig.axes[0].get_legend().get_texts())
        assert "spread across protocols" in legend
        assert "0.2" in legend  # max 0.6 - min 0.4

    def test_custom_labels_are_used(self):
        ax = plot_envelope(self._runs(), "M",
                           variant_labels=["k=5", "k=10", "k=20", "k=40"]).axes[0]
        assert [t.get_text() for t in ax.get_xticklabels()] == \
               ["k=5", "k=10", "k=20", "k=40"]

    def test_a_metric_absent_from_a_run_raises(self):
        with pytest.raises(ValueError, match="not present in run"):
            plot_envelope(self._runs(), "NotThere")


class TestForest:
    def test_one_point_per_group(self):
        result = make_result("M", 0.5,
                             breakdown={"male": 0.6, "female": 0.4, "other": 0.5})
        ax = plot_forest(result).axes[0]
        assert len(ax.get_yticklabels()) == 3

    def test_the_overall_and_neutral_lines_are_both_drawn(self):
        result = make_result("M", 0.55, breakdown={"a": 0.6, "b": 0.5})
        fig = plot_forest(result)
        legend = " ".join(t.get_text() for t in fig.axes[0].get_legend().get_texts())
        assert "overall" in legend and "neutral" in legend

    def test_a_result_without_a_breakdown_raises(self):
        with pytest.raises(ValueError, match="no per-group breakdown"):
            plot_forest(make_result("M", 0.5))

    def test_the_fidelity_badge_is_in_the_title(self):
        result = make_result("M", 0.5, breakdown={"a": 0.5}, fidelity="mismatch")
        assert "MISMATCH" in plot_forest(result).axes[0].get_title(loc="left")


class TestFooterAndSvg:
    def test_the_footer_names_resources_when_present(self):
        protocol = make_protocol("M", seed=1,
                                 resources=[{"name": "hurtlex-1.2", "sha256": "x"}])
        assert "hurtlex-1.2" in protocol_footer(protocol)

    def test_the_footer_names_the_judge_when_present(self):
        protocol = make_protocol("M", seed=1, judge_model="openai/gpt-4o")
        assert "openai/gpt-4o" in protocol_footer(protocol)

    def test_svg_export_is_an_inline_fragment(self):
        svg = figure_to_svg(plot_profile(report_with_mixed_intervals()))
        assert svg.lstrip().startswith("<svg")
        assert "<?xml" not in svg

    def test_the_html_report_can_embed_a_figure_and_stay_offline(self):
        """PLAN.md Section 9 acceptance check."""
        report = report_with_mixed_intervals()
        svg = figure_to_svg(plot_profile(report))
        document = to_html(report).replace("</body>", f"<figure>{svg}</figure></body>")
        assert "<svg" in document

        # An SVG carries XML *namespace* declarations like
        # xmlns="http://www.w3.org/2000/svg". Those are identifiers, never
        # fetched, so a bare "http" scan would fail on every valid SVG. What
        # actually breaks offline rendering is a resource load, so look for
        # those instead.
        import re

        loads = re.findall(
            r'(?:src|href)\s*=\s*["\']https?://|url\(\s*["\']?https?://',
            document,
        )
        assert not loads, f"figure introduced external resource loads: {loads}"


class TestHtmlWithFigures:
    """`to_html(figures=True)` — PLAN.md Section 9's self-contained page."""

    def test_the_profile_is_embedded_inline(self):
        document = to_html(report_with_mixed_intervals(), figures=True)
        assert "<figure>" in document
        assert "<svg" in document

    def test_a_forest_plot_is_added_per_metric_with_a_breakdown(self):
        report = Report(
            model_id="m",
            results=[
                make_result("WithGroups", 0.5, breakdown={"a": 0.6, "b": 0.4}),
                make_result("WithoutGroups", 0.5),
            ],
            protocol=make_protocol("s", seed=42),
        )
        document = to_html(report, figures=True)
        # One profile + one forest.
        assert document.count("<figure>") == 2

    def test_the_page_stays_offline_with_figures(self):
        """Section 9 acceptance check: no external resource loads."""
        import re

        document = to_html(report_with_mixed_intervals(), figures=True)
        loads = re.findall(
            r'(?:src|href)\s*=\s*["\']https?://|url\(\s*["\']?https?://', document
        )
        assert not loads, f"external resource loads in the report: {loads}"

    def test_figures_are_opt_in(self):
        """The default stays fast and dependency-free."""
        assert "<figure>" not in to_html(report_with_mixed_intervals())

    def test_the_numbers_survive_a_missing_matplotlib(self, monkeypatch):
        """A missing plot must not cost the reader the tables."""
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "bias_scope.viz":
                raise ImportError("no matplotlib")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        document = to_html(report_with_mixed_intervals(), figures=True)
        assert "Figures skipped" in document
        assert "bias-scope[viz]" in document
        for result in report_with_mixed_intervals().results:
            assert result.metric in document
