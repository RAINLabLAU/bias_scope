"""One table of agent results across models, from the library's own output.

`scripts/agent/results_table.py` pivots the recorded transcripts: one row per
target model (its latest *complete* run), one column per metric, each cell
the score and n as `summarize_report` printed them, starred when the result
carries a recorded protocol deviation. Nothing is taken from the agent's prose.
"""

from __future__ import annotations

from scripts.agent.results_table import latest_complete_runs, pivot, render_markdown


def _rec(model, scenario, stamp, summary, recommended=("WEAT",), scored=("WEAT",)):
    return {
        "target_model": model,
        "scenario": scenario,
        "recorded_at": stamp,
        "dispatched": [
            {"turn": 1, "tool": "recommend_metrics_tool", "ok": True, "input": {},
             "output": [{"metric": m, "access": ["embeddings"]} for m in recommended]},
            {"turn": 2, "tool": "list_datasets", "ok": True, "input": {},
             "output": [{"dataset": "weat", "metrics": ["WEAT"]},
                        {"dataset": "seat", "metrics": ["SEAT"]}]},
            {"turn": 3, "tool": "run_suite", "ok": True, "input": {"metric_names": list(scored)}},
            {"turn": 3, "tool": "summarize_report", "ok": True, "input": {}, "output": summary},
        ],
    }


_FULL = "embedding:\n  [faithful] WEAT: 0.61 (n=16)\n  [faithful] SEAT: 1.04 (n=128)\n"
_PARTIAL = "embedding:\n  [faithful] WEAT: 0.60 (n=16)\n\nskipped:\n  SEAT: boom\n"
_DEVIATING = (
    "embedding:\n  [faithful] CEAT: 0.08 (n=1000)\n"
    "      deviation: contexts drawn from Wikipedia, not Reddit\n"
)


def test_latest_complete_run_per_model_wins_over_a_newer_incomplete_one():
    both = ("WEAT", "SEAT")
    complete = _rec("m", "embedding", "2026-09-20T10:00:00", _FULL, both, both)
    newer_partial = _rec("m", "embedding", "2026-09-20T11:00:00", _PARTIAL, both, both)
    chosen = latest_complete_runs([newer_partial, complete])
    assert chosen["m"]["recorded_at"] == "2026-09-20T10:00:00"
    assert chosen["m"]["_complete"] is True


def test_a_model_with_no_complete_run_keeps_its_latest_and_is_flagged():
    both = ("WEAT", "SEAT")
    partial = _rec("m", "embedding", "2026-09-20T11:00:00", _PARTIAL, both, both)
    chosen = latest_complete_runs([partial])
    assert chosen["m"]["_complete"] is False


def test_pivot_has_one_row_per_model_and_one_column_per_metric_seen():
    runs = latest_complete_runs([
        _rec("a", "encoder", "2026-09-20T10:00:00", _FULL, ("WEAT", "SEAT"), ("WEAT", "SEAT")),
        _rec("b", "embedding", "2026-09-20T10:00:00", _DEVIATING, ("CEAT",), ("CEAT",)),
    ])
    metrics, rows = pivot(runs)
    assert metrics == ["WEAT", "SEAT", "CEAT"]           # family order, then first seen
    assert rows[0]["model"] == "a" and rows[0]["cells"]["WEAT"] == "0.61 (16)"
    assert rows[0]["cells"]["CEAT"] == ""                 # not applicable to a
    assert rows[1]["cells"]["CEAT"] == "0.08 (1000)*"     # starred: a recorded deviation


def test_markdown_carries_the_star_legend_and_flags_incomplete_rows():
    runs = latest_complete_runs([
        _rec("b", "embedding", "2026-09-20T10:00:00", _DEVIATING, ("CEAT",), ("CEAT",)),
        _rec("m", "embedding", "2026-09-20T11:00:00", _PARTIAL, ("WEAT", "SEAT"), ("WEAT", "SEAT")),
    ])
    text = render_markdown(*pivot(runs), counts=True)
    assert "| b |" in text and "0.08 (1000)*" in text
    assert "0.08*" in render_markdown(*pivot(runs))          # counts off by default
    assert "incomplete" in text and "m" in text
    assert "deviation" in text.lower()


def test_a_model_with_no_scores_is_listed_under_the_table_not_as_an_empty_row():
    runs = latest_complete_runs([
        _rec("ok", "causal", "2026-09-20T10:00:00", _FULL, ("WEAT", "SEAT"), ("WEAT", "SEAT")),
        _rec("dead", "causal", "2026-09-20T10:00:00", "", ("WEAT",), ()),
    ])
    text = render_markdown(*pivot(runs))
    assert "| ok |" in text
    assert "| dead" not in text and "`dead`" in text


def test_the_legend_states_each_metrics_neutral_value_from_the_metadata():
    from bias_scope.metadata import list_metrics

    runs = latest_complete_runs([
        _rec("a", "encoder", "2026-09-20T10:00:00", _FULL, ("WEAT", "SEAT"), ("WEAT", "SEAT")),
    ])
    text = render_markdown(*pivot(runs))
    assert "neutral (no bias)" in text
    assert f"| WEAT | {list_metrics()['WEAT'].neutral_value:g} |" in text


def _two_runs():
    return latest_complete_runs([
        _rec("a", "encoder", "2026-09-20T10:00:00", _FULL, ("WEAT", "SEAT"), ("WEAT", "SEAT")),
        _rec("b", "embedding", "2026-09-20T10:00:00", _DEVIATING, ("CEAT",), ("CEAT",)),
    ])


def test_markdown_table_starts_with_a_neutral_value_row():
    text = render_markdown(*pivot(_two_runs()))
    header, sep, first = text.splitlines()[2:5]
    assert header.startswith("| model | kind | WEAT | SEAT | CEAT |")
    assert first.startswith("| *neutral value* |") and "| 0 | 0 | 0 |" in first


def test_latex_table_has_the_same_rows_with_a_neutral_row_and_starred_deviations():
    from scripts.agent.results_table import render_latex

    text = render_latex(*pivot(_two_runs()), counts=True)
    assert r"\begin{tabular}" in text and r"\toprule" in text
    assert "neutral value &  & 0 & 0 & 0" in text
    assert r"0.08 (1000)$^{*}$" in text            # the deviation star, LaTeX-safe
    assert r"a & encoder & 0.61 (16) & 1.04 (128) &" in text
    assert r"a & encoder & 0.61 & 1.04 &" in render_latex(*pivot(_two_runs()))
    assert "all-MiniLM" not in text                # only the given rows
