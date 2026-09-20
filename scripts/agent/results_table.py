"""One table of agent results across target models.

Rows are target models (each model's latest *complete* run, per
`recommendation_coverage`; a model with no complete run keeps its latest and
is flagged), columns are metrics, and every cell is the score and item count
exactly as `summarize_report` printed them - the library's number, never the
agent's prose. A cell is starred when the result carries a recorded protocol
deviation (a substitute classifier or corpus), because the fidelity badge
alone cannot show that.

Usage:
    python scripts/agent/results_table.py                 # prints Markdown
    python scripts/agent/results_table.py --out results/verification/agent_live/RESULTS.md
    python scripts/agent/results_table.py --format latex \\
        --out results/verification/agent_live/RESULTS.tex
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # `scripts.` when run as a file

from scripts.agent.live_conversation import recommendation_coverage  # noqa: E402
from scripts.agent.summarize_runs import _DEFAULT_DIR, _SCORE_LINE, _load  # noqa: E402

_KIND_ORDER = {"encoder": 0, "embedding": 1, "causal": 2}
_FAMILY_ORDER = ("embedding", "probability", "generated_text", "prompt")


def since(records: List[Dict[str, Any]], stamp: str) -> List[Dict[str, Any]]:
    """Runs recorded at or after `stamp` (ISO date or datetime prefix)."""
    return [r for r in records if r.get("recorded_at", "") >= stamp]


def compare_tables(before: Dict[str, Dict[str, str]], after: Dict[str, Dict[str, str]]):
    """(model, metric, old cell, new cell) for every cell that differs."""
    rows = []
    for model in sorted(set(before) | set(after)):
        metrics = set(before.get(model, {})) | set(after.get(model, {}))
        for metric in sorted(metrics):
            old = before.get(model, {}).get(metric, "")
            new = after.get(model, {}).get(metric, "")
            if old != new:
                rows.append((model, metric, old, new))
    return rows


def cells_by_model(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, str]]:
    """The pivot's rows as {model: {metric: cell}}, counts stripped."""
    return {row["model"]: {m: _strip_counts(c) for m, c in row["cells"].items() if c}
            for row in rows}


def latest_complete_runs(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """target_model -> its latest complete run, else its latest run, flagged."""
    chosen: Dict[str, Dict[str, Any]] = {}
    for record in sorted(records, key=lambda r: r.get("recorded_at", "")):
        record = dict(record)
        record["_complete"] = recommendation_coverage(record)["complete"]
        model = record["target_model"]
        current = chosen.get(model)
        if current is None or record["_complete"] or not current["_complete"]:
            chosen[model] = record
    return chosen


def _cells(record: Dict[str, Any]) -> Dict[str, str]:
    """metric -> "score (n)", starred when a deviation line follows it."""
    cells: Dict[str, str] = {}
    last = None
    for entry in record["dispatched"]:
        if entry.get("tool") != "summarize_report" or not entry.get("ok"):
            continue
        for line in str(entry.get("output", "")).splitlines():
            match = _SCORE_LINE.match(line)
            if match:
                last = match["metric"]
                cells[last] = f"{match['score']} ({match['n'] or '-'})"
            elif last and line.strip().startswith("deviation:") and not cells[last].endswith("*"):
                cells[last] += "*"
    return cells


def _family_rank(metric: str) -> int:
    from bias_scope.metadata import list_metrics

    info = list_metrics().get(metric)
    return _FAMILY_ORDER.index(info.family) if info and info.family in _FAMILY_ORDER else 99


def pivot(runs: Dict[str, Dict[str, Any]]) -> Tuple[List[str], List[Dict[str, Any]]]:
    """(metric columns in family order then first seen, one row per model)."""
    rows = []
    seen: List[str] = []
    for model, record in sorted(
        runs.items(), key=lambda kv: (_KIND_ORDER.get(kv[1].get("scenario"), 9), kv[0])
    ):
        cells = _cells(record)
        seen.extend(m for m in cells if m not in seen)
        rows.append({"model": model, "kind": record.get("scenario", "?"),
                     "complete": record["_complete"], "cells": cells})
    metrics = sorted(seen, key=lambda m: (_family_rank(m), seen.index(m)))
    for row in rows:
        row["cells"] = {m: row["cells"].get(m, "") for m in metrics}
    return metrics, rows


_NOTES = """
## Reading the table

- Each cell is the score exactly as `summarize_report` printed it, from that model's
  latest run in which every feedable metric scored (or its latest run, marked
  *incomplete*, when none did). The item count `n` behind each score is in the
  transcript (`README.md`) and in the table produced with `--with-counts`; for
  StereotypicalAssociations it is the number of target words that co-occurred with
  a group word, which is small.
- `*` marks a result whose protocol records a deviation from the paper's resources:
  CEAT's contexts come from BOLD's Wikipedia sentences rather than the authors' Reddit
  sample and are pooled as sentences (RL-071); EMT is scored by `unitary/toxic-bert`
  rather than the Perspective API (RL-072). Those numbers are not comparable to the
  papers' tables. The badge next to each score in the transcript is the metric's
  fidelity to its paper; `deviation:` lines there say what this run substituted.
- Causal models are run in bf16 (PLAN.md Section 1); the embedding metrics now use the
  same copy of the model, so they are bf16 numbers too, and bf16 moves an effect size
  by up to 0.1 between CPU and GPU on the same weights (RL-077). Encoders and
  sentence encoders are fp32.
- SEAT and CEAT read the hidden state at position 0. On a model whose tokenizer
  prepends a BOS token (Llama, Gemma) that state is the same for every sentence, so
  they decline (empty cell) or return a degenerate 0 (RL-068). GPT-2 and Qwen add no
  BOS.
- A model with no scores at all is listed below the table with the reason.
- Regenerate with `python scripts/agent/results_table.py --out <this file>`; the logs
  behind every cell are in `README.md`, the procedure in `REPRODUCE.md`.
"""


# One line per metric on what the number is. Scale, neutral point and direction
# come from MetricInfo at render time, not from here.
_MEANING = {
    "WEAT": "effect size d of the association between two target word sets and two "
            "attribute sets in the model's embeddings; |d| 0.2 / 0.5 / 0.8 small / medium / large",
    "SEAT": "the same effect size on WEAT's words placed in template sentences",
    "CEAT": "the same effect size pooled over 1,000 random draws of contexts for each word",
    "CrowSPairs": "% of minimal pairs where the model prefers the more stereotypical sentence",
    "AUL": "% of pairs where the model prefers the stereotypical sentence, all tokens unmasked",
    "AULA": "AUL with tokens weighted by attention",
    "CAT": "stereotype score: % of StereoSet items where the stereotypical fill beats the "
           "anti-stereotypical one",
    "ICAT": "language-modelling score x how close the stereotype score is to 50; 100 = ideal, "
            "0 = worst",
    "RegardScore": "difference between two groups in the share of continuations judged "
                   "negative-regard (group A minus group B)",
    "GenderPolarity": "mean over continuations of (male words - female words) / (male + female); "
                      "+1 all male, -1 all female",
    "DemographicRepresentation": "total variation distance between how often each group's words "
                                 "appear in the continuations and a uniform distribution",
    "StereotypicalAssociations": "mean over target adjectives of that distance, counting only "
                                 "continuations where the adjective co-occurs with a group word",
    "HONEST": "share of continuations containing a HurtLex hurtful term",
    "EMT": "expected maximum toxicity: mean over prompts of the most toxic of K continuations",
}


def _direction_words(info: Any) -> str:
    return {
        "signed": "0 is no bias; the sign says which side",
        "higher_more_biased": "higher is more biased",
        "lower_more_biased": "lower is more biased",
    }.get(info.direction, info.direction)


def render_legend(metrics: List[str]) -> str:
    from bias_scope.metadata import list_metrics

    infos = list_metrics()
    lines = ["## What each column means", "",
             "| metric | neutral (no bias) | range | direction | what the number is |",
             "|---|---|---|---|---|"]
    for name in metrics:
        info = infos.get(name)
        if info is None:
            continue
        lo, hi = info.value_range
        rng = f"{lo:g} to {hi:g}".replace("-inf", "-inf").replace("inf", "inf")
        lines.append(f"| {name} | {info.neutral_value:g} | {rng} | {_direction_words(info)} | "
                     f"{_MEANING.get(name, '')} |")
    return "\n".join(lines) + "\n"


def _strip_counts(cell: str) -> str:
    """"0.61 (16)*" -> "0.61*": the score alone; n stays in the transcripts."""
    return re.sub(r" \([^)]*\)", "", cell)


def _neutral(metric: str) -> str:
    from bias_scope.metadata import list_metrics

    info = list_metrics().get(metric)
    return f"{info.neutral_value:g}" if info else ""


def render_markdown(metrics: List[str], rows: List[Dict[str, Any]], counts: bool = False) -> str:
    show = (lambda c: c) if counts else _strip_counts
    scored = [row for row in rows if any(row["cells"].values())]
    unscored = [row for row in rows if not any(row["cells"].values())]
    lines = ["# Agent results across models", "",
             "| model | kind | " + " | ".join(metrics) + " |",
             "|---|---|" + "---|" * len(metrics),
             "| *neutral value* | | " + " | ".join(_neutral(m) for m in metrics) + " |"]
    for row in scored:
        name = row["model"] + ("" if row["complete"] else " (incomplete)")
        lines.append(f"| {name} | {row['kind']} | "
                     + " | ".join(show(row["cells"][m]) for m in metrics) + " |")
    if unscored:
        lines += ["", "No scores at all (every run failed before a metric scored): "
                  + ", ".join(f"`{row['model']}`" for row in unscored)
                  + ". The transcripts record why."]
    return "\n".join(lines) + "\n\n" + render_legend(metrics) + _NOTES


def _tex(text: str) -> str:
    return text.replace("_", r"\_").replace("%", r"\%").replace("*", r"$^{*}$")


def render_latex(metrics: List[str], rows: List[Dict[str, Any]], counts: bool = False) -> str:
    """A booktabs table with the same cells; the neutral value is the first row."""
    show = (lambda c: c) if counts else _strip_counts
    scored = [row for row in rows if any(row["cells"].values())]
    header = " & ".join(["model", "kind"] + [_tex(m) for m in metrics]) + r" \\"
    neutral = " & ".join(["neutral value", ""] + [_neutral(m) for m in metrics]) + r" \\"
    body = []
    for row in scored:
        name = _tex(row["model"]) + ("" if row["complete"] else " (incomplete)")
        body.append(" & ".join([name, row["kind"]] + [_tex(show(row["cells"][m])) for m in metrics])
                    + r" \\")
    return "\n".join([
        r"% Generated by scripts/agent/results_table.py --format latex",
        r"% needs \usepackage{booktabs}",
        r"\begin{table*}[t]",
        r"\centering\small",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{ll" + "r" * len(metrics) + "}",
        r"\toprule",
        header,
        r"\midrule",
        neutral,
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}}",
        r"\caption{Bias scores per model, as reported by the library"
        + (" (score, then $n$ in parentheses)" if counts else "") + ". "
        r"The first row is each metric's neutral value, the score of a model with "
        r"no measured preference. $^{*}$ marks a result whose protocol records a substituted "
        r"resource (CEAT: Wikipedia contexts for the authors' Reddit sample; EMT: a local "
        r"toxicity classifier for the Perspective API).}",
        r"\label{tab:agent-results}",
        r"\end{table*}",
    ]) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=_DEFAULT_DIR)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--format", choices=("markdown", "latex"), default="markdown")
    parser.add_argument("--with-counts", action="store_true", help="append (n) to each score")
    parser.add_argument("--since", default=None, help="only runs recorded at/after this stamp")
    parser.add_argument("--compare-before", default=None,
                        help="also print the cells that changed vs runs recorded before --since")
    args = parser.parse_args()
    records = _load(args.dir)
    chosen = since(records, args.since) if args.since else records
    table = pivot(latest_complete_runs(chosen))
    render = render_latex if args.format == "latex" else render_markdown
    text = render(*table, counts=args.with_counts)
    if args.compare_before and args.since:
        earlier = [r for r in records if r.get("recorded_at", "") < args.since]
        before = cells_by_model(pivot(latest_complete_runs(earlier))[1])
        changed = compare_tables(before, cells_by_model(table[1]))
        text += "\n\n## Cells that changed vs the runs before " + args.since + "\n\n"
        text += "| model | metric | before | after |\n|---|---|---|---|\n"
        text += "\n".join(f"| {m} | {k} | {o} | {n} |" for m, k, o, n in changed) + "\n"
    print(text)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
