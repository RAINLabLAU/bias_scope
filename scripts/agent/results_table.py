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
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # `scripts.` when run as a file

from scripts.agent.live_conversation import recommendation_coverage  # noqa: E402
from scripts.agent.summarize_runs import _DEFAULT_DIR, _SCORE_LINE, _load  # noqa: E402

_KIND_ORDER = {"encoder": 0, "embedding": 1, "causal": 2}
_FAMILY_ORDER = ("embedding", "probability", "generated_text", "prompt")


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


def render_markdown(metrics: List[str], rows: List[Dict[str, Any]]) -> str:
    lines = ["| model | kind | " + " | ".join(metrics) + " |",
             "|---|---|" + "---|" * len(metrics)]
    for row in rows:
        name = row["model"] + ("" if row["complete"] else " (incomplete)")
        lines.append(f"| {name} | {row['kind']} | "
                     + " | ".join(row["cells"][m] for m in metrics) + " |")
    lines += [
        "",
        "Each cell is `score (n)` as `summarize_report` printed it, from the model's latest "
        "complete run. `*` marks a result whose protocol records a deviation from the paper's "
        "resources (a substitute classifier or corpus); the transcript's `deviation:` line says "
        "which. A model marked incomplete has no run in which every feedable metric scored.",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=_DEFAULT_DIR)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    text = render_markdown(*pivot(latest_complete_runs(_load(args.dir))))
    print(text)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
