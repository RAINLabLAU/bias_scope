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


_NOTES = """
## Reading the table

- Each cell is `score (n)` exactly as `summarize_report` printed it, from that model's
  latest run in which every feedable metric scored (or its latest run, marked
  *incomplete*, when none did). `n` is what the metric counts: pairs, sentences,
  sampled contexts, prompts, templates x K, or - for StereotypicalAssociations -
  target words that co-occurred with a group word, which is why it is small.
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


def render_markdown(metrics: List[str], rows: List[Dict[str, Any]]) -> str:
    scored = [row for row in rows if any(row["cells"].values())]
    unscored = [row for row in rows if not any(row["cells"].values())]
    lines = ["# Agent results across models", "",
             "| model | kind | " + " | ".join(metrics) + " |",
             "|---|---|" + "---|" * len(metrics)]
    for row in scored:
        name = row["model"] + ("" if row["complete"] else " (incomplete)")
        lines.append(f"| {name} | {row['kind']} | "
                     + " | ".join(row["cells"][m] for m in metrics) + " |")
    if unscored:
        lines += ["", "No scores at all (every run failed before a metric scored): "
                  + ", ".join(f"`{row['model']}`" for row in unscored)
                  + ". The transcripts record why."]
    return "\n".join(lines) + "\n" + _NOTES


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
