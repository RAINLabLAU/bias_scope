"""Collect the recorded agent runs into one table of bias results.

Reads `results/verification/agent_live/*.json` and reports, per model type,
what each metric scored. Every number here is lifted from `summarize_report`'s
own return value as recorded in the transcript - that is, from the library -
never from the agent's prose about it. If the two ever disagree, this table
shows the library's number and `--check` says which figures in the agent's
final message could not be traced to any tool result.

Usage:
    python scripts/agent/summarize_runs.py
    python scripts/agent/summarize_runs.py --check
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List

_DEFAULT_DIR = Path("results/verification/agent_live")
_SCORE_LINE = re.compile(
    r"\s*\[(?P<fidelity>\w+)\]\s*(?P<metric>\w+):\s*(?P<score>[^\s(]+)"
    r"(?:\s*\(n=(?P<n>\d+)\))?"
)

# What each metric's neutral point is, so a score can be read without the paper
# open. Taken from MetricInfo at render time, not hardcoded here.


def _scores(record: Dict[str, Any]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for entry in record.get("dispatched", []):
        if entry.get("tool") != "summarize_report" or not entry.get("ok"):
            continue
        for line in str(entry.get("output", "")).splitlines():
            match = _SCORE_LINE.match(line)
            if match:
                rows.append(match.groupdict())
    return rows


def _load(directory: Path) -> List[Dict[str, Any]]:
    records = []
    for path in sorted(directory.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        record["_path"] = path.name
        records.append(record)
    return records


def _neutral_values() -> Dict[str, Any]:
    from bias_scope.metadata import list_metrics

    return {
        name: (info.neutral_value, info.direction, info.value_range)
        for name, info in list_metrics().items()
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=_DEFAULT_DIR)
    parser.add_argument("--check", action="store_true", help="also show untraceable figures")
    args = parser.parse_args()

    records = _load(args.dir)
    if not records:
        print(f"no run transcripts in {args.dir}")
        return 1

    meta = _neutral_values()
    header = (
        f"{'model type':10} {'target model':42} {'metric':12} "
        f"{'score':>9} {'n':>6}  {'neutral':>8}  fidelity"
    )
    print(header)
    print("-" * 100)
    for record in records:
        rows = _scores(record)
        if not rows:
            continue
        for row in rows:
            neutral = meta.get(row["metric"], (None,))[0]
            print(
                f"{record.get('scenario', '?'):10} {record.get('target_model', '?'):42} "
                f"{row['metric']:12} {row['score']:>9} {str(row.get('n') or '-'):>6}  "
                f"{str(neutral):>8}  {row['fidelity']}"
            )

    if args.check:
        print("\nfigures in the agent's final message with no matching tool result:")
        for record in records:
            untraceable = record.get("reported_numbers", {}).get(
                "not_traceable_to_a_tool_result", []
            )
            print(f"  {record['_path']}: {untraceable or 'none'}")
        _print_coverage(records)
    return 0


def _print_coverage(records: List[Dict[str, Any]]) -> None:
    """Recommended vs scored, recomputed from each transcript (not read from
    the file, so runs recorded before the check existed are covered too)."""
    from scripts.agent.live_conversation import recommendation_coverage

    print("\nrecommended metrics vs metrics actually scored:")
    print(f"  {'run':70} {'rec':>4} {'feed':>4} {'run':>4} {'scored':>6}  gap")
    for record in records:
        cov = recommendation_coverage(record)
        gap = cov["feedable_not_scored"] + [f"!{m}" for m in cov["scored_not_recommended"]]
        print(
            f"  {record['_path'][:70]:70} {len(cov['recommended']):>4} "
            f"{len(cov['feedable']):>4} {len(cov['run']):>4} {len(cov['scored']):>6}  "
            f"{', '.join(gap) or 'none'}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
