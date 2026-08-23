#!/usr/bin/env python3
"""Recount metrics by family and fidelity for the paper (PLAN.md 7.2, 11.2).

PLAN.md 11.2: "Paper numbers must be produced by these scripts from results/,
never typed by hand." This is the generator for the family x fidelity table and
for the sentence naming how many metrics are faithful.

    python scripts/paper/metric_counts.py            # markdown to stdout
    python scripts/paper/metric_counts.py --csv results/paper/metric_counts.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

import bias_scope  # noqa: E402,F401  (importing populates the registry)
from bias_scope._metric_info import METRIC_INFO  # noqa: E402
from bias_scope.metadata import FIDELITIES  # noqa: E402

FAMILIES = ("embedding", "probability", "generated_text", "prompt")


def counts():
    """Return ``{family: Counter(fidelity)}`` over every registered metric."""
    table = defaultdict(Counter)
    for info in METRIC_INFO.values():
        table[info.family][info.fidelity] += 1
    return table


def to_markdown(table) -> str:
    """Render the family x fidelity table plus the one-sentence summary."""
    lines = [
        "| Family | " + " | ".join(f.capitalize() for f in FIDELITIES) + " | Total |",
        "|" + "---|" * (len(FIDELITIES) + 2),
    ]
    totals = Counter()
    for family in FAMILIES:
        row = table[family]
        totals.update(row)
        lines.append(
            f"| {family} | "
            + " | ".join(str(row.get(f, 0)) for f in FIDELITIES)
            + f" | {sum(row.values())} |"
        )
    lines.append(
        "| **All** | "
        + " | ".join(f"**{totals.get(f, 0)}**" for f in FIDELITIES)
        + f" | **{sum(totals.values())}** |"
    )
    lines += [
        "",
        f"{sum(totals.values())} metrics: {totals.get('faithful', 0)} faithful, "
        f"{totals.get('adaptation', 0)} adaptation, {totals.get('original', 0)} "
        f"original, {totals.get('mismatch', 0)} mismatch, "
        f"{totals.get('unaudited', 0)} unaudited.",
        "",
        "Adaptations and originals are **not** the cited papers' metrics and must "
        "not be presented as such (PLAN.md Section 1).",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, help="also write a CSV here")
    args = parser.parse_args()

    table = counts()
    print(to_markdown(table))

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["family", *FIDELITIES, "total"])
            for family in FAMILIES:
                row = table[family]
                writer.writerow(
                    [family, *(row.get(f, 0) for f in FIDELITIES), sum(row.values())]
                )
        print(f"\nwrote {args.csv}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
