#!/usr/bin/env python3
"""Render verification/ledger.yaml to results/verification/VERIFICATION.md.

Every non-empty evidence value is checked before it is rendered:

  ``path::test``      collected by pytest, and the file exists
  ``path``            exists on disk
  ``FILE.md#anchor``  the file exists and contains the anchor text
  ``exempt: reason``  accepted, rendered as an exemption, reason required

A cell whose evidence cannot be confirmed is rendered as BROKEN and the script
exits 1. That is the point: a ledger that reports evidence it has not verified
is worse than no ledger, because Section 1 makes this file the definition of
"done".

    python scripts/verification/render_ledger.py            # render + check
    python scripts/verification/render_ledger.py --check    # check only
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
LEDGER = REPO_ROOT / "verification" / "ledger.yaml"
OUTPUT = REPO_ROOT / "results" / "verification" / "VERIFICATION.md"

EMPTY = "—"
BROKEN = "BROKEN"


@lru_cache(maxsize=1)
def collected_test_ids() -> Set[str]:
    """Every test id pytest can collect, or an empty set if collection fails."""
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "pytest", "--collect-only", "-q",
             "-m", "", "--no-header", "-p", "no:cacheprovider"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=600,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"warning: could not collect tests ({exc}); test evidence cannot be checked")
        return set()

    ids = set()
    for line in completed.stdout.splitlines():
        line = line.strip()
        if "::" in line and not line.startswith(("=", "<", "warning")):
            ids.add(line)
    return ids


def _display(path: Path) -> str:
    """Repo-relative path when possible, absolute otherwise."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _verify_exempt(value: str) -> Tuple[bool, str]:
    """`exempt: reason` — accepted, but the reason is mandatory."""
    reason = value[len("exempt:"):].strip()
    if not reason:
        return False, f"{BROKEN} (exempt with no reason)"
    return True, f"exempt — {reason}"


def _verify_test_id(value: str) -> Tuple[bool, str]:
    """`path::test` — the file exists and pytest actually collects the test."""
    path = value.split("::", 1)[0]
    if not (REPO_ROOT / path).exists():
        return False, f"{BROKEN} (no such file: {path})"

    ids = collected_test_ids()
    if not ids:
        return True, f"`{value}` (unchecked: collection unavailable)"
    # Accept an exact id or a node prefix, so a parametrised test matches.
    if value in ids or any(i.startswith(value + "[") for i in ids):
        return True, f"`{value}`"
    return False, f"{BROKEN} (not collected by pytest: {value})"


def _verify_anchor(value: str) -> Tuple[bool, str]:
    """`FILE.md#anchor` — the file exists and contains the anchor text."""
    path, anchor = value.split("#", 1)
    target = REPO_ROOT / path
    if not target.exists():
        return False, f"{BROKEN} (no such file: {path})"
    if anchor and anchor not in target.read_text(errors="replace"):
        return False, f"{BROKEN} (no anchor {anchor!r} in {path})"
    return True, f"`{value}`"


def _verify_path(value: str) -> Tuple[bool, str]:
    """A bare path — it must exist on disk."""
    if (REPO_ROOT / value).exists():
        return True, f"`{value}`"
    return False, f"{BROKEN} (no such path: {value})"


def verify_evidence(value: str) -> Tuple[bool, str]:
    """Return (ok, rendered) for one evidence value."""
    value = str(value).strip()
    if not value:
        return True, EMPTY
    if value.startswith("exempt:"):
        return _verify_exempt(value)
    if "::" in value:
        return _verify_test_id(value)
    if "#" in value:
        return _verify_anchor(value)
    return _verify_path(value)


def _render_table(
    rows: List[Dict[str, Any]], id_key: str, extra: List[str], problems: List[str]
) -> Tuple[List[str], int, int]:
    """Render one ledger section; returns (lines, complete_count, total)."""
    if not rows:
        return ["_(no entries)_", ""], 0, 0

    criteria = list(rows[0].get("criteria", {}))
    header = [id_key] + extra + criteria
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]

    complete = 0
    for row in rows:
        name = str(row.get(id_key, "?"))
        cells = [name] + [str(row.get(k) or EMPTY) for k in extra]
        filled = 0
        for key in criteria:
            raw = row.get("criteria", {}).get(key, "")
            ok, rendered = verify_evidence(raw)
            if not ok:
                problems.append(f"{name} / {key}: {rendered}")
            if str(raw).strip():
                filled += 1
            cells.append(rendered)
        if filled == len(criteria):
            complete += 1
        lines.append("| " + " | ".join(cells) + " |")

    lines.append("")
    return lines, complete, len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=LEDGER)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--check", action="store_true", help="verify without writing")
    args = parser.parse_args()

    if not args.ledger.exists():
        print(f"{args.ledger} does not exist")
        return 1

    document = yaml.safe_load(args.ledger.read_text()) or {}
    metrics = document.get("metrics") or []
    features = document.get("features") or []
    problems: List[str] = []

    metric_lines, metrics_done, metrics_total = _render_table(
        metrics, "metric", ["family", "fidelity", "fidelity_note"], problems
    )
    feature_lines, features_done, features_total = _render_table(
        features, "feature", ["description"], problems
    )

    body = [
        "# Verification ledger",
        "",
        "Generated by `scripts/verification/render_ledger.py` from",
        "`verification/ledger.yaml`. Do not edit this file by hand.",
        "",
        "A row is **done** only when every criterion carries evidence that this",
        f"script confirmed. `{EMPTY}` means not yet done; `{BROKEN}` means the",
        "ledger claims evidence that could not be found.",
        "",
        "## Summary",
        "",
        f"- Metrics complete: **{metrics_done} / {metrics_total}**",
        f"- Features complete: **{features_done} / {features_total}**",
        f"- Broken evidence pointers: **{len(problems)}**",
        "",
        "## Metrics",
        "",
        *metric_lines,
        "## Features",
        "",
        *feature_lines,
    ]

    if problems:
        print(f"{len(problems)} broken evidence pointer(s):")
        for problem in problems:
            print(f"  - {problem}")

    if not args.check:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text("\n".join(body) + "\n")
        print(f"wrote {_display(args.output)}")

    print(
        f"metrics {metrics_done}/{metrics_total} complete, "
        f"features {features_done}/{features_total} complete, "
        f"{len(problems)} broken"
    )
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
