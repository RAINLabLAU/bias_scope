#!/usr/bin/env python3
"""Flip a SOURCES.yaml entry from `pending` to `read` (PLAN.md Section 4.0).

Call this only after actually reading the method section and the scoring code.
`check_manifest.py` refuses `status: read` without `sections_read` and
`code.files_read`, so the evidence is not optional.

    python scripts/sources/mark_read.py LPBS \
        --sections "Sec. 2 (four-step procedure)" "Sec. 2 (log probability bias score)" \
        --files "lib/bias_calculator.py:32-74" \
        --note "Paper and code disagree on the prior mask position."
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "sources" / "SOURCES.yaml"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metric")
    parser.add_argument("--sections", nargs="+", required=True,
                        help="sections/equations/tables actually read")
    parser.add_argument("--files", nargs="*", default=[],
                        help="file:line ranges of the scoring code actually read")
    parser.add_argument("--version-read", default="v1")
    parser.add_argument("--note", default="")
    parser.add_argument("--fidelity-note", default="",
                        help="path to docs/fidelity/<metric>.md, recorded so the "
                             "index can match notes to metrics without guessing")
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    args = parser.parse_args()

    text = args.manifest.read_text()
    header, _, body = text.partition("metrics:")
    document = yaml.safe_load("metrics:" + body)
    entries = document["metrics"]

    match = [e for e in entries if e.get("metric") == args.metric]
    if not match:
        print(f"no entry for {args.metric!r}")
        return 1
    entry = match[0]

    entry["status"] = "read"
    entry["sections_read"] = list(args.sections)
    if "code" in entry:
        if not args.files:
            print(f"{args.metric} has a `code` block; --files is required")
            return 1
        entry["code"]["files_read"] = list(args.files)
    entry["paper"]["version_read"] = args.version_read
    entry["retrieved_on"] = entry.get("retrieved_on") or dt.date.today().isoformat()
    if args.note:
        entry["notes"] = args.note
    if args.fidelity_note:
        path = REPO_ROOT / args.fidelity_note
        if not path.exists():
            print(f"{args.fidelity_note} does not exist")
            return 1
        entry["fidelity_note"] = args.fidelity_note

    args.manifest.write_text(
        header + yaml.safe_dump(document, sort_keys=False, allow_unicode=True, width=100)
    )
    print(f"{args.metric}: status -> read ({len(args.sections)} sections, "
          f"{len(args.files)} code file(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
