#!/usr/bin/env python3
"""Validate sources/SOURCES.yaml (PLAN.md Section 4.0). Runs in CI.

Checks, per entry:
  - `metric` is present and unique
  - `status` is `pending` or `read`
  - `paper` has a title and a url or arxiv id, unless `paper_status` says why not
  - `sections_read` is non-empty **once `status: read`** — reading the abstract
    is not reading the paper
  - either a `code` block with url + sha + license, or `code_status: none_found`
    with a non-empty `search_log`
  - every entry in `resources` has a name, a source, and a sha256

`status` exists because retrieval and reading are separate acts. An entry is
created with the confirmed identifiers at `pending`; it becomes `read` only when
someone has actually read the method section and the scoring code and filled in
`sections_read` and `code.files_read`. CI enforces the schema from the moment
the entry exists, and the audit is what flips the status. The count of `pending`
entries is printed so a half-finished audit cannot look like a passing one.

Not yet checked: that every metric class carrying a `MetricInfo` has an entry
here. `MetricInfo` arrives in Phase 2 (PLAN.md 5.1).

Exit status: 0 if the manifest is valid, 1 otherwise.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "sources" / "SOURCES.yaml"

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


VALID_STATUS = ("pending", "read")
VALID_PAPER_STATUS = ("no_cited_source", "not_located")


def _check_paper(paper: Any, problems: List[str], where: str) -> None:
    if not isinstance(paper, dict):
        problems.append(f"{where}: `paper` must be a mapping")
        return

    paper_status = paper.get("paper_status")
    if paper_status is not None:
        if paper_status not in VALID_PAPER_STATUS:
            problems.append(
                f"{where}: `paper.paper_status` must be one of {VALID_PAPER_STATUS}, "
                f"got {paper_status!r}"
            )
        return

    if not paper.get("title"):
        problems.append(f"{where}: `paper.title` is required")
    if not (paper.get("url") or paper.get("arxiv")):
        problems.append(
            f"{where}: `paper` needs a `url` or an `arxiv` id, or a `paper_status` "
            f"from {VALID_PAPER_STATUS} saying why there is none"
        )


def _check_code(entry: Dict[str, Any], problems: List[str], where: str) -> None:
    code = entry.get("code")
    status = entry.get("code_status")

    if code is None and status is None:
        problems.append(
            f"{where}: needs a `code` block, or `code_status: none_found` with a "
            "`search_log` recording the six searches of Section 4.0"
        )
        return

    if status is not None:
        if status != "none_found":
            problems.append(f"{where}: `code_status` must be `none_found`, got {status!r}")
        if not entry.get("search_log"):
            problems.append(f"{where}: `code_status: none_found` requires a `search_log`")
        return

    if not isinstance(code, dict):
        problems.append(f"{where}: `code` must be a mapping")
        return
    for field in ("url", "sha", "license"):
        if not code.get(field):
            problems.append(f"{where}: `code.{field}` is required (pin the commit)")


def _check_resources(entry: Dict[str, Any], problems: List[str], where: str) -> None:
    resources = entry.get("resources") or []
    if not isinstance(resources, list):
        problems.append(f"{where}: `resources` must be a list")
        return
    for i, resource in enumerate(resources):
        at = f"{where}: resources[{i}]"
        if not isinstance(resource, dict):
            problems.append(f"{at}: must be a mapping")
            continue
        if not resource.get("name"):
            problems.append(f"{at}: `name` is required")
        if not resource.get("source"):
            problems.append(f"{at}: `source` is required")
        sha = str(resource.get("sha256", ""))
        if not SHA256_RE.match(sha):
            problems.append(f"{at}: `sha256` must be 64 lowercase hex chars, got {sha!r}")


def _check_status(entry: Dict[str, Any], problems: List[str], where: str) -> None:
    """`status` is the audit gate: `read` demands the evidence of having read."""
    status = entry.get("status")
    if status not in VALID_STATUS:
        problems.append(f"{where}: `status` must be one of {VALID_STATUS}, got {status!r}")
        return
    if status != "read":
        return

    if not entry.get("sections_read"):
        problems.append(
            f"{where}: `status: read` requires a non-empty `sections_read` "
            "(record the sections and equations actually read)"
        )
    code = entry.get("code")
    if isinstance(code, dict) and not code.get("files_read"):
        problems.append(
            f"{where}: `status: read` requires `code.files_read` "
            "(record the file:line ranges of the scoring function you read)"
        )


def _check_entry(entry: Any, index: int, seen: set, problems: List[str]) -> None:
    """Validate one manifest entry, appending to `problems`."""
    where = f"metrics[{index}]"
    if not isinstance(entry, dict):
        problems.append(f"{where}: must be a mapping")
        return

    name = entry.get("metric")
    if not name:
        problems.append(f"{where}: `metric` is required")
    else:
        where = f"metrics[{index}] ({name})"
        if name in seen:
            problems.append(f"{where}: duplicate entry")
        seen.add(name)

    _check_status(entry, problems, where)
    _check_paper(entry.get("paper"), problems, where)
    _check_code(entry, problems, where)
    _check_resources(entry, problems, where)


def check_manifest(path: Path) -> List[str]:
    """Return a list of problems; empty means the manifest is valid."""
    if not path.exists():
        return [f"{path} does not exist"]

    document = yaml.safe_load(path.read_text()) or {}
    if not isinstance(document, dict) or "metrics" not in document:
        return [f"{path}: top level must be a mapping with a `metrics` key"]

    entries = document["metrics"] or []
    if not isinstance(entries, list):
        return [f"{path}: `metrics` must be a list"]

    problems: List[str] = []
    seen: set = set()
    for i, entry in enumerate(entries):
        _check_entry(entry, i, seen, problems)
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=f"path to SOURCES.yaml (default: {DEFAULT_MANIFEST})",
    )
    args = parser.parse_args()

    problems = check_manifest(args.manifest)
    if problems:
        print(f"{args.manifest}: {len(problems)} problem(s)")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    document = yaml.safe_load(args.manifest.read_text()) or {}
    entries = document.get("metrics") or []
    count = len(entries)
    read = sum(1 for e in entries if e.get("status") == "read")
    no_code = sum(1 for e in entries if e.get("code_status") == "none_found")

    print(f"{args.manifest}: OK, {count} metric entr{'y' if count == 1 else 'ies'}.")
    print(f"  read: {read}/{count}   pending: {count - read}   no reference code: {no_code}")
    if read < count:
        print(
            "  note: a `pending` entry means the sources are recorded, NOT that they "
            "have been read. PLAN.md Section 4.0 forbids auditing a metric before its "
            "status here is `read`."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
