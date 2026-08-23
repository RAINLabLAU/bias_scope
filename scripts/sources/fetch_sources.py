#!/usr/bin/env python3
"""Retrieve the primary sources for a metric (PLAN.md Section 4.0).

For each requested metric entry in sources/SOURCES.yaml:
  1. downloads `paper.url` to `paper.local_pdf` under sources/papers/
  2. extracts its text alongside as `<name>.txt` with pypdf, for reading
  3. shallow-clones `code.url` at `code.sha` into `code.local_path`
  4. stamps `retrieved_on` with today's date

Downloading is not reading. This script only puts the sources on disk; the gate
is a filled-in `sections_read` and `code.files_read`, which only a human or an
agent that actually read them can write.

    python scripts/sources/fetch_sources.py --metric LPBS
    python scripts/sources/fetch_sources.py --all

sources/papers/ and third_party/ are git-ignored; SOURCES.yaml is the evidence.
"""

from __future__ import annotations

import argparse
import datetime as dt
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any, Dict, List

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "sources" / "SOURCES.yaml"
PAPERS_DIR = REPO_ROOT / "sources" / "papers"

USER_AGENT = "bias-scope-source-fetcher/0.2 (research use; contact repo maintainer)"


def _download(url: str, dest: Path) -> None:
    """Fetch `url` to `dest`, skipping if it is already there."""
    if dest.exists() and dest.stat().st_size > 0:
        print(f"    pdf: already present at {dest.relative_to(REPO_ROOT)}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"    pdf: downloading {url}")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        dest.write_bytes(response.read())
    print(f"    pdf: wrote {dest.relative_to(REPO_ROOT)}")


def _extract_text(pdf: Path) -> None:
    """Write `<pdf>.txt` next to the PDF so the method section can be read."""
    target = pdf.with_suffix(".txt")
    if target.exists():
        print(f"    txt: already present at {target.relative_to(REPO_ROOT)}")
        return
    try:
        from pypdf import PdfReader
    except ImportError:
        print("    txt: skipped, pypdf not installed (pip install -e '.[dev]')")
        return

    pages = [page.extract_text() or "" for page in PdfReader(str(pdf)).pages]
    target.write_text("\n\n".join(pages), encoding="utf-8")
    print(f"    txt: wrote {target.relative_to(REPO_ROOT)} ({len(pages)} pages)")


def _clone(url: str, sha: str, dest: Path) -> None:
    """Clone `url` and check out `sha`, so the reference code is pinned."""
    if (dest / ".git").exists():
        print(f"    code: already cloned at {dest.relative_to(REPO_ROOT)}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"    code: cloning {url}")
    subprocess.run(["git", "clone", "--quiet", url, str(dest)], check=True)
    if sha and not str(sha).startswith("<"):
        subprocess.run(["git", "-C", str(dest), "checkout", "--quiet", sha], check=True)
        print(f"    code: checked out {sha}")
    else:
        head = subprocess.run(
            ["git", "-C", str(dest), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        print(f"    code: no sha recorded; HEAD is {head} — pin it in SOURCES.yaml")


def fetch_entry(entry: Dict[str, Any], today: str) -> bool:
    """Fetch one metric's sources. Returns True if the entry was modified."""
    name = entry.get("metric", "<unnamed>")
    print(f"  {name}")
    changed = False

    paper = entry.get("paper") or {}
    url = paper.get("url")
    # `url` is recorded as the human-facing landing page; arXiv serves the PDF
    # from a sibling path, and ACL Anthology appends `.pdf` to the page URL.
    if url and "arxiv.org/abs/" in url:
        url = url.replace("/abs/", "/pdf/")
    elif url and "aclanthology.org" in url and not url.endswith(".pdf"):
        url = url.rstrip("/") + ".pdf"
    local_pdf = paper.get("local_pdf")
    if url and local_pdf and not str(url).startswith("<"):
        pdf = REPO_ROOT / local_pdf
        try:
            _download(url, pdf)
            _extract_text(pdf)
            changed = True
        except Exception as exc:  # network, 403, malformed PDF
            print(f"    pdf: FAILED ({exc}); record the failure in SOURCES.yaml notes")
    else:
        print("    pdf: skipped, no `paper.url` + `paper.local_pdf` recorded yet")

    code = entry.get("code") or {}
    if code.get("url") and code.get("local_path") and not str(code["url"]).startswith("<"):
        try:
            _clone(code["url"], code.get("sha", ""), REPO_ROOT / code["local_path"])
            changed = True
        except subprocess.CalledProcessError as exc:
            print(f"    code: FAILED ({exc}); see the obstacle playbook in PLAN.md Sec. 1")
    elif entry.get("code_status") == "none_found":
        print("    code: none_found, per the recorded search log")
    else:
        print("    code: skipped, no `code.url` + `code.local_path` recorded yet")

    if changed:
        entry["retrieved_on"] = today
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--metric", help="fetch sources for one metric by name")
    group.add_argument("--all", action="store_true", help="fetch sources for every entry")
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    args = parser.parse_args()

    document = yaml.safe_load(args.manifest.read_text()) or {}
    entries: List[Dict[str, Any]] = document.get("metrics") or []

    if args.metric:
        selected = [e for e in entries if e.get("metric") == args.metric]
        if not selected:
            known = ", ".join(sorted(str(e.get("metric")) for e in entries)) or "none"
            print(f"no entry for metric {args.metric!r} in {args.manifest} (have: {known})")
            return 1
    else:
        selected = entries

    if not selected:
        print(f"{args.manifest} has no entries yet; nothing to fetch.")
        return 0

    print(f"fetching sources for {len(selected)} metric(s):")
    today = dt.date.today().isoformat()
    changed = [fetch_entry(entry, today) for entry in selected]

    if any(changed):
        PAPERS_DIR.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(yaml.safe_dump(document, sort_keys=False, allow_unicode=True))
        print(f"updated `retrieved_on` in {args.manifest}")

    print("\nDownloading is not reading: fill in `sections_read` and `code.files_read`")
    print("from the method section and the scoring function before auditing the metric.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
