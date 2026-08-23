#!/usr/bin/env python3
"""Record the environment to results/validation/environment.json (PLAN.md Section 3).

Every number in results/ was produced by some version of some library. This
writes down which, so a reproduction that disagrees has somewhere to look
first. Run it whenever the environment changes, and once before any validation
run.

    python scripts/validation/record_environment.py
    python scripts/validation/record_environment.py --print
"""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT = REPO_ROOT / "results" / "validation" / "environment.json"

# Packages whose version can move a metric's number.
PACKAGES = [
    "numpy", "scipy", "torch", "transformers", "tokenizers", "datasets",
    "sentence_transformers", "litellm", "gensim", "pandas", "sklearn", "yaml",
]


def _version(name: str) -> Optional[str]:
    """Installed version of `name`, or None if it is not installed."""
    try:
        module = importlib.import_module(name)
    except Exception:
        return None
    version = getattr(module, "__version__", None)
    return str(version) if version else "installed (no __version__)"


def _torch_devices() -> Dict[str, Any]:
    """CUDA availability and device names; dtype choices depend on these."""
    try:
        import torch
    except ImportError:
        return {"available": False}

    info: Dict[str, Any] = {
        "available": torch.cuda.is_available(),
        "cuda_version": getattr(torch.version, "cuda", None),
    }
    if torch.cuda.is_available():
        info["devices"] = [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ]
    return info


def _git_commit() -> Optional[str]:
    """The commit the results were produced at, if this is a git checkout."""
    try:
        completed = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None


def collect() -> Dict[str, Any]:
    """Build the environment record."""
    try:
        from bias_scope import __version__ as bias_scope_version
    except Exception:
        bias_scope_version = None

    return {
        "python": {
            "version": sys.version.split()[0],
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "bias_scope": {"version": bias_scope_version, "git_commit": _git_commit()},
        "packages": {name: _version(name) for name in PACKAGES},
        "cuda": _torch_devices(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--print", dest="print_only", action="store_true",
                        help="print the record without writing it")
    args = parser.parse_args()

    record = collect()
    rendered = json.dumps(record, indent=2, sort_keys=True)

    if args.print_only:
        print(rendered)
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
