"""Retrieve a metric's vendored sources on demand, before a loader needs them.

`third_party/` is git-ignored, so on a fresh clone every dataset a loader reads
is absent. Until now the loader raised with the command a human should run
(`scripts/sources/fetch_sources.py --metric <name>`), which meant the agent had
to stop mid-conversation and hand the work back. Here the harness runs that
same command itself, for the one manifest entry whose file is missing, and the
loader continues if it worked. The error message is unchanged when it did not:
downloading can fail, and a wrong number is worse than a stopped run.

Only paths under this repository's own `third_party/code` are ever fetched. A
test that points a loader at a temporary tree must not reach the network, and
an installed wheel has no manifest to fetch from. Set
`BIASSCOPE_AGENT_AUTO_FETCH=0` to turn it off and get the old behaviour back.

Downloading is still not reading: this only puts the authors' own releases on
disk. PLAN.md Section 4.0's gate is `sections_read`, which no script can fill.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Set

REPO_ROOT = Path(__file__).resolve().parents[2]
VENDORED_ROOT = REPO_ROOT / "third_party" / "code"
FETCH_SCRIPT = REPO_ROOT / "scripts" / "sources" / "fetch_sources.py"

#: The manifest entries the dataset loaders read files from — the `metric`
#: hints they pass to `datasets_common._require`. Listed here because the
#: preflight has to know them before any loader has run; a test asserts this
#: stays in step with the `_require` calls themselves.
DATASET_SOURCE_METRICS = (
    "CAT",
    "CrowSPairs",
    "DecodingTrustStereotype",
    "DemographicRepresentation",
    "GenderPolarity",
    "HONEST",
    "RegardScore",
    "SEAT",
    "WEAT",
    "WinoBias",
)

#: Metrics already tried this process. A fetch that failed once (no network, a
#: 403, a renamed upstream path) fails the same way every time, and retrying it
#: per file would turn one clear error into a very slow one.
_ATTEMPTED: Set[str] = set()


def reset_attempts() -> None:
    """Forget which metrics were already tried. For tests and long sessions."""
    _ATTEMPTED.clear()


def auto_fetch_enabled() -> bool:
    return os.environ.get("BIASSCOPE_AGENT_AUTO_FETCH", "1").strip().lower() not in {
        "0",
        "false",
        "no",
    }


def _is_vendored(path: Path) -> bool:
    """True if `path` is one of this repo's own git-ignored vendored files."""
    try:
        Path(path).resolve().relative_to(VENDORED_ROOT.resolve())
    except ValueError:
        return False
    return True


def _fetch_metric(metric: str) -> bool:
    """Run the documented fetch command for one manifest entry."""
    if not FETCH_SCRIPT.exists():
        return False
    print(f"[bias-scope] fetching sources for {metric} (third_party/ is git-ignored)")
    result = subprocess.run(
        [sys.executable, str(FETCH_SCRIPT), "--metric", metric],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"[bias-scope] fetch for {metric} failed: {result.stderr.strip()[:400]}")
    return result.returncode == 0


def ensure_metric_sources(
    metric: str,
    path: Optional[Path] = None,
    *,
    runner: Optional[Callable[[str], bool]] = None,
) -> bool:
    """Fetch `metric`'s sources if `path` is a vendored file that is missing.

    Returns True only when a fetch was actually attempted, so the caller can
    re-check the path rather than trust a promise.
    """
    if path is not None and (Path(path).exists() or not _is_vendored(path)):
        return False
    if not auto_fetch_enabled() or metric in _ATTEMPTED:
        return False
    _ATTEMPTED.add(metric)
    (runner or _fetch_metric)(metric)
    return True


def ensure_dataset_sources(
    metrics: Optional[Sequence[str]] = None,
    *,
    runner: Optional[Callable[[str], bool]] = None,
) -> Dict[str, bool]:
    """Preflight every manifest entry the dataset loaders read from.

    Called before a session starts, so the datasets are on disk before the
    agent plans with them rather than at the first file a metric happens to
    open. Each entry is skipped in seconds when it is already there.
    """
    wanted = tuple(metrics) if metrics else DATASET_SOURCE_METRICS
    fetch = runner or _fetch_metric
    done: Dict[str, bool] = {}
    for metric in wanted:
        if metric in _ATTEMPTED or not auto_fetch_enabled():
            done[metric] = False
            continue
        _ATTEMPTED.add(metric)
        done[metric] = bool(fetch(metric))
    return done
