#!/usr/bin/env python3
"""Regenerate frozen golden outputs under tests/golden/ (PLAN.md Section 1, criterion 6).

Goldens exist to catch silent numeric drift during refactors, so they are
regenerated only by running this script deliberately — never automatically, and
never to make a red test green. A golden that changes means either a bug was
fixed or a bug was introduced; the CHANGELOG entry this script demands is where
you say which.

    python scripts/verification/regen_golden.py --list
    python scripts/verification/regen_golden.py --metric WEAT --reason "..."
    python scripts/verification/regen_golden.py --all --reason "..."

Each generator is a zero-argument function returning a JSON-serialisable dict,
registered in GENERATORS below. It must be deterministic: fixed inputs, fixed
seed, tiny model or no model at all.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_DIR = REPO_ROOT / "tests" / "golden"
CHANGELOG = REPO_ROOT / "CHANGELOG.md"

sys.path.insert(0, str(REPO_ROOT / "src"))


def _golden_weat() -> Dict[str, Any]:
    """WEAT effect size on the seeded fixture vectors from tests/conftest.py."""
    import numpy as np

    from bias_scope.embeddings_based import WEAT
    from bias_scope.utils import seed_everything

    seed_everything(42)
    rng = np.random.default_rng(42)
    sets = [rng.standard_normal((8, 16)) for _ in range(4)]

    score = WEAT().evaluate((sets[0], sets[1]), (sets[2], sets[3]))
    return {
        "metric": "WEAT",
        "inputs": "np.random.default_rng(42).standard_normal((8, 16)) x4",
        "score": score,
    }


def _golden_seat() -> Dict[str, Any]:
    """SEAT on seeded fixture vectors: effect size and the >= permutation p-value.

    SEAT inherits WEAT's effect size, so ``score`` here matches WEAT's golden on
    the same inputs; ``p_value`` freezes May et al.'s non-strict convention
    (``tie_policy="conservative"``), which is what distinguishes SEAT from WEAT.
    """
    import numpy as np

    from bias_scope.embeddings_based import SEAT
    from bias_scope.utils import seed_everything

    seed_everything(42)
    rng = np.random.default_rng(42)
    sets = [rng.standard_normal((8, 16)) for _ in range(4)]

    details = SEAT().evaluate(
        (sets[0], sets[1]), (sets[2], sets[3]), return_details=True
    )
    return {
        "metric": "SEAT",
        "inputs": "np.random.default_rng(42).standard_normal((8, 16)) x4",
        "score": details["effect_size"],
        "p_value": details["p_value"],
    }


# metric name -> generator. One entry per metric with a golden; Phases 1-4 grow this.
GENERATORS: Dict[str, Callable[[], Dict[str, Any]]] = {
    "WEAT": _golden_weat,
    "SEAT": _golden_seat,
}


def regenerate(name: str, reason: str) -> Path:
    """Run one generator and write tests/golden/<name>.json."""
    payload = GENERATORS[name]()
    payload["regenerated_because"] = reason

    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    target = GOLDEN_DIR / f"{name.lower()}.json"

    if target.exists():
        previous = json.loads(target.read_text())
        if previous.get("score") != payload.get("score"):
            print(f"  {name}: score {previous.get('score')} -> {payload.get('score')}")
        else:
            print(f"  {name}: unchanged ({payload.get('score')})")

    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--metric", help="regenerate one metric's golden")
    group.add_argument("--all", action="store_true", help="regenerate every golden")
    group.add_argument("--list", action="store_true", help="list registered generators")
    parser.add_argument(
        "--reason",
        help="why this golden is being regenerated; goes into the file and must "
             "also be added to CHANGELOG.md",
    )
    args = parser.parse_args()

    if args.list:
        for name in sorted(GENERATORS):
            print(name)
        return 0

    if not args.reason:
        print("--reason is required: a golden changes only for a stated reason.")
        return 1

    if args.metric:
        if args.metric not in GENERATORS:
            print(f"no generator for {args.metric!r}; have: {', '.join(sorted(GENERATORS))}")
            return 1
        names = [args.metric]
    else:
        names = sorted(GENERATORS)

    print(f"regenerating {len(names)} golden(s):")
    for name in names:
        target = regenerate(name, args.reason)
        print(f"  wrote {target.relative_to(REPO_ROOT)}")

    print(f"\nNow add a CHANGELOG entry saying why ({CHANGELOG.name}), then commit")
    print("the golden and the CHANGELOG together so the change is reviewable.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
