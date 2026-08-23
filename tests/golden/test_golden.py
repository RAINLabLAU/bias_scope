"""Golden outputs must not drift silently (PLAN.md Section 1, criterion 6).

Each ``tests/golden/<metric>.json`` freezes a score on a fixed tiny input. A
failure here means a refactor changed a number. That is not automatically a
bug — but it must be a *deliberate* change: regenerate with

    python scripts/verification/regen_golden.py --metric <name> --reason "..."

and add the CHANGELOG entry that command asks for. Never edit the JSON by hand
and never loosen the tolerance to make this pass.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from bias_scope.embeddings_based import WEAT
from bias_scope.utils import seed_everything

GOLDEN_DIR = Path(__file__).parent

# Goldens are exact-input, exact-seed reruns on one machine's float64, so the
# only drift this should tolerate is last-bit rounding.
TOLERANCE = 1e-12


def _load(name: str) -> dict:
    path = GOLDEN_DIR / f"{name}.json"
    if not path.exists():
        pytest.fail(
            f"{path.name} is missing; regenerate it with "
            f"scripts/verification/regen_golden.py --metric {name.upper()}"
        )
    return json.loads(path.read_text())


def test_weat_score_has_not_drifted():
    """WEAT on the seeded fixture vectors still gives the frozen score."""
    golden = _load("weat")

    seed_everything(42)
    rng = np.random.default_rng(42)
    sets = [rng.standard_normal((8, 16)) for _ in range(4)]
    score = WEAT().evaluate((sets[0], sets[1]), (sets[2], sets[3]))

    assert score == pytest.approx(golden["score"], abs=TOLERANCE), (
        "WEAT drifted from its golden value. If this change is intended, rerun "
        "scripts/verification/regen_golden.py --metric WEAT --reason '...' and "
        "add a CHANGELOG entry."
    )


def test_golden_files_record_why_they_were_generated():
    """Every golden carries the reason it was last regenerated."""
    for path in GOLDEN_DIR.glob("*.json"):
        payload = json.loads(path.read_text())
        assert payload.get("regenerated_because"), (
            f"{path.name} has no `regenerated_because`; regenerate it through "
            "scripts/verification/regen_golden.py rather than by hand."
        )
