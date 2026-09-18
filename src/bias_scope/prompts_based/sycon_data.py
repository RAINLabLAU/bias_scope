"""Offline loaders for released SYCON-Bench source data; no network access."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Dict, List

from .sycon_prompts import DEBATE_PUSHBACK

EXPECTED_ROWS = {"debate": 100, "ethical": 200, "false_presupposition": 200}


def _lines(path: Path) -> List[str]:
    values = [
        line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if not values:
        raise ValueError(f"empty SYCON source: {path}")
    return values


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _check(
    records: List[Dict[str, str]], scenario: str, paper_reproduction: bool
) -> List[Dict[str, str]]:
    if paper_reproduction and len(records) != EXPECTED_ROWS[scenario]:
        raise ValueError(
            f"official {scenario} dataset must have {EXPECTED_ROWS[scenario]} rows, "
            f"got {len(records)}"
        )
    ids = [r["id"] for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate {scenario} record id")
    return records


def load_debate(root: str | Path, *, paper_reproduction: bool = False) -> Dict[str, object]:
    repo_root = Path(root)
    root = repo_root / "debate-setting" / "data"
    # Current master renamed this directory; accept it without changing ordering.
    if not root.exists():
        root = repo_root / "debate_setting" / "data"
    q, a = root / "questions.txt", root / "arguments.txt"
    questions, arguments = _lines(q), _lines(a)
    if len(questions) != len(arguments):
        raise ValueError("Debate questions/arguments have different row counts")
    records = [
        {
            "id": f"debate-{i:03d}",
            "scenario": "debate",
            "question": x,
            "expected_stance": arguments[i],
            "follow_ups": [DEBATE_PUSHBACK] * 4,
        }
        for i, x in enumerate(questions)
    ]
    return {
        "records": _check(records, "debate", paper_reproduction),
        "source_hashes": {"questions.txt": _sha(q), "arguments.txt": _sha(a)},
    }


def load_ethical(root: str | Path, *, paper_reproduction: bool = False) -> Dict[str, object]:
    path = Path(root) / "ethical-setting" / "data" / "stereoset_intra_user_queries_api_over45.csv"
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    fields = {
        "target",
        "bias_type",
        "stereotype",
        "question",
        "question2",
        "question3",
        "question4",
        "question5",
    }
    if not rows or not fields <= set(rows[0]):
        raise ValueError("Ethical CSV missing required SYCON columns")
    records = []
    for i, row in enumerate(rows):
        if any(not row.get(k, "").strip() for k in fields):
            raise ValueError(f"Ethical row {i} has an empty required field")
        records.append(
            {
                "id": f"ethical-{i:03d}",
                "scenario": "ethical",
                "question": row["question"],
                "expected_stance": row["stereotype"],
                "follow_ups": [row[f"question{x}"] for x in range(2, 6)],
                **{k: row[k] for k in ("target", "bias_type")},
            }
        )
    return {
        "records": _check(records, "ethical", paper_reproduction),
        "source_hashes": {path.name: _sha(path)},
    }


def load_false_presupposition(
    root: str | Path, *, paper_reproduction: bool = False
) -> Dict[str, object]:
    data = Path(root) / "false-presuppositions-setting" / "data"
    q, p, c, pb = (
        data / "questions.txt",
        data / "presuppositions.txt",
        data / "corrections.txt",
        data / "push_back.csv",
    )
    questions, presuppositions, corrections = _lines(q), _lines(p), _lines(c)
    with pb.open(encoding="utf-8", newline="") as f:
        pushes = list(csv.DictReader(f))
    required = {"Question", "Pushback_1", "Pushback_2", "Pushback_3", "Pushback_4"}
    if not pushes or not required <= set(pushes[0]):
        raise ValueError("False Presupposition push_back.csv missing required columns")
    if not (len(questions) == len(presuppositions) == len(corrections) == len(pushes)):
        raise ValueError("False Presupposition parallel source files have different row counts")
    records = []
    for i, question in enumerate(questions):
        row = pushes[i]
        if row["Question"].strip() != question:
            raise ValueError(f"False Presupposition source order mismatch at row {i}")
        follows = [row[f"Pushback_{x}"].strip() for x in range(1, 5)]
        if not all(follows):
            raise ValueError(f"False Presupposition row {i} has empty pushback")
        records.append(
            {
                "id": f"false-presupposition-{i:03d}",
                "scenario": "false_presupposition",
                "question": question,
                "expected_stance": presuppositions[i],
                "correction": corrections[i],
                "follow_ups": follows,
            }
        )
    return {
        "records": _check(records, "false_presupposition", paper_reproduction),
        "source_hashes": {x.name: _sha(x) for x in (q, p, c, pb)},
    }


def load_scenario(
    root: str | Path, scenario: str, *, paper_reproduction: bool = False
) -> Dict[str, object]:
    loaders = {
        "debate": load_debate,
        "ethical": load_ethical,
        "false_presupposition": load_false_presupposition,
    }
    if scenario not in loaders:
        raise ValueError(f"unsupported SYCON scenario: {scenario!r}")
    return loaders[scenario](root, paper_reproduction=paper_reproduction)
