"""Private loader for the preserved TruthfulQA ACL-2022 v0 artifacts."""
from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

V0_ROWS = 817
V0_CATEGORIES = 38
V0_COLUMNS = ("Type", "Category", "Question", "Best Answer", "Correct Answers", "Incorrect Answers", "Source")
V0_DATASET_SHA256 = "f4fcc4a841d4474c46a4719c295c6df5f12eef14c187fbb9637a29e70d9ece00"
V0_MC_TASK_SHA256 = "131e6f4156297bee09b38279e2fa847e4359c200504eca99ea3fab4b2d3a6144"


@dataclass(frozen=True)
class TruthfulQAQuestion:
    question_id: str
    row_index: int
    type: str
    category: str
    question: str
    best_answer: str
    correct_answers: tuple[str, ...]
    incorrect_answers: tuple[str, ...]
    source: str


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def locate_v0(data_root: str | Path) -> Path:
    root = Path(data_root)
    candidates = (root / "data" / "v0" / "TruthfulQA.csv", root / "TruthfulQA.csv")
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("Expected caller-supplied data/v0/TruthfulQA.csv; no download is attempted.")


def _answers(value: str, field: str) -> tuple[str, ...]:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string.")
    result = tuple(part.strip() for part in value.split(";") if part.strip())
    if not result:
        raise ValueError(f"{field} must contain at least one answer.")
    return result


def load_v0_questions(data_root: str | Path, *, verify_hash: bool = True) -> tuple[list[TruthfulQAQuestion], dict[str, Any]]:
    path = locate_v0(data_root)
    digest = sha256_file(path)
    if verify_hash and digest != V0_DATASET_SHA256:
        raise ValueError("TruthfulQA paper reproduction requires the 817-row data/v0 artifact; hash mismatch rejects later or altered data.")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = tuple(reader.fieldnames or ())
        if columns != V0_COLUMNS:
            raise ValueError(f"Unexpected TruthfulQA v0 schema: {columns!r}")
        records: list[TruthfulQAQuestion] = []
        for index, row in enumerate(reader):
            correct = _answers(row["Correct Answers"], "Correct Answers")
            incorrect = _answers(row["Incorrect Answers"], "Incorrect Answers")
            best = row["Best Answer"]
            if not best or best not in correct:
                raise ValueError(f"Row {index}: Best Answer must occur exactly in Correct Answers.")
            content = "\x1f".join(str(row[name]) for name in V0_COLUMNS)
            qid = f"v0:{index}:{hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]}"
            records.append(TruthfulQAQuestion(qid, index, row["Type"], row["Category"], row["Question"], best, correct, incorrect, row["Source"]))
    if len(records) != V0_ROWS or len({r.category for r in records}) != V0_CATEGORIES:
        raise ValueError("Not the paper-era 817-question / 38-category TruthfulQA v0 dataset.")
    return records, {"source_path": str(path), "sha256": digest, "row_count": len(records), "schema": list(columns), "dataset_version": "v0"}


def validate_v0_mc_task(data_root: str | Path, *, verify_hash: bool = True) -> dict[str, Any]:
    root = Path(data_root)
    candidates = (root / "data" / "v0" / "mc_task.json", root / "mc_task.json")
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        raise FileNotFoundError("Expected caller-supplied data/v0/mc_task.json; no download is attempted.")
    digest = sha256_file(path)
    if verify_hash and digest != V0_MC_TASK_SHA256:
        raise ValueError("TruthfulQA MC task hash mismatch; refusing non-v0/binary-MC data.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or len(payload) != V0_ROWS:
        raise ValueError("v0 mc_task.json must be a list of 817 items.")
    for row in payload:
        if not isinstance(row, dict) or not isinstance(row.get("question"), str) or not isinstance(row.get("mc1_targets"), dict) or not isinstance(row.get("mc2_targets"), dict):
            raise ValueError("Malformed v0 mc_task.json item.")
    return {"source_path": str(path), "sha256": digest, "row_count": len(payload), "schema": ["question", "mc1_targets", "mc2_targets"], "dataset_version": "v0"}
