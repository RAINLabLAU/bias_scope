"""Private loader for caller-supplied official RealToxicityPrompts JSONL."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

PAPER_SOURCE_ROWS = 99_442
PAPER_SCOREABLE_PROMPTS = 99_016


@dataclass(frozen=True)
class RealToxicityPromptsData:
    """Ordered released records and immutable source provenance."""

    records: list[dict[str, Any]]
    sha256: str
    source_row_count: int
    scoreable_prompt_count: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _toxicity(value: Any, *, field: str, line: int) -> float | None:
    if value is None:
        return None
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"line {line}: {field} must be null or a finite value in [0, 1]")
    return float(value)


def _part(value: Any, *, name: str, line: int) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"line {line}: {name} must be an object")
    if not isinstance(value.get("text"), str):
        raise ValueError(f"line {line}: {name}.text must be a string")
    if "toxicity" not in value:
        raise ValueError(f"line {line}: {name}.toxicity is required")
    part = dict(value)
    part["toxicity"] = _toxicity(value["toxicity"], field=f"{name}.toxicity", line=line)
    return part


def load_official_prompts(  # noqa: C901
    path: str | Path,
    *,
    paper_reproduction: bool = False,
) -> RealToxicityPromptsData:
    """Load ``prompts.jsonl`` without downloading, reordering, or filtering it."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"RealToxicityPrompts source file not found: {source}")
    records: list[dict[str, Any]] = []
    identities: set[str] = set()
    with source.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                raise ValueError(f"line {line_number}: blank JSONL rows are not permitted")
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {line_number}: invalid JSON") from exc
            if not isinstance(row, Mapping):
                raise ValueError(f"line {line_number}: record must be an object")
            filename, begin, end = row.get("filename"), row.get("begin"), row.get("end")
            if not isinstance(filename, str) or not filename:
                raise ValueError(f"line {line_number}: filename must be a non-empty string")
            if type(begin) is not int or type(end) is not int:
                raise ValueError(f"line {line_number}: begin and end must be integers")
            if type(row.get("challenging")) is not bool:
                raise ValueError(f"line {line_number}: challenging must be boolean")
            identity = f"{filename}:{begin}:{end}"
            if identity in identities:
                raise ValueError(f"duplicate source identity: {identity}")
            identities.add(identity)
            record = dict(row)
            record["id"] = identity
            record["prompt"] = _part(row.get("prompt"), name="prompt", line=line_number)
            record["continuation"] = _part(
                row.get("continuation"), name="continuation", line=line_number
            )
            records.append(record)
    if not records:
        raise ValueError("RealToxicityPrompts source cannot be empty")
    if paper_reproduction and len(records) != PAPER_SOURCE_ROWS:
        raise ValueError(
            f"paper reproduction requires {PAPER_SOURCE_ROWS} source rows, got {len(records)}"
        )
    scoreable = sum(record["prompt"]["toxicity"] is not None for record in records)
    if paper_reproduction and scoreable != PAPER_SCOREABLE_PROMPTS:
        raise ValueError(
            "paper reproduction requires "
            f"{PAPER_SCOREABLE_PROMPTS} records with usable prompt toxicity, got {scoreable}"
        )
    return RealToxicityPromptsData(records, _sha256(source), len(records), scoreable)
