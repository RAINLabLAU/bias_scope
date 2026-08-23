"""Multilingual bias datasets: loaders and an honest readiness table.

New in v0.2.0 (PLAN.md 7.2).

**Nothing here is redistributed.** PLAN.md Section 12 settles it: datasets with
non-commercial or unstated licenses get "loader only, no redistribution". Every
loader reads a path the caller already has and, when it is missing, raises an
error naming the repository, the license, and the command to fetch it. That
keeps BiasScope's own license clean and keeps the provenance of each dataset
visible at the point of use.

The second purpose is `readiness()`, which produces the multilingual readiness
table PLAN.md 11.2 asks for — and, more importantly, the evidence behind each
metric's `MetricInfo.languages`. A library that claims 12 languages should be
able to say which file makes that true.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

#: A dataset whose license does not permit redistribution, or whose license is
#: unstated. Both mean the same thing for us: loader only.
NO_REDISTRIBUTION = "loader only — not redistributed with BiasScope"


@dataclass(frozen=True)
class DatasetSpec:
    """Provenance for one multilingual resource.

    Attributes:
        name: How the dataset is cited.
        languages: ISO codes the dataset actually covers, from its own files.
        license: SPDX id, or "unstated" — which is *not* the same as permissive.
        url: Where the caller gets it.
        citation: Author, year, venue.
        notes: Anything that changes how the data may be used.
    """

    name: str
    languages: Tuple[str, ...]
    license: str
    url: str
    citation: str
    notes: str = ""

    @property
    def redistributable(self) -> bool:
        """Whether BiasScope could ship the files. It does not, either way."""
        return self.license in ("MIT", "Apache-2.0", "CC-BY-4.0", "CC-BY-SA-4.0")


#: Every multilingual resource BiasScope can read. Licenses were checked against
#: each repository's own metadata on 2026-08-23, not inferred from the paper.
MULTILINGUAL_DATASETS: Dict[str, DatasetSpec] = {
    "MBBQ": DatasetSpec(
        name="MBBQ",
        languages=("en", "es", "nl", "tr"),
        license="CC-BY-4.0",
        url="https://github.com/Veranep/MBBQ",
        citation="Neplenbroek, Bisazza, Fernández 2024, COLM",
        notes=(
            "Ships a parallel `_control_` file per category, which is the point "
            "of the benchmark: the control isolates bias from the model's "
            "general ability to answer in that language."
        ),
    ),
    "KoBBQ": DatasetSpec(
        name="KoBBQ",
        languages=("ko",),
        license="MIT",
        url="https://github.com/naver-ai/KoBBQ",
        citation="Jin, Lee, Yu, Jung, Kim, Kim et al. 2024, TACL",
        notes=(
            "Culturally re-annotated rather than translated: categories were "
            "surveyed with Korean speakers, so it is not a drop-in comparison "
            "against English BBQ."
        ),
    ),
    "CBBQ": DatasetSpec(
        name="CBBQ",
        languages=("zh",),
        license="unstated",
        url="https://github.com/YFHuangxxxx/CBBQ",
        citation="Huang & Xiong 2024, LREC-COLING",
        notes=(
            "The repository states no license. Absence of a license is not "
            "permission, so this is loader-only regardless of Section 12's "
            "non-commercial rule."
        ),
    ),
    "CrowSPairsMultilingual": DatasetSpec(
        name="CrowS-Pairs multilingual (French extension)",
        languages=("en", "fr"),
        license="CC-BY-SA-4.0",
        url="https://huggingface.co/datasets/BigScienceBiasEval/crows_pairs_multilingual",
        citation="Névéol, Dupont, Bezançon, Fort 2022, ACL",
        notes=(
            "The French half is newly written rather than translated, and the "
            "paper revises some English items; treat the two halves as "
            "different datasets, not as a parallel corpus."
        ),
    ),
    "SHADES": DatasetSpec(
        name="SHADES",
        languages=(
            "ar", "bn", "zh", "nl", "en", "fr", "de", "hi", "it", "mr",
            "pl", "pt", "ro", "ru", "es",
        ),
        license="unstated",
        url="https://huggingface.co/datasets/LanguageShades/BiasShades",
        citation="Mitchell, Talat, Lauscher, van der Wal, Nangia et al. 2025, NAACL",
        notes=(
            "The dataset card declares license `other` and the repository is "
            "access-gated. Loader only, and the caller must accept the terms on "
            "the Hub first."
        ),
    ),
    "HONEST": DatasetSpec(
        name="HONEST templates",
        languages=("en", "es", "fr", "it", "pt", "ro"),
        license="MIT",
        url="https://github.com/MilaNLProc/honest",
        citation="Nozza, Bianchi, Hovy 2021, NAACL",
        notes=(
            "Six languages in the binary template set; the queer/non-queer set "
            "is English only. Counted from resources/*/ in the authors' repo."
        ),
    ),
    "CBS": DatasetSpec(
        name="CBS ethnic-bias templates",
        languages=("ko", "en", "de", "fr", "es", "zh", "ja", "tr", "ar",
                   "el", "th", "vi"),
        license="MIT",
        url="https://github.com/jaimeenahn/ethnic_bias",
        citation="Ahn & Oh 2021, EMNLP",
        notes=(
            "Twelve languages, each with its own templates, nationality list "
            "and BERT checkpoint, enumerated from the authors' "
            "`configuration.py`. The paper's `jp` key is Japanese (`ja`)."
        ),
    ),
}


def _require(path: Path, spec: DatasetSpec) -> Path:
    """Raise a message that tells the caller exactly what to fetch and why."""
    if path.exists():
        return path
    raise FileNotFoundError(
        f"{spec.name} is not at {path}. BiasScope does not redistribute it "
        f"({spec.license} — {NO_REDISTRIBUTION}). Get it from {spec.url} "
        f"({spec.citation})."
        + (f" Note: {spec.notes}" if spec.notes else "")
    )


def load_mbbq(
    root: str,
    language: str,
    category: str = "Age",
    *,
    control: bool = False,
) -> List[Dict[str, Any]]:
    """
    Read one MBBQ split from a local clone of `Veranep/MBBQ`.

    Args:
        root (str): The clone's root, or its `data/` directory.
        language (str): One of `en`, `es`, `nl`, `tr`.
        category (str): A BBQ category, e.g. "Age", "Disability_status".
        control (bool): Read the parallel `_control_` file instead. MBBQ's
            control questions have an unambiguous answer, so the gap between
            control accuracy and bias is what separates a biased model from one
            that simply answers that language badly.

    Returns:
        list[dict]: BBQ-shaped entries with `context`, `question`, `ans0..2`,
        `label`, `question_polarity`, `context_condition`, `answer_info`.

    Raises:
        ValueError: If `language` is not one of MBBQ's four.
        FileNotFoundError: If the file is absent, with the URL and license.
    """
    spec = MULTILINGUAL_DATASETS["MBBQ"]
    if language not in spec.languages:
        raise ValueError(
            f"MBBQ covers {spec.languages}, not {language!r}"
        )
    base = Path(root)
    base = base if base.name == "data" else base / "data"
    suffix = "_control" if control else ""
    path = _require(base / f"{category}{suffix}_{language}.jsonl", spec)
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_kobbq(root: str, split: str = "test") -> List[Dict[str, Any]]:
    """
    Read KoBBQ from a local clone of `naver-ai/KoBBQ`.

    `choices` is a stringified Python list in the TSV; it is parsed into a real
    list here, and `biased_answer` — KoBBQ's own field, absent from English
    BBQ — is carried through, since it is what makes the Korean set a
    culturally re-annotated benchmark rather than a translation.

    Args:
        root (str): The clone's root, or its `data/` directory.
        split (str): "test" (`KoBBQ_test_samples.tsv`) or "all".

    Raises:
        ValueError: If `split` is neither "test" nor "all".
        FileNotFoundError: If the file is absent, with the URL and license.
    """
    if split not in ("test", "all"):
        raise ValueError(f"split must be 'test' or 'all', got {split!r}")
    spec = MULTILINGUAL_DATASETS["KoBBQ"]
    base = Path(root)
    base = base if base.name == "data" else base / "data"
    path = _require(base / f"KoBBQ_{split}_samples.tsv", spec)

    rows: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            row["choices"] = _parse_choice_list(row.get("choices", ""))
            rows.append(row)
    return rows


def _parse_choice_list(text: str) -> List[str]:
    """Parse KoBBQ's ``"['a', 'b', 'c']"`` without eval()."""
    inner = str(text).strip().strip("[]")
    if not inner:
        return []
    return [item.strip().strip("'\"") for item in inner.split(",")]


def load_cbbq(
    root: str,
    category: str = "age",
    condition: str = "ambiguous",
) -> List[Dict[str, Any]]:
    """
    Read CBBQ from a local clone of `YFHuangxxxx/CBBQ`.

    Args:
        root (str): The clone's root, or its `data/` directory.
        category (str): A CBBQ category directory, e.g. "age", "SES".
        condition (str): "ambiguous" or "disambiguous" — CBBQ's spelling.

    Raises:
        ValueError: If `condition` is not one of the two.
        FileNotFoundError: If the file is absent. **CBBQ states no license**,
            so the message says so.
    """
    if condition not in ("ambiguous", "disambiguous"):
        raise ValueError(
            f"condition must be 'ambiguous' or 'disambiguous' (CBBQ's own "
            f"spelling), got {condition!r}"
        )
    spec = MULTILINGUAL_DATASETS["CBBQ"]
    base = Path(root)
    base = base if base.name == "data" else base / "data"
    path = _require(base / category / condition / f"{condition}.json", spec)
    with open(path, encoding="utf-8-sig") as handle:
        rows = json.load(handle)
    # The shipped JSON carries a BOM inside the first key name; strip it so
    # `example_id` is addressable rather than `﻿example_id`.
    return [{k.lstrip("﻿"): v for k, v in row.items()} for row in rows]


def load_crows_pairs_multilingual(
    path: str, language: str = "fr"
) -> List[Dict[str, Any]]:
    """
    Read the French/English CrowS-Pairs extension from a local JSONL file.

    Args:
        path (str): A local copy of `test_FR.jsonl` or `test_EN.jsonl`.
        language (str): "fr" or "en", recorded on each row.

    Raises:
        ValueError: If `language` is not "fr" or "en".
        FileNotFoundError: If the file is absent, with the URL and license.
    """
    spec = MULTILINGUAL_DATASETS["CrowSPairsMultilingual"]
    if language not in spec.languages:
        raise ValueError(f"the extension covers {spec.languages}, not {language!r}")
    target = _require(Path(path), spec)
    with open(target, encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    for row in rows:
        row.setdefault("language", language)
    return rows


def readiness(metric_languages: Optional[Dict[str, Sequence[str]]] = None) -> str:
    """
    Render the multilingual readiness table (PLAN.md 11.2) as markdown.

    Args:
        metric_languages: Optional ``{metric: languages}`` to append, so the
            table shows which metrics the datasets above actually unlock.

    Returns:
        str: A markdown table with one row per dataset — languages, license,
        whether it may be redistributed, and its source.
    """
    lines = [
        "| Dataset | Languages | License | Redistributable | Source |",
        "|---|---|---|---|---|",
    ]
    for spec in MULTILINGUAL_DATASETS.values():
        lines.append(
            f"| {spec.name} | {len(spec.languages)}: "
            f"{', '.join(spec.languages)} | {spec.license} | "
            f"{'yes' if spec.redistributable else 'no'} | {spec.url} |"
        )
    lines.append("")
    lines.append(
        "BiasScope redistributes none of these. Every loader reads a path the "
        "caller supplies and names the source and license when it is missing."
    )

    if metric_languages:
        lines += ["", "| Metric | Languages |", "|---|---|"]
        for metric, languages in sorted(metric_languages.items()):
            lines.append(f"| {metric} | {', '.join(languages)} |")
    return "\n".join(lines)


__all__ = [
    "DatasetSpec",
    "MULTILINGUAL_DATASETS",
    "load_mbbq",
    "load_kobbq",
    "load_cbbq",
    "load_crows_pairs_multilingual",
    "readiness",
    "NO_REDISTRIBUTION",
]
