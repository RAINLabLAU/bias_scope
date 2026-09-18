"""Private, offline reconstruction helpers for OpinionQA consistency.

This module intentionally is not part of the public prompt-metric API.  It
never downloads data or queries a model: callers supply the official release.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from .opinion_consistency_across_personas import OpinionConsistencyAcrossPersonas

OFFICIAL_REPOSITORY = "https://github.com/tatsu-lab/opinions_qa"
PAPER_EVIDENCE_COMMIT = "612e5c1592803c1900b43bd832a27cddb3707f60"
PARITY_PROTOCOL_VERSION = "opinionqa-consistency-parity-v1"
PAPER_FORMULA = "paper_formula"
NOTEBOOK_RECONSTRUCTION = "official_notebook_reconstruction"
PEW_SURVEY_LIST = (26, 27, 29, 32, 34, 36, 41, 42, 43, 45, 49, 50, 54, 82, 92)


def _vector(value: str, field: str) -> list[float]:
    try:
        result = OpinionConsistencyAcrossPersonas._parse_vector(value)
        OpinionConsistencyAcrossPersonas._validate_distribution(result, field)
        return result
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {field}: {value!r}") from exc


def file_description(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    return {
        "path": str(path),
        "sha256": digest,
        "row_count": len(rows),
        "columns": reader.fieldnames or [],
    }


def load_topic_mapping(path: str | Path) -> Mapping[str, Any]:
    """Load official ``topic_mapping.npy``; numpy is a project dependency."""
    import numpy as np

    mapping = np.load(Path(path), allow_pickle=True).item()
    if not isinstance(mapping, Mapping):
        raise ValueError("Official topic_mapping.npy must contain a mapping.")
    return mapping


def load_combined_csv(  # noqa: C901
    path: str | Path,
    topic_mapping: Mapping[str, Any],
    *,
    model: str | None = None,
    granularity: str = "cg",
    source_offset: int = 0,
    survey_wave: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Parse an official processed ``*_combined.csv`` without fabricating data."""
    if granularity != "cg":
        raise ValueError("Main OpinionQA consistency parity requires coarse topics ('cg').")
    description = file_description(path)
    required = {
        "qkey",
        "question",
        "D_M",
        "D_H",
        "ordinal",
        "attribute",
        "group",
        "group_order",
        "model_name",
    }
    missing = required - set(description["columns"])
    if missing:
        raise ValueError(f"Combined artifact lacks required columns: {sorted(missing)}")
    records: list[dict[str, Any]] = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row_index, row in enumerate(csv.DictReader(handle)):
            if model is not None and row["model_name"] != model:
                continue
            info = topic_mapping.get(row["question"])
            if not info or not isinstance(info.get("cg"), list):
                raise ValueError(f"No coarse topic mapping for question {row['qkey']!r}.")
            ordinal = OpinionConsistencyAcrossPersonas._parse_vector(row["ordinal"])
            if (
                not ordinal
                or not all(math.isfinite(x) for x in ordinal)
                or max(ordinal) == min(ordinal)
            ):
                raise ValueError(f"Invalid ordinal for question {row['qkey']!r}.")
            dm, dh = (
                _vector(row["D_M"], "model distribution"),
                _vector(row["D_H"], "human distribution"),
            )
            if len(dm) != len(dh) or len(dm) != len(ordinal):
                raise ValueError(
                    f"Distribution/ordinal length mismatch for question {row['qkey']!r}."
                )
            identity = {
                key: row.get(key, "")
                for key in ("model_name", "context_type", "results_path", "model_order")
            }
            wd = row.get("WD", "")
            if wd != "":
                try:
                    wd = float(wd)
                except ValueError as exc:
                    raise ValueError(
                        f"Malformed official normalized WD for {row['qkey']!r}."
                    ) from exc
                if not math.isfinite(wd) or not 0 <= wd <= 1 + 1e-12:
                    raise ValueError(f"Invalid official normalized WD for {row['qkey']!r}.")
            for topic in info["cg"]:
                stable_id = (
                    f"W{survey_wave}:{row['qkey']}" if survey_wave is not None else row["qkey"]
                )
                records.append(
                    {
                        "row_index": source_offset + row_index,
                        "source_row_id": source_offset + row_index,
                        "survey_wave": survey_wave,
                        "original_qkey": row["qkey"],
                        "question_id": stable_id,
                        "question": row["question"],
                        "topic": str(topic),
                        "attribute": row["attribute"],
                        "group": row["group"],
                        "group_order": int(row["group_order"]),
                        "distribution_model": dm,
                        "distribution_human": dh,
                        "ordinal": ordinal,
                        "model_identity": identity,
                        "notebook_wd": wd,
                    }
                )
    if not records:
        raise ValueError("No records matched the requested model.")
    return records, description


def load_all_combined(
    data_root: str | Path,
    topic_mapping: Mapping[str, Any],
    *,
    model: str,
    allow_partial: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load all paper-era wave artifacts in ``helpers.PEW_SURVEY_LIST`` order."""
    root = Path(data_root) / "distributions"
    paths = [
        root / f"American_Trends_Panel_W{wave}_default_combined.csv" for wave in PEW_SURVEY_LIST
    ]
    missing = [str(path) for path in paths if not path.exists()]
    if missing and not allow_partial:
        raise ValueError(
            "Full official-data parity requires all PEW waves; missing: " + ", ".join(missing)
        )
    selected = [path for path in paths if path.exists()]
    if not selected:
        raise ValueError("No expected official default combined CSVs were found.")
    records, descriptions, offset = [], [], 0
    for path in selected:
        wave = int(path.name.split("_W", 1)[1].split("_", 1)[0])
        loaded, description = load_combined_csv(
            path, topic_mapping, model=model, source_offset=offset, survey_wave=wave
        )
        description["survey_wave"] = wave
        records.extend(loaded)
        descriptions.append(description)
        offset += description["row_count"]
    return records, descriptions


def reference_alignment(
    model: list[float], human: list[float], ordinal: list[float]
) -> dict[str, float]:
    """Official helpers.py normalized 1-D Wasserstein calculation."""
    if len(model) != len(human) or len(model) != len(ordinal) or len(model) < 2:
        raise ValueError("Distribution and ordinal lengths must agree.")
    if (
        any(x < 0 or not math.isfinite(x) for x in model + human)
        or not math.isclose(sum(model), 1, abs_tol=1e-8)
        or not math.isclose(sum(human), 1, abs_tol=1e-8)
    ):
        raise ValueError("Distributions must be normalized finite probabilities.")
    buckets: dict[float, list[float]] = {}
    for point, left, right in zip(ordinal, model, human):
        if not math.isfinite(point):
            raise ValueError("Ordinal coordinates must be finite.")
        pair = buckets.setdefault(float(point), [0.0, 0.0])
        pair[0] += left
        pair[1] += right
    points, left_cdf, right_cdf, wd = sorted(buckets), 0.0, 0.0, 0.0
    maximum = points[-1] - points[0]
    if maximum == 0:
        raise ValueError("Ordinal coordinates require non-zero range.")
    for left, right in zip(points, points[1:]):
        left_cdf += buckets[left][0]
        right_cdf += buckets[left][1]
        wd += abs(left_cdf - right_cdf) * (right - left)
    return {"normalized_wd": wd / maximum, "alignment": 1.0 - wd / maximum}


def _choose(scores: Mapping[str, float], orders: Mapping[str, int]) -> tuple[str, list[str]]:
    maximum = max(scores.values())
    tied = sorted(
        (group for group, value in scores.items() if value == maximum),
        key=lambda group: orders[group],
    )
    return tied[-1], tied


def _notebook_choose(
    scores: Mapping[str, float], last_rows: Mapping[str, int]
) -> tuple[str, list[str]]:
    """Approximate ``sort_values('Rep').groupby(...).last()`` for exact ties.

    Pandas does not make a scientific tie rule here.  We retain CSV row order
    and expose it as notebook-incidental behavior rather than claiming a rule.
    """
    maximum = max(scores.values())
    tied = [group for group, value in scores.items() if value == maximum]
    return max(tied, key=lambda group: last_rows[group]), tied


def reconstruct(  # noqa: C901
    records: Iterable[Mapping[str, Any]], *, mode: str
) -> dict[str, Any]:
    """Compute either stated-paper parity or literal notebook-style reconstruction."""
    if mode not in {PAPER_FORMULA, NOTEBOOK_RECONSTRUCTION}:
        raise ValueError(f"Unknown parity mode: {mode!r}")
    rows = list(records)
    identities = {
        (
            r["model_identity"]["model_name"],
            r["model_identity"]["model_order"],
            r["model_identity"]["context_type"],
        )
        for r in rows
    }
    if len(identities) != 1:
        raise ValueError("Reconstruct one model_name/model_order/context identity at a time.")
    model_name, model_order, context = next(iter(identities))
    if context not in ("", "default"):
        raise ValueError("Official default-consistency parity requires default context.")
    result_paths = sorted({r["model_identity"]["results_path"] for r in rows})
    by_attribute: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_attribute[str(row["attribute"])].append(row)
    attributes: dict[str, Any] = {}
    excluded: list[dict[str, Any]] = []
    for attribute, attr_rows in by_attribute.items():
        groups = {r["group"]: r["group_order"] for r in attr_rows}
        last_rows = {r["group"]: r["row_index"] for r in attr_rows}
        topics: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
        topic_questions: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
        source_scores: dict[str, dict[int, float]] = defaultdict(dict)
        for row in attr_rows:  # preserves official CSV ordering for notebook-incidental ties
            calculated = reference_alignment(
                row["distribution_model"], row["distribution_human"], row["ordinal"]
            )
            if mode == NOTEBOOK_RECONSTRUCTION:
                if row["notebook_wd"] == "":
                    raise ValueError(
                        "Official notebook reconstruction requires normalized WD in "
                        "every combined CSV row."
                    )
                score = 1.0 - float(row["notebook_wd"])
                excluded.append(
                    {
                        "question_id": row["question_id"],
                        "official_csv_wd": row["notebook_wd"],
                        "recomputed_wd": calculated["normalized_wd"],
                        "absolute_difference": abs(
                            float(row["notebook_wd"]) - calculated["normalized_wd"]
                        ),
                    }
                )
            else:
                score = calculated["alignment"]
            topics[row["topic"]][row["group"]][row["question_id"]] = score
            topic_questions[row["topic"]][row["group"]].add(row["question_id"])
            source_scores[row["group"]][row["source_row_id"]] = score
        topic_results = {}
        for topic, values in topics.items():
            common_questions = (
                set.intersection(*(set(values.get(group, {})) for group in groups))
                if groups
                else set()
            )
            if mode == PAPER_FORMULA and not common_questions:
                excluded.append(
                    {"attribute": attribute, "topic": topic, "reason": "missing_subgroup_topic"}
                )
                continue
            if mode == PAPER_FORMULA:
                for group in groups:
                    for question_id in set(values.get(group, {})) - common_questions:
                        excluded.append(
                            {
                                "attribute": attribute,
                                "topic": topic,
                                "question_id": question_id,
                                "reason": "missing_subgroup_distribution",
                            }
                        )
                reps = {
                    group: sum(values[group][q] for q in common_questions) / len(common_questions)
                    for group in groups
                }
            else:
                reps = {
                    group: sum(scores.values()) / len(scores) for group, scores in values.items()
                }
            best, ties = (
                _choose(reps, groups)
                if mode == PAPER_FORMULA
                else _notebook_choose(reps, last_rows)
            )
            topic_results[topic] = {
                "representativeness": reps,
                "best_subgroup": best,
                "tied_best_subgroups": ties,
                "num_questions": len(common_questions)
                if mode == PAPER_FORMULA
                else len(set().union(*topic_questions[topic].values())),
                "num_source_rows": sum(map(len, values.values())),
                "subgroup_question_counts": {
                    group: len(question_ids)
                    for group, question_ids in topic_questions[topic].items()
                },
            }
        if not topic_results:
            continue
        if mode == PAPER_FORMULA:
            overall = {
                group: sum(item["representativeness"][group] for item in topic_results.values())
                / len(topic_results)
                for group in groups
            }
            policy = "equal coarse-topic mean; Overall excluded from final demographic mean"
        else:
            # consistency.ipynb's `grouped`: mean WD over each unexpanded
            # combined-CSV source row. Source identity avoids topic expansion
            # and deliberately does not collapse duplicate qkeys.
            overall = {
                group: sum(scores.values()) / len(scores) for group, scores in source_scores.items()
            }
            policy = (
                "question-row mean before topic expansion; CSV-order-dependent "
                "groupby().last() ties"
            )
        overall_best, overall_ties = (
            _choose(overall, groups)
            if mode == PAPER_FORMULA
            else _notebook_choose(overall, last_rows)
        )
        attributes[attribute] = {
            "topics": topic_results,
            "overall_representativeness": overall,
            "overall_best_subgroup": overall_best,
            "overall_tied_best_subgroups": overall_ties,
            "consistency": sum(
                item["best_subgroup"] == overall_best for item in topic_results.values()
            )
            / len(topic_results),
            "num_topics": len(topic_results),
        }
    included = (
        attributes
        if mode == NOTEBOOK_RECONSTRUCTION
        else {key: value for key, value in attributes.items() if key != "Overall"}
    )
    if not included:
        raise ValueError("No attributes remain after parity-mode filtering.")
    return {
        "parity_mode": mode,
        "topic_granularity": "cg",
        "protocol_version": PARITY_PROTOCOL_VERSION,
        "model_identity": {
            "model_name": model_name,
            "model_order": model_order,
            "context_type": context,
            "results_paths": result_paths,
        },
        "attributes": attributes,
        "included_attributes": list(included),
        "final_consistency": sum(value["consistency"] for value in included.values())
        / len(included),
        "excluded": excluded,
        "policy": policy,
        "tie_policy": (
            "highest group_order (paper formula) or pandas/order-dependent "
            "notebook-incidental behavior (notebook reconstruction)"
        ),
    }


def compare_with_public_metric(
    records: Iterable[Mapping[str, Any]], reference: Mapping[str, Any]
) -> dict[str, Any]:
    """Run the strict public metric on the same official distributions."""
    models, humans, seen_models, seen_humans = [], [], set(), set()
    for row in records:
        model_key = (row["question_id"], row["topic"])
        if model_key not in seen_models:
            models.append(
                {
                    "question_id": row["question_id"],
                    "distribution": row["distribution_model"],
                    "ordinal": row["ordinal"],
                    "topic": row["topic"],
                    "model_name": row["model_identity"]["model_name"],
                    "context": "default",
                    "run_id": "official-opinionqa-default",
                }
            )
            seen_models.add(model_key)
        human_key = (row["question_id"], row["attribute"], row["group"])
        if row["group"] != "Overall" and human_key not in seen_humans:
            humans.append(
                {
                    "question_id": row["question_id"],
                    "attribute": row["attribute"],
                    "subgroup": row["group"],
                    "subgroup_order": row["group_order"],
                    "distribution": row["distribution_human"],
                }
            )
            seen_humans.add(human_key)
    public = OpinionConsistencyAcrossPersonas().evaluate(models, humans, return_details=True)
    return {
        "parity_mode": PAPER_FORMULA,
        "reference_final_consistency": reference["final_consistency"],
        "biasscope_final_consistency": public["opinion_consistency"],
        "absolute_difference": abs(reference["final_consistency"] - public["opinion_consistency"]),
        "biasscope_result": public,
    }


def build_manifest(
    *,
    combined: list[Mapping[str, Any]],
    topic_mapping_path: str | Path,
    result: Mapping[str, Any],
    records: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    topic_path = Path(topic_mapping_path).resolve()
    topic_mapping = load_topic_mapping(topic_path)
    topic_schema = sorted(
        {key for value in topic_mapping.values() if isinstance(value, Mapping) for key in value}
    )
    rows = list(records)
    return {
        "official_repository": OFFICIAL_REPOSITORY,
        "paper_evidence_commit": PAPER_EVIDENCE_COMMIT,
        "protocol_version": PARITY_PROTOCOL_VERSION,
        "parity_mode": result["parity_mode"],
        "topic_granularity": "cg",
        "model_identity": result["model_identity"],
        "consumed_combined_csvs": combined,
        "topic_mapping": {
            "path": str(topic_path),
            "sha256": hashlib.sha256(topic_path.read_bytes()).hexdigest(),
            "item_count": len(topic_mapping),
            "schema": topic_schema,
        },
        "total_file_rows": sum(item["row_count"] for item in combined),
        "selected_model_source_rows": len({row["source_row_id"] for row in rows}),
        "total_topic_expanded_records": len(rows),
        "demographic_attributes": sorted({r["attribute"] for r in rows}),
        "coarse_topics": sorted({r["topic"] for r in rows}),
        "survey_wave_order": [item["survey_wave"] for item in combined],
    }


def write_json(path: str | Path, value: Mapping[str, Any]) -> None:
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
