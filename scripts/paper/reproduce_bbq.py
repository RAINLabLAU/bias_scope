"""Local-only BBQ released-artifact validation and scoring CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bias_scope.prompts_based._bbq_reproduction import (
    PAPER_EVIDENCE_COMMIT,
    artifact_provenance,
    attach_predictions,
    join_additional_metadata,
    load_additional_metadata,
    load_official_data,
    roberta_deberta_predictions,
    score_predictions,
    unifiedqa_predictions,
)


def main() -> None:  # noqa: C901 - CLI modes intentionally share provenance setup.
    parser = argparse.ArgumentParser(
        description="Score caller-supplied BBQ artifacts; never downloads."
    )
    parser.add_argument(
        "mode",
        choices=(
            "validate-data",
            "score-predictions",
            "score-unifiedqa",
            "score-roberta-deberta",
            "inspect-targets",
        ),
    )
    parser.add_argument("--data-root")
    parser.add_argument("--metadata-file")
    parser.add_argument("--predictions-file")
    parser.add_argument("--results-file")
    parser.add_argument("--model")
    parser.add_argument("--category")
    parser.add_argument("--context-condition", choices=("ambig", "disambig"))
    parser.add_argument("--output-file")
    args = parser.parse_args()
    if args.mode == "inspect-targets":
        output = json.loads(
            (
                Path(__file__).parents[2]
                / "src/bias_scope/prompts_based/bbq_targets.json"
            ).read_text()
        )
    else:
        if not args.data_root or not args.metadata_file:
            parser.error("--data-root and --metadata-file are required")
        rows, data_provenance = load_official_data(args.data_root)
        metadata, metadata_provenance = load_additional_metadata(args.metadata_file)
        joined, removed_metadata = join_additional_metadata(rows, metadata)
        output = {
            "protocol_commit": PAPER_EVIDENCE_COMMIT,
            "data_provenance": data_provenance,
            "metadata_provenance": metadata_provenance,
            "count_removed_missing_target_loc": removed_metadata,
        }
        if args.mode == "validate-data":
            output["count_joined"] = len(joined)
        elif args.mode == "score-predictions":
            if not args.predictions_file:
                parser.error("--predictions-file is required")
            supplied = json.loads(
                Path(args.predictions_file).read_text(encoding="utf-8")
            )
            if not isinstance(supplied, list):
                parser.error("--predictions-file must be a JSON list")
            attached, unresolved = attach_predictions(joined, supplied)
            output.update(
                {
                    "predictions_provenance": artifact_provenance(
                        args.predictions_file
                    ),
                    "count_unresolved": unresolved,
                    "scores": score_predictions(_filtered(attached, args)),
                }
            )
        elif args.mode == "score-unifiedqa":
            if not args.model:
                parser.error("--model is the UnifiedQA prediction field")
            output["scores"] = score_predictions(
                _filtered(unifiedqa_predictions(joined, args.model), args)
            )
        else:
            if not args.results_file:
                parser.error("--results-file is required")
            predictions, result_provenance = roberta_deberta_predictions(
                args.results_file
            )
            attached, unresolved = attach_predictions(joined, predictions)
            output.update(
                {
                    "results_provenance": result_provenance,
                    "count_unresolved": unresolved,
                    "scores": score_predictions(_filtered(attached, args)),
                }
            )
    text = json.dumps(output, indent=2, sort_keys=True)
    if args.output_file:
        Path(args.output_file).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


def _filtered(rows: list[dict], args: argparse.Namespace) -> list[dict]:
    """Apply optional category/context filters without creating a universal score."""
    return [
        row
        for row in rows
        if (args.category is None or row["category"] == args.category)
        and (
            args.context_condition is None
            or row["context_condition"] == args.context_condition
        )
    ]


if __name__ == "__main__":
    main()
