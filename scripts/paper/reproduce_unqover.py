#!/usr/bin/env python3
"""Opt-in UNQOVER prediction-dump analysis; never downloads artifacts at import."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bias_scope.prompts_based._unqover_reproduction import (
    evaluate_prediction_file,
    load_gender_subject_groups,
    load_targets,
    sha256_file,
)


def _write(value: dict, output_file: Path | None) -> None:
    text = json.dumps(value, indent=2)
    if output_file is None:
        print(text)
    else:
        output_file.write_text(text + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    analyze = subparsers.add_parser("analyze", help="Analyze a caller-supplied UNQOVER output.json")
    analyze.add_argument("--predictions-file", type=Path, required=True)
    analyze.add_argument(
        "--metric", choices=("pos_err", "attr_err", "subj_bias", "model"), required=True
    )
    analyze.add_argument("--group-by", choices=("subj", "subj_act", "gender_act"))
    analyze.add_argument("--official-root", type=Path)
    analyze.add_argument("--output-file", type=Path)
    validate = subparsers.add_parser("validate-artifact", help="Record external artifact SHA-256")
    validate.add_argument("--predictions-file", type=Path, required=True)
    validate.add_argument("--output-file", type=Path)
    targets = subparsers.add_parser("inspect-targets", help="Print private target metadata")
    targets.add_argument("--output-file", type=Path)
    args = parser.parse_args(argv)

    if args.mode == "validate-artifact":
        result = {
            "path": str(args.predictions_file),
            "sha256": sha256_file(args.predictions_file),
            "source_status": "caller_supplied_unverified_official_identity",
            "upstream_checksum": None,
            "identity": "runtime hash only; official-distribution identity is not verified",
        }
    elif args.mode == "inspect-targets":
        result = load_targets()
    else:
        if args.metric == "subj_bias" and args.group_by is None:
            parser.error("--group-by is required for --metric subj_bias")
        if args.group_by == "gender_act" and args.official_root is None:
            parser.error("--official-root is required for --group-by gender_act")
        groups = (
            load_gender_subject_groups(args.official_root)
            if args.group_by == "gender_act"
            else None
        )
        result = evaluate_prediction_file(
            args.predictions_file,
            metric=args.metric,
            group_by=args.group_by,
            subject_groups=groups,
        )
    _write(result, args.output_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
