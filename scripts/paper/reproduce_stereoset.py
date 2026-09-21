#!/usr/bin/env python3
"""Opt-in StereoSet paper-era evaluator; never downloads models at import time."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bias_scope.prompts_based._stereoset_reproduction import (
    evaluate_official_files,
    reconstruct_bert_base,
)


def _write(result: dict, output_file: Path | None) -> None:
    """Write JSON to an explicit output file or standard output."""
    text = json.dumps(result, indent=2)
    if output_file:
        output_file.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    parity = subparsers.add_parser("official-parity", help="Score external official predictions")
    parity.add_argument(
        "--gold-file", type=Path, required=True, help="Official pinned-repo dev.json"
    )
    parity.add_argument(
        "--predictions-file", type=Path, required=True, help="Official numeric prediction JSON"
    )
    parity.add_argument("--output-file", type=Path)
    bert = subparsers.add_parser(
        "bert-base", help="Run an opt-in historical BERT-base reconstruction"
    )
    bert.add_argument(
        "--gold-file", type=Path, required=True, help="Official pinned-repo gold JSON"
    )
    bert.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    bert.add_argument("--model-id", default="bert-base-cased")
    bert.add_argument("--output-file", type=Path)
    args = parser.parse_args(argv)
    if args.mode == "official-parity":
        result = evaluate_official_files(args.gold_file, args.predictions_file)
    else:
        gold = json.loads(args.gold_file.read_text(encoding="utf-8"))
        result = reconstruct_bert_base(gold, device=args.device, model_id=args.model_id)
    _write(result, args.output_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
