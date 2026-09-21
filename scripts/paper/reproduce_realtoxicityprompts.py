"""Explicit RealToxicityPrompts paper-reconstruction entry point.

Planning is offline. ``--run`` is the sole path that constructs a local model
or Perspective client.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
from typing import Any, Optional


SUPPORTED_RUN_MODEL = "gpt2"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RealToxicityPrompts paper-reconstruction runner")
    parser.add_argument("--prompts-jsonl", required=True, help="Caller-supplied official prompts.jsonl")
    parser.add_argument("--model", default="gpt2", help="Paper model identifier; only gpt2 can execute")
    parser.add_argument("--model-revision", help="Optional model revision/commit")
    parser.add_argument("--tokenizer-revision", help="Optional tokenizer revision; defaults to model revision")
    parser.add_argument("--dtype", choices=("fp32", "fp16", "bf16"), default="fp32")
    parser.add_argument("--device", help="Optional explicit Torch device")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--reconstruction-mode",
        choices=("current_perspective_reconstruction", "historical_score_parity", "substitute_scorer"),
        default="current_perspective_reconstruction",
    )
    parser.add_argument("--perspective-api-key", help="Prefer PERSPECTIVE_API_KEY; never written to artifacts")
    parser.add_argument("--tolerance", type=float, help="Optional researcher-supplied comparison tolerance")
    execute = parser.add_mutually_exclusive_group()
    execute.add_argument("--run", action="store_true", help="Construct clients and execute")
    execute.add_argument("--dry-run", action="store_true", help="Validate source data and print the offline plan")
    return parser


def _load_data(args: argparse.Namespace):
    from bias_scope.prompts_based.realtoxicityprompts_data import load_official_prompts

    return load_official_prompts(args.prompts_jsonl, paper_reproduction=True)


def build_plan(args: argparse.Namespace, data: Any) -> dict[str, Any]:
    from bias_scope.prompts_based.realtoxicityprompts_protocol import (
        GPT2_PAPER_GENERATION,
        PAPER_EVIDENCE_COMMIT,
        RTP_PROTOCOL_VERSION,
    )
    from bias_scope.prompts_based.realtoxicityprompts_reproduction import load_table2_targets

    scoreable = [row for row in data.records if row["prompt"]["toxicity"] is not None]
    toxic = sum(row["prompt"]["toxicity"] > 0.5 for row in scoreable)
    effective_tokenizer_revision = args.tokenizer_revision or args.model_revision
    target = load_table2_targets()["models"].get(args.model)
    execution_scores = data.source_row_count * GPT2_PAPER_GENERATION["samples_per_prompt"]
    table2_scores = data.scoreable_prompt_count * GPT2_PAPER_GENERATION["samples_per_prompt"]
    table2_batches = (table2_scores + 24) // 25
    execution_batches = (execution_scores + 24) // 25
    return {
        "action": "realtoxicityprompts_paper_reconstruction_plan",
        "will_execute": False,
        "dataset_path": str(Path(args.prompts_jsonl)),
        "dataset_sha256": data.sha256,
        "source_row_count": data.source_row_count,
        "scoreable_prompt_count": data.scoreable_prompt_count,
        "historical_split_counts": {"toxic": toxic, "non_toxic": len(scoreable) - toxic},
        "challenging_subset_filtered": False,
        "execution_population": "all_source_rows (released generation runner behavior)",
        "model_id": args.model,
        "model_revision": args.model_revision,
        "tokenizer_revision": effective_tokenizer_revision,
        "dtype": args.dtype,
        "device": args.device,
        "generation_settings": GPT2_PAPER_GENERATION,
        "paper_faithful_dtype": args.dtype == GPT2_PAPER_GENERATION["dtype"],
        "dtype_note": None if args.dtype == GPT2_PAPER_GENERATION["dtype"] else "non-paper-faithful dtype; --run will reject it",
        "expected_local_generations": execution_scores,
        "expected_perspective_scores_before_cache": execution_scores,
        "expected_perspective_batches_before_cache": execution_batches,
        "table2_aggregation_continuation_scores": table2_scores,
        "table2_aggregation_perspective_batches": table2_batches,
        "table2_paper_rate_limit_minimum_seconds": table2_batches,
        "table2_paper_rate_limit_minimum_hours": round(table2_batches / 3600, 2),
        "rate_limit_note": "Theoretical minimum at one 25-request batch/second; network and API overhead increase runtime.",
        "perspective": {
            "provider": "perspective_api",
            "api_version": "v1alpha1",
            "batch_size": 25,
            "retry_policy": "none (paper-era behavior)",
        },
        "reconstruction_mode": args.reconstruction_mode,
        "protocol_version": RTP_PROTOCOL_VERSION,
        "paper_era_evidence_commit": PAPER_EVIDENCE_COMMIT,
        "published_target": target,
        "tolerance": args.tolerance,
        "tolerance_policy": "differences only; no pass/fail without a user-supplied tolerance"
        if args.tolerance is None else "user-supplied tolerance controls within_tolerance only",
        "historical_warning": (
            "Current Perspective scores and modern local sampling are not expected to match "
            "the unpinned 2020 deployment exactly."
        ),
    }


def _validate_runtime(args: argparse.Namespace) -> str:
    if args.model != SUPPORTED_RUN_MODEL:
        raise RuntimeError(
            "RealToxicityPrompts --run currently supports only 'gpt2'. GPT-1 and CTRL "
            "profiles will be enabled after their exact protocol behavior is implemented and tested."
        )
    if args.reconstruction_mode != "current_perspective_reconstruction":
        raise RuntimeError(
            "--run supports only current_perspective_reconstruction. historical_score_parity "
            "needs unavailable original 2020 generation-score artifacts; substitute_scorer is an adaptation."
        )
    if args.dtype != "fp32":
        raise RuntimeError(
            "The released GPT-2 paper profile requires fp32. fp16/bf16 are non-paper "
            "configurations and cannot run under the paper-reconstruction label."
        )
    missing = [
        name
        for name in ("torch", "transformers", "googleapiclient")
        if importlib.util.find_spec(name) is None
    ]
    if missing:
        raise RuntimeError(
            "RealToxicityPrompts --run requires optional dependencies: " + ", ".join(missing)
            + ". Install bias-scope[torch,perspective] before running."
        )
    key = args.perspective_api_key or os.environ.get("PERSPECTIVE_API_KEY")
    if not key:
        raise RuntimeError("--run requires PERSPECTIVE_API_KEY or --perspective-api-key. No API call was made.")
    return key


def _run(args: argparse.Namespace, data: Any, *, generator: Any = None, scorer: Any = None) -> dict[str, Any]:
    # Model/mode safety is unconditional, including injected offline fakes.
    if args.model != SUPPORTED_RUN_MODEL:
        raise RuntimeError(
            "RealToxicityPrompts --run currently supports only 'gpt2'. GPT-1 and CTRL "
            "profiles will be enabled after their exact protocol behavior is implemented and tested."
        )
    if args.reconstruction_mode != "current_perspective_reconstruction":
        raise RuntimeError(
            "--run supports only current_perspective_reconstruction. historical_score_parity "
            "needs unavailable original 2020 generation-score artifacts; substitute_scorer is an adaptation."
        )
    if args.dtype != "fp32":
        raise RuntimeError(
            "The released GPT-2 paper profile requires fp32. fp16/bf16 are non-paper "
            "configurations and cannot run under the paper-reconstruction label."
        )
    key = _validate_runtime(args) if scorer is None or generator is None else None
    from bias_scope.backends import HuggingFaceCausalGenerator
    from bias_scope.prompts_based.realtoxicityprompts_reproduction import (
        PaperPerspectiveScorer,
        RealToxicityPromptsReproductionRunner,
        compare_to_paper,
    )

    if generator is None:
        generator = HuggingFaceCausalGenerator(
            args.model, revision=args.model_revision, tokenizer_revision=args.tokenizer_revision,
            dtype=args.dtype, device=args.device,
        )
    if scorer is None:
        scorer = PaperPerspectiveScorer(key)
    runner = RealToxicityPromptsReproductionRunner(
        generator, scorer, args.output_dir, model_id=args.model,
        reconstruction_mode=args.reconstruction_mode,
    )
    results = runner.run(data.records)
    output = Path(args.output_dir)
    dataset = {
        "path": str(Path(args.prompts_jsonl)), "sha256": data.sha256,
        "source_row_count": data.source_row_count, "execution_row_count": len(data.records),
        "scoreable_prompt_count": data.scoreable_prompt_count,
    }
    metadata = runner.metadata(dataset=dataset)
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (output / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    comparison = compare_to_paper(args.model, results, tolerance=args.tolerance)
    (output / "comparison_to_paper.json").write_text(
        json.dumps({"tolerance": args.tolerance, "quantities": comparison}, indent=2), encoding="utf-8"
    )
    return results


def main(argv: Optional[list[str]] = None, *, generator: Any = None, scorer: Any = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    if args.tolerance is not None and args.tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    data = _load_data(args)
    plan = build_plan(args, data)
    print(json.dumps(plan, indent=2))
    if not args.run:
        return plan
    return _run(args, data, generator=generator, scorer=scorer)


if __name__ == "__main__":
    main()
