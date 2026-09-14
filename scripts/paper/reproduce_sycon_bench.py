"""Explicit research entry point for SYCON-Bench paper reproduction.

``--help`` and planning are offline.  Model/API clients are built only for an
intentional ``--run`` invocation.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
from typing import Any, Optional


SCENARIOS = ("debate", "ethical", "false_presupposition")
SUPPORTED_RUN_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Explicit SYCON-Bench paper-reproduction runner"
    )
    parser.add_argument("--sycon-root", required=True, help="Local SYCON-Bench repository root")
    parser.add_argument("--model", required=True, help="Hugging Face causal-model identifier")
    parser.add_argument("--model-revision", help="Optional model revision/commit; unpinned when omitted")
    parser.add_argument(
        "--tokenizer-revision", help="Optional tokenizer revision/commit; defaults to model revision"
    )
    parser.add_argument("--dtype", choices=("fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--quantization", choices=("none", "4bit", "8bit"), default="none")
    parser.add_argument("--scenario", choices=(*SCENARIOS, "all"), default="all")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--judge-api-key", help="GPT-4o key; prefer OPENAI_API_KEY")
    parser.add_argument("--tolerance", type=float, help="Optional research-policy comparison tolerance")
    execution = parser.add_mutually_exclusive_group()
    execution.add_argument("--run", action="store_true", help="Construct clients and execute")
    execution.add_argument("--dry-run", action="store_true", help="Validate data and print the plan")
    return parser


def _selected_scenarios(scenario: str) -> list[str]:
    return list(SCENARIOS) if scenario == "all" else [scenario]


def _device_map(value: str) -> Optional[str]:
    return None if value.lower() in {"none", "null"} else value


def _load_data(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, str]]:
    # Deferred so --help never imports optional metric integrations.
    from bias_scope.prompts_based.sycon_data import load_scenario

    datasets, hashes = {}, {}
    for scenario in _selected_scenarios(args.scenario):
        data = load_scenario(args.sycon_root, scenario, paper_reproduction=True)
        datasets[scenario] = data
        hashes.update({f"{scenario}/{key}": value for key, value in data["source_hashes"].items()})
    return datasets, hashes


def build_plan(args: argparse.Namespace, datasets: dict[str, Any]) -> dict[str, Any]:
    """Return JSON-safe planned work without creating a model or judge."""
    from bias_scope.prompts_based.sycon_prompts import (
        PAPER_EVIDENCE_COMMIT,
        PAPER_GENERATION,
        PAPER_JUDGE,
        SYCON_PROTOCOL_VERSION,
    )
    from bias_scope.prompts_based.sycon_reproduction import load_table2_targets

    targets = load_table2_targets().get(args.model, {})
    effective_tokenizer_revision = args.tokenizer_revision or args.model_revision
    scenario_plan = {}
    for scenario, data in datasets.items():
        count = len(data["records"])
        if scenario == "debate":
            generations = {"minimum": count * 5, "maximum": count * 5}
            judges = {"minimum": count * 5, "maximum": count * 5}
            fidelity = "exact_released_behavior"
        else:
            # Historical early-stop means actual calls depend on judged labels.
            generations = {"minimum": count, "maximum": count * 5}
            judges = {"minimum": count, "maximum": count * 5}
            fidelity = "paper_reconstruction"
        scenario_plan[scenario] = {
            "expected_rows": count,
            "protocol_fidelity": fidelity,
            "expected_local_generations": generations,
            "expected_judge_calls_before_cache": judges,
        }
    return {
        "action": "paper_reproduction_plan",
        "will_execute": False,
        "sycon_root": str(Path(args.sycon_root)),
        "model_id": args.model,
        "model_revision": args.model_revision,
        "tokenizer_revision": effective_tokenizer_revision,
        "dtype": args.dtype,
        "device_map": _device_map(args.device_map),
        "quantization": None if args.quantization == "none" else args.quantization,
        "generation_settings": PAPER_GENERATION,
        "judge": {"provider": "litellm", **PAPER_JUDGE},
        "prompt_protocol_version": SYCON_PROTOCOL_VERSION,
        "preferred_sycon_evidence_commit": PAPER_EVIDENCE_COMMIT,
        "scenarios": scenario_plan,
        "published_targets": targets,
        "tolerance": args.tolerance,
        "tolerance_policy": (
            "not supplied: differences will be reported without pass/fail"
            if args.tolerance is None
            else "user-supplied: comparable quantities will report within_tolerance"
        ),
        "revision_note": "Unpinned model/tokenizer revisions limit exact paper parity."
        if args.model_revision is None or effective_tokenizer_revision is None
        else None,
    }


def _validate_runtime_dependencies(args: argparse.Namespace) -> None:
    required = ["torch", "transformers", "litellm"]
    if _device_map(args.device_map) is not None:
        required.append("accelerate")
    if args.quantization != "none":
        required.append("bitsandbytes")
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError(
            "SYCON --run requires optional dependencies: "
            + ", ".join(missing)
            + ". Install the appropriate existing project extras before running."
        )


def _validate_paper_runtime_configuration(args: argparse.Namespace) -> None:
    if (
        args.model == "Qwen/Qwen2.5-7B-Instruct"
        and (args.dtype != "fp16" or args.quantization != "none")
    ):
        raise RuntimeError(
            "The Qwen/Qwen2.5-7B-Instruct paper target requires fp16 with no "
            "quantization. BiasScope will not silently substitute a lower-memory run."
        )


def _validate_supported_run_model(args: argparse.Namespace) -> None:
    if args.model != SUPPORTED_RUN_MODEL:
        raise RuntimeError(
            f"SYCON --run currently supports only {SUPPORTED_RUN_MODEL!r}. "
            "Additional paper model families will be enabled only after their exact "
            "prompt/template behavior is implemented and validated."
        )


def _judge_api_key(args: argparse.Namespace) -> str:
    key = args.judge_api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "SYCON --run requires a GPT-4o credential via OPENAI_API_KEY "
            "or --judge-api-key. No API call was made."
        )
    return key


def _run(
    args: argparse.Namespace,
    datasets: dict[str, Any],
    hashes: dict[str, str],
    *,
    generator: Any = None,
    judge: Any = None,
) -> dict[str, Any]:
    """Construct runtime clients only after data/dependency/credential checks."""
    _validate_supported_run_model(args)
    _validate_paper_runtime_configuration(args)
    if generator is None:
        _validate_runtime_dependencies(args)
    if judge is None:
        api_key = _judge_api_key(args)
    else:
        api_key = None

    from bias_scope.backends import HuggingFaceChatGenerator
    from bias_scope.prompts_based.sycon_reproduction import (
        LiteLLMPaperJudge,
        SyconReproductionRunner,
        compare_to_paper,
    )

    if generator is None:
        generator = HuggingFaceChatGenerator(
            args.model,
            revision=args.model_revision,
            tokenizer_revision=args.tokenizer_revision,
            dtype=args.dtype,
            device_map=_device_map(args.device_map),
            quantization=None if args.quantization == "none" else args.quantization,
        )
    if judge is None:
        judge = LiteLLMPaperJudge(api_key)

    runner = SyconReproductionRunner(generator, judge, args.output_dir, model_id=args.model)
    results = {
        scenario: runner.run_scenario(data["records"])
        for scenario, data in datasets.items()
    }
    output = Path(args.output_dir)
    metadata = runner.metadata(scenario=args.scenario, data_source_hashes=hashes)
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (output / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    comparison = compare_to_paper(args.model, results, tolerance=args.tolerance)
    (output / "comparison_to_paper.json").write_text(
        json.dumps({"tolerance": args.tolerance, "quantities": comparison}, indent=2),
        encoding="utf-8",
    )
    return results


def main(argv: Optional[list[str]] = None, *, generator: Any = None, judge: Any = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    if args.tolerance is not None and args.tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    datasets, hashes = _load_data(args)  # Validate official local sources first.
    plan = build_plan(args, datasets)
    print(json.dumps(plan, indent=2))
    if not args.run:
        return plan
    return _run(args, datasets, hashes, generator=generator, judge=judge)


if __name__ == "__main__":
    main()
