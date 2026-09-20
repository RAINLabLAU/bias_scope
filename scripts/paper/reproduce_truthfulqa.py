"""Offline dry-run / explicit local reproduction runner for TruthfulQA v0."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Allow direct execution from a source checkout without requiring installation.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
# LiteLLM 1.100.0 documents this switch as its bundled cost-map-only mode.
# Set it before importing BiasScope because BiasScope exposes optional LiteLLM
# support at package import time. This is confined to this offline script.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "true")

from bias_scope.prompts_based._truthfulqa_data import load_v0_questions, validate_v0_mc_task
from bias_scope.prompts_based._truthfulqa_protocol import MODEL_PROFILES, PAPER_EVIDENCE_COMMIT, REPRODUCTION_PROTOCOL_VERSION
from bias_scope.prompts_based._truthfulqa_reproduction import (
    collect_generation, compare_targets, reconstruct_mc, runtime_metadata, targets_for,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline TruthfulQA ACL-2022 v0 reconstruction planner.")
    parser.add_argument("--data-root", required=True, help="Caller-supplied official TruthfulQA repository or data root.")
    parser.add_argument("--model", default="gpt2-xl", choices=sorted(MODEL_PROFILES))
    parser.add_argument("--mode", default="mc_paper_reproduction", choices=("mc_paper_reproduction", "generation_collection", "historical_generation_score_parity"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--run", action="store_true", help="Permit local model execution; never used by default.")
    args = parser.parse_args()
    questions, dataset = load_v0_questions(args.data_root)
    mc_task = validate_v0_mc_task(args.data_root)
    plan = {"offline": not args.run, "mode": args.mode, "model": MODEL_PROFILES[args.model].as_dict(),
            "dataset": dataset, "mc_task": mc_task, "paper_evidence_commit": PAPER_EVIDENCE_COMMIT,
            "protocol_version": REPRODUCTION_PROTOCOL_VERSION, "targets": targets_for(args.model, "multiple_choice" if args.mode == "mc_paper_reproduction" else "generation"),
            "runtime": runtime_metadata(), "notice": "Reconstruction comparison; historical revisions were not pinned."}
    print(json.dumps(plan, indent=2))
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "truthfulqa_manifest.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    if not args.run:
        return 0
    if args.mode == "historical_generation_score_parity":
        raise SystemExit("Historical GPT-judge/GPT-info artifacts are unavailable; this mode cannot be synthesized.")
    if not plan["runtime"]["torch"] or not plan["runtime"]["transformers"]:
        raise SystemExit("--run requires torch and transformers before any model loading; install bias-scope[torch] separately.")
    profile = MODEL_PROFILES[args.model]
    if not profile.local or profile.model_id is None:
        raise SystemExit("This historical/proprietary model has no supported local checkpoint profile.")
    if args.mode == "generation_collection":
        if profile.architecture == "causal":
            from bias_scope.backends import HuggingFaceCausalGenerator
            generator = HuggingFaceCausalGenerator(profile.model_id)
            generate = lambda prompt: generator.generate(prompt, do_sample=False, max_new_tokens=50)
        else:
            from bias_scope.backends import HuggingFaceSeq2SeqGenerator
            generator = HuggingFaceSeq2SeqGenerator(profile.model_id)
            # Paper-era UQA passed only top_k=1; config defaults were unpinned.
            generate = lambda prompt: generator.generate(prompt, do_sample=False, top_k=1)
        result = collect_generation(questions, model_id=profile.model_id, architecture=profile.architecture, generator=generate)
        result.update({"dataset": dataset, "runtime": runtime_metadata(), "backend": generator.protocol_fields() if hasattr(generator, "protocol_fields") else {"model_id": profile.model_id}})
        if args.output_dir:
            (args.output_dir / "truthfulqa_generation_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps({"generation_collection": "complete", "historical_generation_score_parity": "unavailable"}, indent=2))
        return 0
    if profile.architecture == "causal":
        from bias_scope.backends import HuggingFaceCausalLikelihoodScorer
        scorer = HuggingFaceCausalLikelihoodScorer(profile.model_id)
        score = lambda prefix, appended: scorer.score(prefix, appended, skip_continuation_tokens=3)
    else:
        from bias_scope.backends import HuggingFaceSeq2SeqLikelihoodScorer
        scorer = HuggingFaceSeq2SeqLikelihoodScorer(profile.model_id)
        score = lambda source, target: scorer.score(source, target, include_eos=False)
    result = reconstruct_mc(questions, model_id=profile.model_id, architecture=profile.architecture,
                            scorer=score, model_revision=scorer.revision,
                            tokenizer_revision=scorer.tokenizer_revision,
                            cache_dir=args.output_dir / "candidate_cache" if args.output_dir else None,
                            dataset_sha256=dataset["sha256"])
    result["dataset"] = dataset
    result["runtime"] = runtime_metadata(requested_device=scorer.device, dtype=scorer.dtype)
    result["backend"] = scorer.protocol_fields()
    result["runtime"]["effective_device"] = result["backend"]["effective_device"]
    comparison = compare_targets(result, args.model)
    if args.output_dir:
        (args.output_dir / "truthfulqa_mc_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        (args.output_dir / "truthfulqa_target_comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(json.dumps(comparison, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
