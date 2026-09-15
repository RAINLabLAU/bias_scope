"""Offline OpinionQA consistency parity runner; never downloads or queries models."""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# LiteLLM 1.100.0 documents this flag to use its bundled cost map rather than
# fetching GitHub at import time. Set only for this offline research script.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
sys.path.insert(0, str(ROOT / "src"))
from bias_scope.prompts_based._opinionqa_parity import (NOTEBOOK_RECONSTRUCTION, PAPER_FORMULA, PEW_SURVEY_LIST, build_manifest, compare_with_public_metric, load_all_combined, load_topic_mapping, reconstruct, write_json)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True, help="Caller-supplied official OpinionQA data root.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", choices=[PAPER_FORMULA, NOTEBOOK_RECONSTRUCTION], required=True)
    parser.add_argument("--topic-granularity", default="cg", choices=["cg"])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-partial-data", action="store_true", help="Diagnostic only; never represents full official-data parity.")
    args = parser.parse_args()
    mapping_path = args.data_root / "human_resp" / "topic_mapping.npy"
    if not mapping_path.exists():
        parser.error("Offline plan: supply human_resp/topic_mapping.npy and generated distributions/*_default_combined.csv. Run the official process_results.ipynb locally first; this script will not download or generate model outputs.")
    mapping = load_topic_mapping(mapping_path)
    try:
        records, descriptions = load_all_combined(args.data_root, mapping, model=args.model, allow_partial=args.allow_partial_data)
    except ValueError as exc:
        parser.error(str(exc))
    if args.allow_partial_data:
        print("PARTIAL / DIAGNOSTIC ONLY: this is not full official-data parity.")
    print("Using waves: " + ", ".join(str(item["survey_wave"]) for item in descriptions))
    result = reconstruct(records, mode=args.mode)
    manifest = build_manifest(combined=descriptions, topic_mapping_path=mapping_path, result=result, records=records)
    loaded_waves = [item["survey_wave"] for item in descriptions]
    scope = {"data_scope": "full_official" if len(loaded_waves) == len(PEW_SURVEY_LIST) else "partial_diagnostic", "is_full_parity": len(loaded_waves) == len(PEW_SURVEY_LIST), "expected_waves": list(PEW_SURVEY_LIST), "loaded_waves": loaded_waves, "missing_waves": [wave for wave in PEW_SURVEY_LIST if wave not in loaded_waves]}
    result.update(scope); manifest.update(scope)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "opinionqa_manifest.json", manifest)
    name = "paper_formula_results.json" if args.mode == PAPER_FORMULA else "official_notebook_results.json"
    write_json(args.output_dir / name, result)
    if args.mode == PAPER_FORMULA:
        comparison = compare_with_public_metric(records, result)
        comparison.update(scope)
        write_json(args.output_dir / "parity_comparison.json", comparison)
    print(f"offline parity complete: {args.output_dir / name}")

if __name__ == "__main__":
    main()
