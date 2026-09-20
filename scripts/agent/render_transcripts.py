"""Render the recorded agent runs as a README of the actual interaction.

For each target model's latest complete run: the exact command that produced
it, the environment variables that chose the agent LLM, every user turn and
agent reply verbatim, every tool call (rejected ones included, with the
error), the provenance each `prepare_inputs` returned, and `summarize_report`'s
own output. Nothing is paraphrased; this is the log.

Usage:
    python scripts/agent/render_transcripts.py --out results/verification/agent_live/README.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # `scripts.` when run as a file

from scripts.agent.results_table import latest_complete_runs  # noqa: E402
from scripts.agent.summarize_runs import _DEFAULT_DIR, _load  # noqa: E402

_ENV = """set -a; . ./.env; set +a          # OPENROUTER_API_KEY lives in .env, never in the repo
export BIASSCOPE_AGENT_PROVIDER={provider}
export BIASSCOPE_AGENT_MODEL={model}"""

_BATCH = """```bash
set -a; . ./.env; set +a
export BIASSCOPE_AGENT_PROVIDER=openrouter
export BIASSCOPE_AGENT_MODEL=deepseek/deepseek-v4.1-flash
run() { python scripts/agent/live_conversation.py --scenario "$1" --model-id "$2" --device cuda; }
for m in gpt2 gpt2-medium Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-1.5B-Instruct \\
         meta-llama/Llama-3.2-1B-Instruct google/gemma-3-1b-it \\
         Qwen/Qwen2.5-3B-Instruct; do run causal "$m"; done
# gated checkpoints already downloaded run with HF_HUB_OFFLINE=1 (see REPRODUCE.md);
# google/gemma-2-2b-it is not accessible to this account (REVIEW_LATER RL-079)
for m in bert-base-uncased bert-base-cased roberta-base; do run encoder "$m"; done
for m in sentence-transformers/all-MiniLM-L6-v2 sentence-transformers/all-mpnet-base-v2; do
  run embedding "$m"
done
python scripts/agent/summarize_runs.py --check
python scripts/agent/results_table.py --out results/verification/agent_live/RESULTS.md
python scripts/agent/render_transcripts.py --out results/verification/agent_live/README.md
```"""


def _command(record: Dict[str, Any]) -> str:
    return (
        f"python scripts/agent/live_conversation.py --scenario {record['scenario']} "
        f"--model-id {record['target_model']} --device {record.get('device', 'cuda')}"
    )


def _tool_lines(record: Dict[str, Any]) -> List[str]:
    lines = []
    for entry in record["dispatched"]:
        args = json.dumps(entry.get("input", {}), ensure_ascii=False)
        if len(args) > 300:
            args = args[:300] + " ..."
        status = "ok" if entry.get("ok") else f"REJECTED: {entry.get('error', '')}"
        lines.append(f"- turn {entry['turn']} `{entry['tool']}` {args} -> {status}")
        if entry["tool"] == "prepare_inputs" and entry.get("ok"):
            prov = (entry.get("output") or {}).get("provenance", {})
            lines.append("  - provenance: " + json.dumps(prov, ensure_ascii=False))
    return lines


def _report(record: Dict[str, Any]) -> str:
    for entry in reversed(record["dispatched"]):
        if entry["tool"] == "summarize_report" and entry.get("ok"):
            return str(entry["output"])
    return "(no report: the run never reached summarize_report)"


def render_run(record: Dict[str, Any]) -> str:
    cov = record.get("recommendation_coverage") or {}
    untraceable = record.get("reported_numbers", {}).get("not_traceable_to_a_tool_result")
    parts = [
        f"## {record['target_model']} ({record['scenario']}, {record.get('dtype')}, "
        f"{record.get('device')}) - recorded {record.get('recorded_at')}",
        "",
        "```bash",
        _ENV.format(provider=record.get("agent_provider"), model=record.get("agent_model")),
        _command(record),
        "```",
        "",
        f"Coverage: recommended {len(cov.get('recommended', []))}, feedable "
        f"{len(cov.get('feedable', []))}, scored {len(cov.get('scored', []))}; "
        f"complete: {cov.get('complete')}. Figures in the agent's final message not traceable "
        f"to a tool result: {untraceable}",
        "",
        "### Conversation",
        "",
    ]
    for exchange in record["exchanges"]:
        parts += [f"**you (turn {exchange['turn']})>** {exchange['user']}", "",
                  "**agent>**", "", exchange["agent"], "", "---", ""]
    parts += ["### Tool calls (the dispatch log)", ""] + _tool_lines(record)
    parts += ["", "### The library's report (`summarize_report`)", "", "```", _report(record),
              "```", ""]
    return "\n".join(parts)


def render(records: List[Dict[str, Any]]) -> str:
    runs = latest_complete_runs(records)
    ordered = sorted(runs.values(), key=lambda r: ({"encoder": 0, "embedding": 1, "causal": 2}
                                                   .get(r["scenario"], 9), r["target_model"]))
    head = [
        "# Agent runs: commands and interaction logs",
        "",
        "Generated by `python scripts/agent/render_transcripts.py` from the JSON transcripts in "
        "this directory; one section per target model, its latest run in which every feedable "
        "metric scored. The agent LLM is the one named in each section; the target model is "
        "the one being evaluated. Every reply is verbatim. `RESULTS.md` has the table; "
        "`REPRODUCE.md` explains how to run this yourself.",
        "",
        "Batch commands used for the 2026-09-20 experiments:",
        "",
        _BATCH,
        "",
        "Models: " + ", ".join(r["target_model"] for r in ordered),
        "",
    ]
    return "\n".join(head) + "\n".join(render_run(r) for r in ordered)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=_DEFAULT_DIR)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    text = render(_load(args.dir))
    if args.out:
        args.out.write_text(text, encoding="utf-8")
        print(f"wrote {args.out} ({len(text)} chars)")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
