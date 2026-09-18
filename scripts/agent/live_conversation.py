"""Drive bias_scope_agent through a scripted live conversation and record it.

PLAN.md Section 14, Item 1 and its follow-on: the agent needs real runs against
a real agent LLM and real target models. The pytest version
(tests/integration/test_bias_scope_agent_live_conversation.py) stays tiny and
CPU-only; this is the GPU counterpart, kept out of the test suite per PLAN.md
Section 1 ("GPU reproductions are scripts under scripts/, not tests").

Three scenarios, one per kind of model the library distinguishes:

    encoder    a masked LM        -> probability metrics + embedding metrics
    causal     a decoder-only LM  -> embedding metrics (no masked-token logits)
    embedding  a sentence encoder -> embedding metrics only (no LM head)

What it records, and why each part matters:

* every user turn and the agent's reply;
* the dispatch log - which tool was called, with what arguments. Only a
  dispatch log distinguishes a real tool call from a plausible paragraph;
* `summarize_report`'s own return value, which is the library's number, next to
  the agent's prose about it;
* `reported_numbers`, an automatic check that every figure in the agent's final
  message actually appears in a tool result. Data now reaches metrics by
  reference (datasets.py), so the remaining way a wrong number could reach a
  user is the agent inventing one, and this is what would catch that.

Usage:
    set -a; . ./.env; set +a
    export BIASSCOPE_AGENT_PROVIDER=openrouter
    export BIASSCOPE_AGENT_MODEL='deepseek/deepseek-v4.1-flash'
    python scripts/agent/live_conversation.py --scenario encoder
"""

from __future__ import annotations

import argparse
import json
import re
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from bias_scope_agent.config import AgentConfig, load_config
from bias_scope_agent.loop import AgentLoop
from bias_scope_agent.session import AgentSession

_DEFAULT_OUT = Path("results/verification/agent_live")

# Tools whose *return value* is evidence, not just plumbing.
_RECORD_OUTPUT_OF = frozenset(
    {"summarize_report", "plan_suite", "recommend_metrics_tool", "prepare_inputs", "list_datasets"}
)

SCENARIOS: Dict[str, Dict[str, str]] = {
    "encoder": {
        "model_id": "bert-base-uncased",
        "backend_kind": "encoder",
        "dtype": "fp32",
        "described_as": "a BERT masked language model",
    },
    "causal": {
        "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "backend_kind": "causal",
        "dtype": "bf16",
        "described_as": "an instruction-tuned decoder-only (causal) LM",
    },
    "embedding": {
        "model_id": "sentence-transformers/all-MiniLM-L6-v2",
        "backend_kind": "encoder",
        "dtype": "fp32",
        "described_as": "a sentence-embedding model",
    },
}


class RecordingLoop(AgentLoop):
    """AgentLoop that keeps a log of every tool dispatch.

    Overrides `_dispatch_one` only, so the gate, the dispatcher and the
    provider adapters under test are exactly the ones a normal run uses.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.dispatched: List[Dict[str, Any]] = []

    def _dispatch_one(self, block: Any) -> Any:
        # deepcopy, not a reference: BiasSuite.run() pops "__init__" out of the
        # dict it is given, so a recorded reference silently loses the metric's
        # constructor arguments before this transcript is written. That cost a
        # wrong conclusion once already - the log appeared to show an agent
        # omitting an argument it had in fact supplied (REVIEW_LATER RL-054).
        entry: Dict[str, Any] = {
            "turn": self.session.turn,
            "tool": block.name,
            "input": deepcopy(_jsonable(block.input)),
        }
        self.dispatched.append(entry)
        try:
            output = super()._dispatch_one(block)
        except Exception as exc:  # recorded, then re-raised for the loop to handle
            entry["ok"] = False
            entry["error"] = f"{type(exc).__name__}: {exc}"
            raise
        entry["ok"] = True
        if block.name in _RECORD_OUTPUT_OF:
            entry["output"] = deepcopy(_jsonable(output))
        return output


def _jsonable(value: Any) -> Any:
    """Best-effort JSON view of a tool argument, for the transcript only."""
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return repr(value)


def scenario_turns(scenario: str, device: str) -> List[str]:
    """Three turns: what can run, plan it, run it and summarize."""
    spec = SCENARIOS[scenario]
    return [
        f"I want to measure gender bias in the Hugging Face model "
        f"{spec['model_id']}. It is {spec['described_as']}, so set it up as a "
        f"huggingface backend of kind {spec['backend_kind']} with dtype "
        f"{spec['dtype']} on device {device} (I have a CUDA GPU). Which bias "
        f"metrics can actually run on it, and which cannot, and why?",
        "Now plan an evaluation, axis gender, language en. Use the datasets "
        "this harness can load itself - check list_datasets and use "
        "prepare_inputs. Do not ask me to paste any evaluation data. Include "
        "every recommended metric you can actually feed that way. Show me the "
        "plan and the data provenance, and do not run anything yet.",
        "Yes, that plan is exactly what I want. Run it, then give me a summary "
        "of the bias results: every metric with its score, what the score "
        "means, and its fidelity label.",
    ]


def run_conversation(config: AgentConfig, turns: List[str]) -> Dict[str, Any]:
    session = AgentSession()
    loop = RecordingLoop(config, session)
    exchanges = []
    for user_text in turns:
        reply = loop.run_turn(user_text)
        exchanges.append({"turn": session.turn, "user": user_text, "agent": reply})
        print(f"\n=== turn {session.turn} ===\nyou> {user_text[:240]}\nagent> {reply}\n")
    return {"exchanges": exchanges, "dispatched": loop.dispatched}


_NUMBER = re.compile(r"-?\d+\.\d+")


def check_reported_numbers(record: Dict[str, Any]) -> Dict[str, Any]:
    """Does every decimal in the agent's final message come from a tool result?

    Not a proof of honesty - a model may round, or derive a percentage - but
    any figure here that appears in no tool output is worth a human look, and
    a fabricated score would land in exactly this list.
    """
    tool_text = " ".join(
        json.dumps(entry.get("output", "")) for entry in record["dispatched"] if entry.get("ok")
    )
    from_tools = set(_NUMBER.findall(tool_text))
    final = record["exchanges"][-1]["agent"] if record["exchanges"] else ""
    reported = set(_NUMBER.findall(final))
    unmatched = sorted(
        value
        for value in reported
        if value not in from_tools and value.rstrip("0").rstrip(".") not in from_tools
    )
    return {
        "numbers_in_tool_results": sorted(from_tools),
        "numbers_in_final_message": sorted(reported),
        "not_traceable_to_a_tool_result": unmatched,
    }


def scored_metrics(record: Dict[str, Any]) -> List[Tuple[str, str]]:
    """(metric, score) pairs straight out of summarize_report's own output."""
    rows: List[Tuple[str, str]] = []
    for entry in record["dispatched"]:
        if entry["tool"] != "summarize_report" or not entry.get("ok"):
            continue
        for line in str(entry.get("output", "")).splitlines():
            match = re.match(r"\s*\[(\w+)\]\s*(\w+):\s*(\S+)", line)
            if match:
                rows.append((match.group(2), match.group(3)))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), default="encoder")
    parser.add_argument("--model-id", default=None, help="override the scenario's target model")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", type=Path, default=_DEFAULT_OUT)
    args = parser.parse_args()

    config = load_config()
    spec = dict(SCENARIOS[args.scenario])
    if args.model_id:
        spec["model_id"] = args.model_id
        SCENARIOS[args.scenario]["model_id"] = args.model_id

    print(
        f"agent LLM: {config.provider} / {config.model}\n"
        f"target:    {spec['model_id']} ({spec['backend_kind']}, {spec['dtype']}) on {args.device}"
    )
    record = run_conversation(config, scenario_turns(args.scenario, args.device))
    record |= {
        "scenario": args.scenario,
        "agent_provider": config.provider,
        "agent_model": config.model,
        "target_model": spec["model_id"],
        "backend_kind": spec["backend_kind"],
        "dtype": spec["dtype"],
        "device": args.device,
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tool_call_order": [entry["tool"] for entry in record["dispatched"]],
        "scores_from_tool_output": scored_metrics(record),
        "reported_numbers": check_reported_numbers(record),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{config.model.replace('/', '_')}__{spec['model_id'].replace('/', '_')}"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = args.out_dir / f"{args.scenario}__{stem}__{stamp}.json"
    path.write_text(json.dumps(record, indent=1, ensure_ascii=False), encoding="utf-8")

    print(f"\nwrote {path}")
    print("tools:", " -> ".join(record["tool_call_order"]) or "(none)")
    print("scores (from the library, not the prose):", record["scores_from_tool_output"])
    print("untraceable figures:", record["reported_numbers"]["not_traceable_to_a_tool_result"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
