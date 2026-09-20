"""Did the agent actually run every recommended metric the harness can feed?

`recommend_metrics_tool` says what *should* run; `run_suite` and
`summarize_report` say what *did*. Nothing compared the two until now, so a
recommended metric quietly left out of the plan looked identical to one that
was run. `recommendation_coverage` reads a recorded transcript
(scripts/agent/live_conversation.py) and lists the gap, so a live run can be
checked mechanically rather than by reading the agent's prose.

The transcript shape is the one live_conversation.py writes: `dispatched`
entries with `tool`, `input`, `ok` and (for evidence tools) `output`.
"""

import json
from pathlib import Path

from scripts.agent.live_conversation import recommendation_coverage

_RECORDED = Path("results/verification/agent_live")


def _rec(metric, access):
    return {"metric": metric, "family": "x", "fidelity": "faithful", "access": list(access)}


def _record(recommended, planned, run, summary):
    return {
        "dispatched": [
            {"turn": 1, "tool": "recommend_metrics_tool", "ok": True, "input": {},
             "output": recommended},
            {"turn": 2, "tool": "plan_suite", "ok": True,
             "input": {"metric_names": planned}, "output": {}},
            {"turn": 3, "tool": "run_suite", "ok": False,
             "input": {"metric_names": run}, "error": "ValueError: rejected"},
            {"turn": 3, "tool": "run_suite", "ok": True, "input": {"metric_names": run}},
            {"turn": 3, "tool": "summarize_report", "ok": True, "input": {}, "output": summary},
        ]
    }


def test_full_coverage_when_every_feedable_metric_is_scored():
    recommended = [_rec("CrowSPairs", ["logits"]), _rec("WEAT", ["embeddings"]),
                   _rec("LPBS", ["logits"])]
    summary = "Bias report\n\nembedding:\n  [faithful] WEAT: 0.61 (n=16)\n\n" \
              "probability:\n  [faithful] CrowSPairs: 55.73 (n=262)\n"
    cov = recommendation_coverage(
        _record(recommended, ["CrowSPairs", "WEAT"], ["CrowSPairs", "WEAT"], summary)
    )
    assert cov["recommended"] == ["CrowSPairs", "LPBS", "WEAT"]
    # LPBS is recommended but no dataset provider serves it: not feedable.
    assert cov["feedable"] == ["CrowSPairs", "WEAT"]
    assert cov["not_feedable"] == ["LPBS"]
    assert cov["planned"] == ["CrowSPairs", "WEAT"]
    assert cov["run"] == ["CrowSPairs", "WEAT"]        # only the ok=True call counts
    assert cov["scored"] == ["CrowSPairs", "WEAT"]
    assert cov["feedable_not_scored"] == []
    assert cov["scored_not_recommended"] == []
    assert cov["complete"] is True


def test_feedable_is_judged_by_the_datasets_the_run_was_offered():
    """A provider added later must not make an old transcript look incomplete.
    When the run recorded a list_datasets result, "feedable" means served by
    a dataset in that result; only a run that never listed datasets falls back
    to today's table."""
    recommended = [_rec("CrowSPairs", ["logits"]), _rec("WEAT", ["embeddings"])]
    record = _record(recommended, ["WEAT"], ["WEAT"], "embedding:\n  [faithful] WEAT: 0.6 (n=16)\n")
    record["dispatched"].insert(1, {
        "turn": 2, "tool": "list_datasets", "ok": True, "input": {},
        "output": [{"dataset": "weat", "metrics": ["WEAT"], "axes": ["gender"]}],
    })
    cov = recommendation_coverage(record)
    assert cov["feedable"] == ["WEAT"]
    assert cov["complete"] is True


def test_a_run_that_never_listed_datasets_is_judged_by_todays_table():
    recommended = [_rec("CrowSPairs", ["logits"]), _rec("WEAT", ["embeddings"])]
    cov = recommendation_coverage(_record(recommended, ["WEAT"], ["WEAT"], ""))
    assert cov["feedable"] == ["CrowSPairs", "WEAT"]


def test_a_feedable_metric_left_out_of_the_plan_is_reported():
    recommended = [_rec("CrowSPairs", ["logits"]), _rec("AUL", ["logits"]),
                   _rec("WEAT", ["embeddings"])]
    summary = "probability:\n  [faithful] CrowSPairs: 55.73 (n=262)\n"
    cov = recommendation_coverage(_record(recommended, ["CrowSPairs"], ["CrowSPairs"], summary))
    assert cov["feedable_not_scored"] == ["AUL", "WEAT"]
    assert cov["complete"] is False


def test_a_metric_the_suite_skipped_is_not_counted_as_scored():
    recommended = [_rec("CrowSPairs", ["logits"]), _rec("AUL", ["logits"])]
    summary = ("probability:\n  [faithful] CrowSPairs: 55.73 (n=262)\n\n"
               "skipped:\n  AUL: RuntimeError: boom\n")
    cov = recommendation_coverage(
        _record(recommended, ["CrowSPairs", "AUL"], ["CrowSPairs", "AUL"], summary)
    )
    assert cov["run"] == ["AUL", "CrowSPairs"]
    assert cov["scored"] == ["CrowSPairs"]
    assert cov["skipped"] == {"AUL": "RuntimeError: boom"}
    assert cov["feedable_not_scored"] == ["AUL"]
    assert cov["complete"] is False


def test_a_generating_dataset_is_feedable_only_with_completions_access():
    # bold_regard needs `completions`. An encoder has none, so RegardScore is
    # not feedable there even if it were recommended; a causal LM has it.
    encoder_like = [_rec("RegardScore", ["completions"]), _rec("WEAT", ["embeddings"])]
    cov = recommendation_coverage(_record(encoder_like, ["WEAT"], ["WEAT"], ""))
    assert "RegardScore" in cov["feedable"]
    only_embeddings = [_rec("WEAT", ["embeddings"])]
    cov = recommendation_coverage(_record(only_embeddings, ["WEAT"], ["WEAT"], ""))
    assert cov["feedable"] == ["WEAT"]


def test_a_metric_scored_but_never_recommended_is_flagged():
    recommended = [_rec("WEAT", ["embeddings"])]
    summary = "embedding:\n  [faithful] WEAT: 0.6 (n=16)\n  [faithful] SEAT: 1.0 (n=128)\n"
    cov = recommendation_coverage(_record(recommended, ["WEAT", "SEAT"], ["WEAT", "SEAT"], summary))
    assert cov["scored_not_recommended"] == ["SEAT"]
    assert cov["complete"] is False


# Runs kept as evidence of a defect they exposed; each is incomplete by
# construction and says so in REVIEW_LATER.
_INCOMPLETE_BY_DESIGN = {
    "causal__deepseek_deepseek-v4.1-flash__gpt2__20260918T215303Z.json": "RL-067: no pad token",
    "causal__deepseek_deepseek-v4.1-flash__gpt2__20260918T215749Z.json": "RL-067: second path",
    # First runs with the six new providers: greedy 30-token continuations gave
    # StereotypicalAssociations nothing to count (RL-073), and rtp_toxicity's
    # axis tag "toxicity" kept EMT out of a gender-axis plan.
    "causal__deepseek_deepseek-v4.1-flash__gpt2__20260920T142104Z.json": "RL-073; EMT axis tag",
    "causal__deepseek_deepseek-v4.1-flash__Qwen_Qwen2.5-0.5B-Instruct__20260920T142706Z.json":
        "RL-073",
    # BOS-prepending causal LMs: position-0 pooling gives identical vectors, so
    # SEAT and CEAT decline with a zero standard deviation (RL-068).
    "causal__deepseek_deepseek-v4.1-flash__meta-llama_Llama-3.2-1B-Instruct__20260920T155044Z.json":
        "RL-068: BOS at position 0",
    "causal__deepseek_deepseek-v4.1-flash__meta-llama_Llama-3.2-1B-Instruct__20260920T162153Z.json":
        "RL-068: BOS at position 0 (7 of 9 scored; SEAT and CEAT decline by construction)",
    "causal__deepseek_deepseek-v4.1-flash__meta-llama_Llama-3.2-1B-Instruct__20260920T200233Z.json":
        "RL-068: BOS at position 0 (8 of 9 after the merge; SEAT declines, CEAT now scores)",
    "causal__deepseek_deepseek-v4.1-flash__google_gemma-3-1b-it__20260920T165201Z.json":
        "RL-068: BOS at position 0 (8 of 9; CEAT declines, SEAT returns a degenerate 0)",
    # WEAT's sentence-transformers loader tried to build an image processor for
    # the Gemma 3 family and nothing scored (RL-076); rerun after the fix.
    "causal__deepseek_deepseek-v4.1-flash__google_gemma-3-1b-it__20260920T155315Z.json":
        "RL-076: sentence-transformers loader",
    # Second gemma-3 run: generations all came from the cache, so the backend
    # never loaded and sharing was never registered (RL-078); rerun after the fix.
    "causal__deepseek_deepseek-v4.1-flash__google_gemma-3-1b-it__20260920T160646Z.json":
        "RL-078: cache hit, no registration",
    # Only a partial local download of the weights, run offline: every load
    # failed and nothing scored. Rerun online after RL-076 removed the loader
    # that needed the offline workaround.
    "causal__deepseek_deepseek-v4.1-flash__google_gemma-2-2b-it__20260920T155642Z.json":
        "RL-079: gated, no access (offline attempt)",
    "causal__deepseek_deepseek-v4.1-flash__google_gemma-2-2b-it__20260920T161101Z.json":
        "RL-079: gated, no access (online attempt)",
    # Ran out of GPU memory before any metric scored (RL-075); rerun after the fix.
    "causal__deepseek_deepseek-v4.1-flash__Qwen_Qwen2.5-3B-Instruct__20260920T154817Z.json":
        "RL-075: OOM",
}
# Every deepseek run from this stamp on was made after the RL-060..RL-065 fixes.
_FIRST_FINAL_STAMP = "20260918T193634Z"


def test_recorded_final_runs_have_full_coverage():
    """Every recorded run since the 2026-09-18 fixes shows no feedable metric
    left unscored, except those listed above with the defect they document."""
    final = [
        path
        for path in sorted(_RECORDED.glob("*__deepseek_deepseek-v4.1-flash__*.json"))
        if path.stem.rsplit("__", 1)[-1] >= _FIRST_FINAL_STAMP
        and path.name not in _INCOMPLETE_BY_DESIGN
    ]
    assert len(final) >= 3, [p.name for p in final]
    for path in final:
        cov = recommendation_coverage(json.loads(path.read_text(encoding="utf-8")))
        gap = (cov["feedable_not_scored"], cov["scored_not_recommended"])
        assert cov["complete"], (path.name, gap)


def test_the_incomplete_by_design_list_is_not_stale():
    for name in _INCOMPLETE_BY_DESIGN:
        path = _RECORDED / name
        assert path.exists(), name
        cov = recommendation_coverage(json.loads(path.read_text(encoding="utf-8")))
        assert not cov["complete"], f"{name} is complete now; drop it from the list"
