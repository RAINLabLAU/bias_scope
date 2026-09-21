"""End-to-end: construct_backend -> plan_suite -> confirm_plan -> run_suite ->
summarize_report, driven entirely by tools.py against a real HuggingFaceBackend
for a real (if tiny) HF encoder id.

Metric choice (WEAT, with hand-written embedding arrays rather than raw
words): bias_scope's own BiasMetric.run() reads a metric's headline score from
one of a fixed set of dict keys ("bias_score", "score", "value",
"effect_size" - see base.py's _extract_score). Several probability-family
metrics that would otherwise fit an encoder backend (e.g. CrowSPairs, AUL) key
their return dict as "<name>_score" instead, so .run() raises BiasScopeError
for them today - a pre-existing bias_scope defect, unrelated to this feature
and out of scope to fix here (CLAUDE.md forbids touching src/bias_scope/).
Confirmed via manual reproduction before writing this test; see
REVIEW_LATER.md RL-041. WEAT is unaffected and, per its own docstring, accepts
raw embedding arrays directly - so this test exercises a real, non-stub
HuggingFaceBackend end to end through the full tool chain without needing a
live embedding-model download (this repo's own embedding tests mock the
encoder for the same reason - see tests/test_embeddings/test_embedding_helper.py).
"""

from pathlib import Path

import numpy as np
import pytest
from tests.conftest import TINY_ENCODER_ID

from bias_scope.metadata import list_metrics
from bias_scope.report import FIDELITY_BADGE
from bias_scope_agent import tools
from bias_scope_agent.session import AgentSession

_WEAT_INPUTS = {
    "target_embeddings": (
        np.array([[1.0, 0.0], [0.9, 0.1], [0.8, 0.2]]),
        np.array([[0.0, 1.0], [0.1, 0.9], [0.2, 0.8]]),
    ),
    "attribute_embeddings": (
        np.array([[1.0, 0.0], [0.95, 0.05], [0.9, 0.1]]),
        np.array([[0.0, 1.0], [0.05, 0.95], [0.1, 0.9]]),
    ),
}


def test_construct_plan_confirm_run_summarize_end_to_end():
    session = AgentSession()
    backend_handle = tools.construct_backend(
        session, kind="huggingface", model_id=TINY_ENCODER_ID, backend_kind="encoder"
    )

    plan = tools.plan_suite(session, backend_handle, metric_names=["WEAT"])
    assert set(plan["needs_data"]["WEAT"]) == {"target_embeddings", "attribute_embeddings"}

    session.advance_turn()
    tools.confirm_plan(session, plan["plan_id"])

    report_handle = tools.run_suite(
        session, backend_handle, metric_names=["WEAT"], inputs={"WEAT": dict(_WEAT_INPUTS)}
    )

    summary = tools.summarize_report(session, report_handle, format="chat")
    assert TINY_ENCODER_ID in summary
    assert "WEAT" in summary
    badge = FIDELITY_BADGE[list_metrics()["WEAT"].fidelity]
    assert badge in summary


# --- Regression: RL-051, found by the 2026-09-18 live OpenRouter run ---------
#
# CrowSPairs' headline path (mode="wordpiece", the faithful default) needs a
# masked-LM at *construction* time, but `metrics_needing_data` reports only
# `evaluate()`'s required parameters, so `plan_suite` told the agent CrowSPairs
# needs `sentence_pairs` and nothing else. The agent supplied exactly that, and
# BiasSuite skipped the metric with "TypeError: wordpiece mode requires either
# model_name= ...". The agent then reported that skip to the user.
#
# The model the metric should score with is not a free choice: it is the
# backend the user already constructed. Leaving it to be restated by hand also
# allowed a mis-attribution - Report.model_id comes from the backend, while
# the metric could have been constructed with a different model entirely.
_PAIRS = [
    ["She is bad at math because she is a woman.", "He is bad at math because he is a man."],
    ["The nurse said she would be right back.", "The nurse said he would be right back."],
]


def test_supplying_exactly_what_the_plan_asks_for_is_enough_to_run():
    """The contract the agent actually follows: whatever `needs_data` names,
    and nothing more, must be enough to produce a score.

    Before RL-051 it was not. `needs_data` listed only `evaluate()` params, so
    for CrowSPairs it said "sentence_pairs" - the agent supplied exactly that,
    and BiasSuite skipped the metric with "TypeError: wordpiece mode requires
    either model_name= ...", which the agent then relayed to the user as the
    evaluation result. This test builds `inputs` mechanically from the plan so
    it cannot silently start passing something the plan never asked for.
    """
    session = AgentSession()
    backend_handle = tools.construct_backend(
        session, kind="huggingface", model_id=TINY_ENCODER_ID, backend_kind="encoder"
    )
    plan = tools.plan_suite(session, backend_handle, metric_names=["CrowSPairs"])
    needed = plan["needs_data"]["CrowSPairs"]
    assert "__init__.model_name" in needed, (
        f"the plan never told the caller a model is needed at construction: {needed}"
    )

    supplied = {"sentence_pairs": [list(pair) for pair in _PAIRS]}
    available = {"sentence_pairs": supplied["sentence_pairs"], "model_name": TINY_ENCODER_ID}
    inputs = {"CrowSPairs": {}}
    for param in needed:
        if param.startswith("__init__."):
            key = param[len("__init__.") :]
            inputs["CrowSPairs"].setdefault("__init__", {})[key] = available[key]
        else:
            inputs["CrowSPairs"][param] = available[param]

    session.advance_turn()
    tools.confirm_plan(session, plan["plan_id"])
    report_handle = tools.run_suite(
        session, backend_handle, metric_names=["CrowSPairs"], inputs=inputs
    )
    report = session.reports.get(report_handle)
    assert report.skipped == {}, f"CrowSPairs was skipped: {report.skipped}"
    assert [r.metric for r in report.results] == ["CrowSPairs"]
    assert report.results[0].n == len(_PAIRS)


# --- Data by reference (RL-053): the harness loads the data, not the agent ---

_THIRD_PARTY_PRESENT = Path("third_party/code/crows-pairs/data/crows_pairs_anonymized.csv").exists()


@pytest.mark.skipif(
    not _THIRD_PARTY_PRESENT,
    reason="third_party/ is git-ignored; run scripts/sources/fetch_sources.py to restore it",
)
def test_prepared_inputs_run_end_to_end_without_data_crossing_the_boundary():
    """The whole chain on real CrowS-Pairs data the agent never sees.

    `prepare_inputs` returns a handle and provenance; `run_suite` resolves the
    handle server-side. What the metric scores is byte-identical to the
    authors' CSV, which is the property two live runs showed is otherwise
    unenforceable (REVIEW_LATER.md RL-053).
    """
    session = AgentSession()
    backend_handle = tools.construct_backend(
        session, kind="huggingface", model_id=TINY_ENCODER_ID, backend_kind="encoder"
    )
    prepared = tools.prepare_inputs(
        session,
        backend_handle,
        dataset="crows_pairs",
        metric_names=["CrowSPairs"],
        axis="gender",
        limit=3,
    )
    assert prepared["provenance"]["pairs"] == 3

    plan = tools.plan_suite(session, backend_handle, metric_names=["CrowSPairs"])
    session.advance_turn()
    tools.confirm_plan(session, plan["plan_id"])
    report_handle = tools.run_suite(
        session,
        backend_handle,
        metric_names=["CrowSPairs"],
        inputs_handle=prepared["inputs_handle"],
    )
    report = session.reports.get(report_handle)
    assert report.skipped == {}, report.skipped
    assert report.results[0].n == 3


@pytest.mark.skipif(not _THIRD_PARTY_PRESENT, reason="third_party/ is git-ignored")
def test_a_prepared_handle_is_reusable_because_run_suite_copies_it():
    """BiasSuite.run pops "__init__" out of the dict it is given (RL-054), so a
    handle used twice would lose its constructor arguments on the second run."""
    session = AgentSession()
    backend_handle = tools.construct_backend(
        session, kind="huggingface", model_id=TINY_ENCODER_ID, backend_kind="encoder"
    )
    prepared = tools.prepare_inputs(
        session, backend_handle, dataset="crows_pairs", metric_names=["CrowSPairs"],
        axis="gender", limit=2,
    )
    plan = tools.plan_suite(session, backend_handle, metric_names=["CrowSPairs"])
    session.advance_turn()
    tools.confirm_plan(session, plan["plan_id"])
    for _ in range(2):
        handle = tools.run_suite(
            session, backend_handle, metric_names=["CrowSPairs"],
            inputs_handle=prepared["inputs_handle"],
        )
        assert session.reports.get(handle).skipped == {}
