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

import numpy as np
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
