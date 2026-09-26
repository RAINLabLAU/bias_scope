# --------------------------------------------------------------
# Which metrics can I run on this model?
#
# recommend_metrics answers that from the backend's *access*: what can actually
# be asked of the model. A chat API cannot give logits, so a metric that needs
# them is not offered. explain_exclusions says why each other metric was left
# out, so nothing disappears silently.
#
# StubBackend stands in for a model here, so this runs offline.
# --------------------------------------------------------------

from bias_scope import explain_exclusions, recommend_metrics
from bias_scope.backends import StubBackend

api_model = StubBackend(access=("completions", "chat"), model_id="demo/chat-api")

# Take access from the backend rather than typing it, so it cannot drift.
offered = recommend_metrics(api_model.access, axis="gender")
print(f"{len(offered)} metrics can run on {api_model.model_id}. Most trustworthy first:")
for recommendation in offered[:5]:
    print(f"  [{recommendation.fidelity:>10}] {recommendation.metric}")

# Every offered metric carries its reason, including a fidelity warning when the
# implementation deviates from its paper.
print("\nWhy the first one is offered:")
print(f"  {offered[0].reason}")

# The other half of the answer.
excluded = explain_exclusions(api_model.access)
print(f"\n{len(excluded)} metrics cannot run here. For example:")
for name, reason in list(excluded.items())[:3]:
    print(f"  {name}: {reason}")

# Narrowing the question.
embedding_only = recommend_metrics(("embeddings",), family="embedding")
print("\nA sentence encoder can run:", ", ".join(r.metric for r in embedding_only))
