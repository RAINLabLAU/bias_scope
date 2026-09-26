# --------------------------------------------------------------
# Run metrics against one model and get a report
#
# BiasSuite plans which metrics the backend can support, runs them, and returns a
# Report. It never invents data: a metric with no inputs is skipped with a
# reason, not scored as zero.
#
# This example uses random embeddings in place of a real model, so it runs
# offline. With a real encoder you would pass HuggingFaceBackend instead.
# --------------------------------------------------------------

import numpy as np

from bias_scope import BiasSuite, compare, to_markdown
from bias_scope.backends import StubBackend

backend = StubBackend(access=("embeddings",), model_id="demo/encoder")

# --- Plan without running anything ---
suite = BiasSuite(backend, axis="gender")
print("Would run:", ", ".join(name for name, _ in suite.plan()))

# --- Run one metric ---
rng = np.random.default_rng(0)
X, Y, A, B = (rng.normal(size=(8, 4)) for _ in range(4))
weat_only = BiasSuite(backend, axis="gender", metrics=["WEAT"])
report = weat_only.run(
    seed=1,
    inputs={"WEAT": {"target_embeddings": (X, Y), "attribute_embeddings": (A, B)}},
)

result = report.results[0]
print(f"\n{result}")
print(f"  score {result.score:+.3f} on n={result.n}, "
      f"{result.ci_method} interval [{result.ci[0]:+.2f}, {result.ci[1]:+.2f}]")
print(f"  fidelity: {result.info.fidelity}; protocol hash {result.protocol['hash'][:8]}")

# --- A metric with no inputs is skipped, with the reason ---
skipped = suite.run().skipped
print(f"\nWithout inputs, {len(skipped)} metrics were skipped, for example:")
for name, reason in list(skipped.items())[:2]:
    print(f"  {name}: {reason[:60]}...")

# --- Render, and compare two runs ---
print("\n" + to_markdown(report).splitlines()[0])

swapped = weat_only.run(
    seed=1,
    inputs={"WEAT": {"target_embeddings": (X, Y), "attribute_embeddings": (B, A)}},
)
for delta in compare(report, swapped):
    print(f"\n{delta.metric}: {delta.a:+.3f} -> {delta.b:+.3f} "
          f"(intervals overlap: {delta.intervals_overlap})")
