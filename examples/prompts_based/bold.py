"""Offline use of the public BOLD orchestration adaptation.

The public class accepts arbitrary scalar scorers. Its means and gaps are
BiasScope diagnostics, not the paper's categorical BOLD results.
"""

from __future__ import annotations

import sys
from pathlib import Path

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / "src"))

from bias_scope.prompts_based.bold import BOLD  # noqa: E402

prompts = {
    "gender": {
        "male": ["Person was known for"],
        "female": ["Person was admired for"],
    }
}


def generate(prompt: str) -> str:
    return f"{prompt} being thoughtful."


def demo_scalar_scorer(texts: list[str]) -> list[float]:
    return [float("thoughtful" in text) for text in texts]


result = BOLD(model_name="offline-demo").evaluate(
    prompts=prompts,
    generate_fn=generate,
    scorers={"sentiment": demo_scalar_scorer},
)
print(result["scores"])
print("BiasScope-only gaps:", result["gaps"])
