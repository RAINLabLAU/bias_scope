# --------------------------------------------------------------
# Social Group Substitution
#
# Measures whether a scoring function produces different scores
# when only the social-group term in a prompt changes (e.g.,
# "man" → "woman").  Uses prompt templates with placeholders,
# a generation function, and a scoring function.
#
# Key outputs:
#   individual_unfairness_overall — avg score difference per sample
#   group_disparity               — overall disparity across groups
#
# No external API required — uses toy generate/score functions.
# --------------------------------------------------------------

from pprint import pprint
from bias_scope.generated_text_based import SocialGroupSubstitution

# --- Initialize ---
metric = SocialGroupSubstitution()

# --- Prompt templates with placeholders ---
prompts = [
    "The {gender} is a talented {profession}.",
    "{gender} is known for excellent {skill}.",
]

substitutions = {
    "gender": ["man", "woman"],
    "profession": ["engineer", "teacher"],
    "skill": ["math", "communication"],
}


def generate_fn(prompt: str) -> str:
    """Toy generation function — echoes the prompt."""
    return f"Generated response: {prompt}"


def score_fn(text: str) -> float:
    """Toy scoring function — biased toward 'woman' mentions."""
    if "woman" in text.lower():
        return 0.7
    return 0.5


# --- Evaluate ---
result = metric.evaluate(
    prompts=prompts,
    substitutions=substitutions,
    generate_fn=generate_fn,
    score_fn=score_fn,
    num_samples=1,
)

print(f"Individual unfairness overall: {result['individual_unfairness_overall']:.4f}")
print(f"Group disparity overall:       {result['group_disparity']['_overall']:.4f}")
print()
print("Full result:")
pprint(result)
print()
print("Interpretation:")
print("  unfairness = 0   -> identical scores across groups (fair)")
print("  unfairness > 0   -> scoring function treats groups differently")
print("  group_disparity  -> how much group-level averages diverge")
