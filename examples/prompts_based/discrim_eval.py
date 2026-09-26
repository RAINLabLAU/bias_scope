# --------------------------------------------------------------
# DiscrimEval (Tamkin et al. 2023)
#
# Asks the same yes/no decision question with the demographic details changed
# and compares the model's probability of "yes" across groups, on the logit
# scale. 0 means no discrimination; the headline is the largest absolute gap
# from the baseline group, so one badly treated group cannot hide in an average.
#
# This example supplies the probabilities directly, so it runs offline. With a
# real model, decision_probabilities would return the model's P(yes) / P(no).
# --------------------------------------------------------------

from bias_scope.prompts_based import DiscrimEval

BASELINE = "white_male_60"  # the paper's baseline group; the default
OTHER = "black_female_30"

decisions = [
    {"prompt": f"[{group}] question {q}: should the loan be approved?",
     "group": group, "question_id": q}
    for q in range(3)
    for group in (BASELINE, OTHER)
]

P_YES = {BASELINE: 0.70, OTHER: 0.55}  # a model that favours the baseline group


def decision_probabilities(prompt):
    group = BASELINE if BASELINE in prompt else OTHER
    return {"yes": P_YES[group], "no": 1.0 - P_YES[group]}


result = DiscrimEval().evaluate(
    decisions,
    decision_probabilities=decision_probabilities,
    return_details=True,
)

print(f"Largest absolute discrimination score: {result['bias_score']:.3f}")
for group, score in result["discrimination_scores"].items():
    print(f"  {group:>16s}: {score:+.3f}")
print(f"Most favoured:  {result['most_favoured']}")
print(f"Least favoured: {result['least_favoured']}")
print("Positive means favoured relative to the baseline; negative means disfavoured.")
