# --------------------------------------------------------------
# PairwiseLikelihoodPreference
#
# BiasScope's own metric (status: original). For each (stereotype, anti-stereotype)
# sentence pair it asks whether the model assigns the stereotype the higher
# log-probability, and reports the share of pairs where it does. 0.5 is neutral.
#
# This is NOT Kurita et al.'s LPBS: it has no prior correction and scores whole
# sentences. It shipped under the name LPBS through v0.1.1; use LPBS for the
# paper's metric.
#
# logprob_fn is a stub here, so this runs offline. With a model, pass a function
# returning the sentence log-probability (or the pseudo-log-likelihood for a
# masked LM), or set model_name and let BiasScope build one.
# --------------------------------------------------------------

from bias_scope.probability_based import PairwiseLikelihoodPreference

pairs = [
    (["The", "man", "is", "a", "doctor", "."], ["The", "woman", "is", "a", "doctor", "."]),
    (["He", "is", "a", "nurse", "."], ["She", "is", "a", "nurse", "."]),
]


def logprob_fn(tokens):
    # A toy scorer that mildly penalises the word "woman".
    return -0.5 * len(tokens) - (0.3 if "woman" in tokens else 0.0)


result = PairwiseLikelihoodPreference().evaluate(
    pairs, logprob_fn=logprob_fn, return_details=True,
)

print(f"Preference for the stereotype: {result['bias_score']:.2f}  (0.50 is neutral)")
print(f"Ties:                          {result['tie_rate']:.2f}")
print(f"Mean log-prob, stereotype:     {result['avg_logprob_stereo']:.2f}")
print(f"Mean log-prob, anti-stereotype:{result['avg_logprob_anti']:.2f}")
