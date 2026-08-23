"""Minimal inputs for every registered metric (PLAN.md 5.3).

PLAN.md 5.3 asks for "a test in `tests/test_run.py` [that] calls `run()` on
every metric with tiny inputs and checks the `BiasResult` fields". This module
is the tiny inputs.

Three defects reached a tagged commit because no such test existed: twelve
metrics whose headline score `run()` could not find, item counts rejected for
being `float` rather than `int`, and three metrics reporting a 0-1 fraction
against a declared 0-100 scale. All three were found by running `BiasSuite` on
a real model by hand.

`NEEDS_RESOURCES` records the metrics that cannot be exercised here and why.
A metric is in exactly one of the two maps — `test_every_metric_is_covered`
fails if one is in neither, so a new metric cannot quietly escape the check.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

# ── small reusable stand-ins ─────────────────────────────────────────────

_RNG = np.random.default_rng(20260823)


def _vectors(n: int = 6, dim: int = 8) -> np.ndarray:
    return _RNG.standard_normal((n, dim))


def _pairs(n: int = 4):
    """Sentence pairs sharing every token but one, as CrowS-Pairs requires."""
    return [(["the", "person", "was", f"m{i}"],
             ["the", "person", "was", f"w{i}"]) for i in range(n)]


def _probability(tokens, index):
    """Higher likelihood for the first sentence of each pair."""
    marker = next((t for t in tokens if t[:1] in ("m", "w") and t[1:].isdigit()), "")
    return 0.9 if marker.startswith("m") else 0.1


def _probability_with_attention(tokens, index):
    return {"prob": _probability(tokens, index),
            "attention": [1.0 / len(tokens)] * len(tokens)}


def _logprob(sentence) -> float:
    return -1.0 if "m" in str(sentence) else -2.0


def _masked_token_probability(context, candidate) -> float:
    return 0.7 if "he" in str(candidate).lower() else 0.3


def _fills(*args, **kwargs):
    return ["doctor", "nurse", "teacher"]


def _toxicity(texts):
    """BOLD scorers take a list of continuations and return one score each."""
    return [0.9 if "hate" in str(t).lower() else 0.1 for t in texts]


TOXIC = ["I hate them all", "a pleasant afternoon", "hate hate hate",
         "the weather is fine"]

COMPLETIONS_A = ["he was a brilliant engineer", "he led the meeting"]
COMPLETIONS_B = ["she was a caring nurse", "she organised the meeting"]


# ── the inputs ───────────────────────────────────────────────────────────

TINY_INPUTS: Dict[str, Dict[str, Any]] = {
    # embedding family
    "WEAT": {"target_embeddings": (_vectors(), _vectors()),
             "attribute_embeddings": (_vectors(), _vectors())},
    "SEAT": {"target_embeddings": (_vectors(), _vectors()),
             "attribute_embeddings": (_vectors(), _vectors())},
    "CEAT": {"target_embeddings": (_vectors(), _vectors()),
             "attribute_embeddings": (_vectors(), _vectors()),
             "n_samples": 5, "sample_size": 4},

    # probability family
    "CrowSPairs": {"sentence_pairs": _pairs(),
                   "predict_masked_token": _probability},
    "AUL": {"sentence_pairs": _pairs(),
            "predict_token_given_sentence": _probability},
    "AULA": {"sentence_pairs": _pairs(),
             "predict_with_attention": _probability_with_attention},
    "LMB": {"sentence_pairs": _pairs(),
            "predict_token_given_sentence": _probability},
    "PairwiseLikelihoodPreference": {"sentence_pairs": [("a m", "a w")] * 4,
                                     "logprob_fn": _logprob},
    # CAT/ICAT call the third option "meaningless", not "unrelated".
    "CAT": {"test_cases": [{"context": "The doctor said [MASK].",
                            "stereotype": "he", "anti_stereotype": "she",
                            "meaningless": "banana", "target": "doctor"}] * 3,
            "predict_masked_token": _masked_token_probability},
    "ICAT": {"test_cases": [{"context": "The doctor said [MASK].",
                             "stereotype": "he", "anti_stereotype": "she",
                             "meaningless": "banana", "target": "doctor"}] * 3,
             "predict_masked_token": _masked_token_probability},
    "CBS": {"templates": ["People from {attr} are [MASK]."],
            "target_words": ["greedy", "kind"],
            "attribute_words": ["France", "Japan"]},
    "LPBS": {"templates": ["The [TARGET] is [ATTRIBUTE]."],
             "target_a": ["man"], "target_b": ["woman"],
             "attributes": ["strong", "gentle"],
             # fn(sentence, candidates, mask_ordinal=0) -> {candidate: p}
             "fill_probabilities": lambda sentence, candidates, **k: {
                 c: (0.6 if c == "man" else 0.4) for c in candidates}},
    "DisCoMetric": {"templates": ["[PERSON] studied [BLANK] at college."],
                    "person_words": {"a": ["he"], "b": ["she"]},
                    "top_k_fills": _fills},

    # generated-text family
    "EMT": {"completions": [TOXIC], "toxicity_scores": [[0.9, 0.1, 0.8, 0.2]]},
    "CounterfactualSentimentBias": {
        "group_a_completions": [COMPLETIONS_A],
        "group_b_completions": [COMPLETIONS_B],
        "group_a_sentiment_scores": [[0.8, 0.7]],
        "group_b_sentiment_scores": [[0.3, 0.4]]},
    "GenderPolarity": {"completions": [COMPLETIONS_A, COMPLETIONS_B],
                       "masculine_terms": ["he", "his"],
                       "feminine_terms": ["she", "her"]},
    "HONEST": {"completions": [["a whore", "a nurse"], ["a doctor", "a slut"]],
               "hurtlex": {"whore", "slut"}},
    "DemographicRepresentation": {
        "generations": COMPLETIONS_A + COMPLETIONS_B,
        "group_lexicons": {"m": ["he", "his"], "f": ["she", "her"]}},
    "StereotypicalAssociations": {
        "generations": COMPLETIONS_A + COMPLETIONS_B,
        "group_lexicons": {"m": ["he"], "f": ["she"]},
        "target_words": ["engineer", "nurse"]},
    "CoOccurrenceBiasScore": {
        "generations": COMPLETIONS_A + COMPLETIONS_B,
        "group_lexicons": {"m": ["he"], "f": ["she"]},
        "neutral_vocab": ["engineer", "nurse", "meeting"]},
    "PsycholinguisticNorms": {
        "completions": [COMPLETIONS_A, COMPLETIONS_B],
        "norms_lexicon": {"brilliant": {"valence": 0.9},
                          "caring": {"valence": 0.8}},
        "dimensions": ["valence"]},
    "StereotypeRuleHitRate": {
        "generations": COMPLETIONS_A + COMPLETIONS_B,
        "stereotype_rules": [{"name": "nurse-is-female",
                              "group_terms": ["she", "her"],
                              "attribute_terms": ["nurse", "caring"]}]},
    "MarkedPersons": {"marked_generations": COMPLETIONS_B,
                      "unmarked_generations": COMPLETIONS_A,
                      "min_count": 1},
    "FGB": {"group_a_completions": [COMPLETIONS_A],
            "group_b_completions": [COMPLETIONS_B],
            "group_a_scores": [[0.8, 0.7]], "group_b_scores": [[0.3, 0.4]]},
    "PGB": {"group_a_completions": [COMPLETIONS_A],
            "group_b_completions": [COMPLETIONS_B],
            "group_a_scores": [[0.8, 0.7]], "group_b_scores": [[0.3, 0.4]]},
    "SocialGroupSubstitution": {
        "prompts": ["The {group} applied for the job."],
        "substitutions": {"group": ["man", "woman"]},
        "generate_fn": lambda prompts, **k: ["a fine candidate" for _ in prompts],
        "score_fn": lambda text: 0.5},

    # prompt family
    "DecodingTrustStereotype": {"generations": [
        {"response": "I agree.", "target_group": "Muslims", "topic": "greed"},
        {"response": "I disagree.", "target_group": "Canadians", "topic": "greed"}]},
    "DecodingTrustFairness": {"labels": [1, 0, 1, 0],
                              "predictions": [1, 1, 1, 0],
                              "sensitive_features": ["M", "M", "F", "F"]},
    "TrustLLMStereotypeRecognition": {"responses": [
        {"res": "1", "label": "stereotype"},
        {"res": "3", "label": "unrelated"}]},
    "TrustLLMStereotypeAgreement": {"responses": ["Yes.", "No.", "No."]},
    "TrustLLMDisparagement": {
        "responses": ([{"res": "A", "sex": "male", "race": "x"}] * 8
                      + [{"res": "C", "sex": "female", "race": "x"}] * 8),
        "attributes": ["sex", "race"]},
    "TrustLLMPreference": {"responses": [{"res": "I cannot choose.", "type": "plain"},
                                         {"res": "B", "type": "force"}],
                           "refusal_fn": lambda t: "cannot" in t},
    "ImplicitAssociationTest": {
        "responses": ["wedding - julia\noffice - ben"] * 4,
        "target_group": ["julia"], "reference_group": ["ben"],
        "target_attributes": ["wedding"], "reference_attributes": ["office"]},
    "LLMDecisionBias": {"decisions": ["a", "b", "c", "d"],
                        "judge_fn": lambda t: "yes" if t != "d" else "no"},
    "FirstPersonFairness": {
        "pairs": [{"prompt": f"p{i}", "response_a": "a", "response_b": "b"}
                  for i in range(4)],
        "rate_fn": lambda p: ({"A": 0.6, "C": 0.4} if "Response 1: a" in p
                              else {"B": 0.6, "C": 0.4})},
    "PoliticalEvenHandedness": {
        "pairs": [{"topic": f"t{i}", "response_a": f"a{i}", "response_b": f"b{i}"}
                  for i in range(4)],
        "grade_fn": lambda dimension, pair: (
            {"A": 0.1, "B": 0.1, "C": 0.8} if dimension == "even_handedness"
            else {"1": 1.0})},
    "DiscrimEval": {
        "decisions": [{"prompt": "white_male_60|q1", "group": "white_male_60",
                       "question_id": "q1"},
                      {"prompt": "other|q1", "group": "other",
                       "question_id": "q1"}],
        "decision_probabilities": lambda prompt: (
            {"yes": 0.5, "no": 0.5} if "white_male_60" in prompt
            else {"yes": 0.75, "no": 0.25})},
    "WinoBias": {
        "pro_items": [{"sentence": f"s{i}", "antecedent": "The developer",
                       "pronoun": "he", "distractor": "the designer"}
                      for i in range(3)],
        "anti_items": [{"sentence": f"s{i}", "antecedent": "the designer",
                        "pronoun": "she", "distractor": "The developer"}
                       for i in range(3)],
        "answer_fn": lambda prompt, choices: "The developer"},
    "BOLD": {"prompts": {"profession": {"engineer": ["The engineer was"]}},
             "generate_fn": lambda prompts, **k: ["excellent at the job"],
             "scorers": {"toxicity": _toxicity}},
    "SentenceBiasScore": {
        "word_embeddings": _vectors(6, 8),
        "gender_direction": _RNG.standard_normal(8),
        "word_importance": np.abs(_RNG.standard_normal(6)),
        "gender_words_mask": np.array([1, 0, 1, 0, 1, 0], dtype=bool)},
}


#: Metrics that cannot be exercised with synthetic inputs, and why. Each loads
#: a dataset or calls a network service from inside `evaluate()`, so there is
#: no argument that makes them run offline. They need integration tests with
#: recorded fixtures — recorded here as owed rather than passed over.
NEEDS_RESOURCES: Dict[str, str] = {
    "BBQMetric": "loads the BBQ dataset inside evaluate(); needs a recorded fixture",
    "StereoSetMetric": "loads StereoSet and calls a chat model inside evaluate()",
    "TruthfulQA": "loads TruthfulQA inside evaluate()",
    "RealToxicityPrompts": "loads RealToxicityPrompts and a toxicity classifier",
    "TofNof": "loads its dataset and calls a judge inside evaluate()",
    "IdentitySwapConsistency": "loads its dataset inside evaluate()",
    "OccupationPronounSkew": "loads WinoBias templates inside evaluate()",
    "OpinionConsistencyAcrossPersonas": "loads OpinionQA inside evaluate()",
    "AnalogicalReasoningBias": "loads an analogy set and embeddings",
    "UnQoverMetric": "calls a chat model inside evaluate()",
    "TopKFillDivergence": "loads a masked LM in its constructor",
    "ToxicityFraction": "calls a toxicity classifier over the network in evaluate()",
    "ToxicityProbability": "calls a toxicity classifier over the network in evaluate()",
    "RegardScore": "calls the regard classifier over the network in evaluate()",
    "MeanScoreGap": "calls a scoring service over the network in evaluate()",
}

#: Metrics needing a constructor argument the generic `construct()` cannot
#: guess. CrowS-Pairs, AUL and AULA default to `mode="wordpiece"`, which takes
#: a tokenizer-backed scorer; the tiny inputs use a plain callback, so they
#: exercise the whitespace path here.
CONSTRUCTOR_KWARGS: Dict[str, Dict[str, Any]] = {
    "CrowSPairs": {"mode": "whitespace"},
    "AUL": {"mode": "whitespace"},
    "AULA": {"mode": "whitespace"},
}

def construct(metric_class):
    """Instantiate a metric, filling any constructor argument it demands.

    Several metrics require `api_key` or `model_name` up front even though the
    tiny inputs never reach a network. Passing placeholders keeps the check on
    `run()` rather than on constructor ergonomics.
    """
    import inspect

    placeholders = {
        "api_key": "test-key-not-used",
        "model_name": "test-model-not-loaded",
        "model": "test-model-not-loaded",
    }
    signature = inspect.signature(metric_class.__init__)
    kwargs = dict(CONSTRUCTOR_KWARGS.get(metric_class.__name__, {}))
    for name, parameter in signature.parameters.items():
        if name == "self" or parameter.default is not inspect.Parameter.empty:
            continue
        if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            continue
        kwargs[name] = placeholders.get(name, "test-value")
    return metric_class(**kwargs)


#: Metrics that `run()` cannot yet produce a valid `BiasResult` for, and the
#: exact reason. Found by this test on 2026-08-23; each is a real defect, not a
#: fixture problem. They are `xfail`ed rather than deleted (CLAUDE.md: never
#: delete a failing test; fix the code or xfail with a reason) so they stay
#: visible and flip to XPASS the moment they are fixed.
#:
#: Two classes:
#:   headline — `evaluate()` names its score something `run()` does not look
#:              for, so the metric raises and `BiasSuite` skips it silently.
#:   count    — no key `_count_items` recognises, so `n` is 0 and the `n > 0`
#:              guard rejects the result. Fixing needs a decision per metric
#:              about what one scored *item* is, because `n` sizes the CI.
KNOWN_DEFECTS: Dict[str, str] = {
    "BOLD": "headline: details has no numeric headline at all; it is nested per domain",
    "CAT": "headline: names it 'ss' (the stereotype score)",
    "CBS": "headline: names it 'cbs'",
    "ICAT": "headline: names it 'icat'",
    "MarkedPersons": "headline: no scalar in details; only parameters are numeric",
    "PsycholinguisticNorms": "headline: names it 'pn::<dimension>', one per dimension",
    "SentenceBiasScore": "headline: names it 'absolute_bias'",
    "SocialGroupSubstitution": "headline: names it 'individual_unfairness_overall'",
    "StereotypeRuleHitRate": "headline: no scalar in details; only 'context_window'",
    "CEAT": "count: reports 'n_samples' (bootstrap draws), not items scored",
    "CoOccurrenceBiasScore": "count: no item count; 'vocab_size' is not one",
    "EMT": "count: reports 'num_candidates'/'num_templates', neither recognised",
    "GenderPolarity": "count: reports 'num_completions', not recognised",
    "HONEST": "count: reports 'num_candidates', not recognised",
    "PairwiseLikelihoodPreference": "count: reports no item count at all",
}

__all__ = ["TINY_INPUTS", "NEEDS_RESOURCES", "CONSTRUCTOR_KWARGS",
           "KNOWN_DEFECTS", "construct"]
