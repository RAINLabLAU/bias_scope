"""Offline Abid-style analogy aggregation; no API call is made."""
from bias_scope.prompts_based import AnalogicalReasoningBias, CounterfactualAnalogyDiagnostic

probe = AnalogicalReasoningBias()
print(probe.evaluate(2, precomputed_completions={
    "Muslim": ["terrorism", "mosque"], "Christian": ["faithfulness", "church"],
    "Sikh": ["turban", "faith"], "Jewish": ["money", "Judaism"],
    "Buddhist": ["enlightenment", "temple"], "Atheist": ["godless", "reason"],
})["groups"]["Muslim"]["noun_distribution"])

custom = CounterfactualAnalogyDiagnostic("unused-offline")
prompt = custom.ANALOGY_TEMPLATES[0]["prompt"]
print(custom.evaluate(1, precomputed_completions={prompt: ["homemaker", "programmer"]}))
