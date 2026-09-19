"""Offline use of published OpinionQA Consistency with precomputed inputs.

For paper-compatible inputs, derive these records from the official OpinionQA
release with ``records_from_official_csv``; no model API call is made here.
"""
from bias_scope.prompts_based import OpinionConsistencyAcrossPersonas

models = [
    {"question_id": "Q1", "distribution": [0.75, 0.25], "ordinal": [0, 1], "topic": "Politics"},
    {"question_id": "Q2", "distribution": [0.25, 0.75], "ordinal": [0, 1], "topic": "Economy"},
]
humans = [
    {"question_id": q, "attribute": "POLPARTY", "subgroup": "Democrat", "subgroup_order": 0, "distribution": democrat}
    for q, democrat in [("Q1", [0.8, 0.2]), ("Q2", [0.2, 0.8])]
] + [
    {"question_id": q, "attribute": "POLPARTY", "subgroup": "Republican", "subgroup_order": 1, "distribution": republican}
    for q, republican in [("Q1", [0.2, 0.8]), ("Q2", [0.8, 0.2])]
]

result = OpinionConsistencyAcrossPersonas().evaluate(models, humans, return_details=True)
print(result["opinion_consistency"])
