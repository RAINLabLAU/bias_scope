"""Offline scoring for original-2022 TruthfulQA precomputed results."""
from bias_scope.prompts_based.truthfulqa import TruthfulQA

metric = TruthfulQA()
print(metric.evaluate_generation([
    {"question_id": "q1", "truthfulness_probability": .8, "informativeness_probability": .9},
], return_details=True))
print(metric.evaluate_multiple_choice([
    {"question_id": "q1", "true_logprobs": [-2.0, -4.0], "false_logprobs": [-3.0], "best_true_index": 0},
], return_details=True))
