import math

import pytest

from bias_scope.prompts_based.truthfulqa import ReferenceOverlapTruthfulness, TruthfulQA


def test_generation_uses_official_judge_thresholds_and_separates_metrics():
    result = TruthfulQA().evaluate_generation(
        [
            {
                "question_id": "q1",
                "truthfulness_probability": 0.5,
                "informativeness_probability": 0.2,
            },
            {
                "question_id": "q2",
                "truthfulness_probability": 0.2,
                "informativeness_probability": 0.9,
            },
        ],
        return_details=True,
    )
    assert result["generation_truthfulness"] == 0.5
    assert result["generation_informativeness"] == 0.5
    assert result["generation_truthful_and_informative"] == 0
    assert result["per_question"]["q1"]["truthful"] is True


def test_generation_accepts_refusal_and_injected_scorer_but_excludes_bad_results():
    def scorer(row):
        return {
            **row,
            "truthfulness_probability": 0.8,
            "informativeness_probability": 0.1,
        }
    result = TruthfulQA().evaluate_generation(
        [{"question_id": "refusal", "answer": "I have no comment."}], scorer=scorer
    )
    assert result["num_valid_questions"] == 1
    assert result["scoring_provenance"] == "injected_scorer_adaptation"
    with pytest.raises(ValueError):
        TruthfulQA().evaluate_generation(
            [
                {
                    "question_id": "bad",
                    "truthfulness_probability": 2,
                    "informativeness_probability": 0.2,
                }
            ]
        )


def test_mc1_mc2_match_official_likelihood_formula_stably():
    result = TruthfulQA().evaluate_multiple_choice(
        [
            {
                "question_id": "q1",
                "true_logprobs": [-1000, -1001],
                "false_logprobs": [-1002, -1003],
                "best_true_index": 0,
            },
            {
                "question_id": "q2",
                "true_logprobs": [-3, -4],
                "false_logprobs": [-2],
                "best_true_index": 0,
            },
        ],
        return_details=True,
    )
    assert result["mc1"] == 0.5
    assert result["per_question"]["q1"]["mc2"] == pytest.approx(
        (1 + math.exp(-1)) / (1 + math.exp(-1) + math.exp(-2) + math.exp(-3))
    )


@pytest.mark.parametrize(
    "record",
    [
        {"question_id": "q", "true_logprobs": [], "false_logprobs": [0], "best_true_index": 0},
        {"question_id": "q", "true_logprobs": [0], "false_logprobs": [0], "best_true_index": 2},
        {
            "question_id": "q",
            "true_logprobs": [float("nan")],
            "false_logprobs": [0],
            "best_true_index": 0,
        },
    ],
)
def test_mc_rejects_invalid_answer_scores(record):
    with pytest.raises(ValueError):
        TruthfulQA().evaluate_multiple_choice([record])


def test_generation_and_mc_are_separate_apis():
    metric = TruthfulQA()
    assert "generation_truthfulness" in metric.evaluate_generation(
        [{"question_id": "q", "truthfulness_probability": 1, "informativeness_probability": 1}]
    )
    assert "mc1" in metric.evaluate_multiple_choice(
        [{"question_id": "q", "true_logprobs": [0], "false_logprobs": [-1], "best_true_index": 0}]
    )


def test_legacy_token_f1_diagnostic_is_preserved_and_not_named_published_metric():
    result = ReferenceOverlapTruthfulness.evaluate(
        [
            {
                "question_id": "q",
                "answer": "Paris",
                "correct_answers": ["Paris"],
                "incorrect_answers": ["London"],
            }
        ]
    )
    assert result["reference_overlap_truthfulness"] == 1
    assert result["per_question"]["q"]["margin"] > 0


def test_duplicate_and_missing_generation_ids_are_rejected():
    with pytest.raises(ValueError):
        TruthfulQA().evaluate_generation(
            [
                {
                    "question_id": "q",
                    "truthfulness_probability": 1,
                    "informativeness_probability": 1,
                },
                {
                    "question_id": "q",
                    "truthfulness_probability": 1,
                    "informativeness_probability": 1,
                },
            ]
        )


def test_generation_preflights_duplicate_ids_before_scorer_even_after_invalid_input():
    calls = []

    def scorer(row):
        calls.append(row["question_id"])
        return {"truthfulness_probability": 0.8, "informativeness_probability": 0.8}

    with pytest.raises(ValueError, match="unique non-empty input"):
        TruthfulQA().evaluate_generation(
            [
                {"question_id": "q", "truthfulness_probability": "bad"},
                {
                    "question_id": "q",
                    "truthfulness_probability": 0.8,
                    "informativeness_probability": 0.8,
                },
            ],
            scorer=scorer,
        )
    assert calls == []


def test_scorer_failure_is_excluded_and_question_identity_cannot_be_replaced():
    def scorer(row):
        if row["question_id"] == "bad":
            raise RuntimeError("unavailable")
        return {
            "question_id": "good",
            "truthfulness_probability": 0.9,
            "informativeness_probability": 0.7,
        }

    result = TruthfulQA().evaluate_generation(
        [{"question_id": "bad"}, {"question_id": "good"}], scorer=scorer, return_details=True
    )
    assert result["num_valid_questions"] == 1
    assert result["excluded"] == [{"question_id": "bad", "reason": "scorer_error: RuntimeError"}]
    assert set(result["per_question"]) == {"good"}


def test_scorer_returning_a_different_question_id_is_excluded():
    with pytest.raises(ValueError, match="undefined"):
        TruthfulQA().evaluate_generation(
            [{"question_id": "original"}],
            scorer=lambda _: {
                "question_id": "replacement",
                "truthfulness_probability": 1,
                "informativeness_probability": 1,
            },
        )


def test_mc_permits_negative_infinity_but_marks_all_zero_mass_undefined():
    result = TruthfulQA().evaluate_multiple_choice(
        [
            {
                "question_id": "one_zero",
                "true_logprobs": [-float("inf"), -1],
                "false_logprobs": [-2],
                "best_true_index": 1,
            },
            {
                "question_id": "all_zero",
                "true_logprobs": [-float("inf")],
                "false_logprobs": [-float("inf")],
                "best_true_index": 0,
            },
        ],
        return_details=True,
    )
    assert result["per_question"]["one_zero"]["mc2"] == pytest.approx(
        math.exp(-1) / (math.exp(-1) + math.exp(-2))
    )
    assert result["per_question"]["all_zero"]["mc2"] is None
    assert result["num_defined_mc2_questions"] == 1
    undefined = TruthfulQA().evaluate_multiple_choice(
        [
            {
                "question_id": "none",
                "true_logprobs": [-float("inf")],
                "false_logprobs": [-float("inf")],
                "best_true_index": 0,
            }
        ]
    )
    assert undefined["mc2"] is None and undefined["num_defined_mc2_questions"] == 0
