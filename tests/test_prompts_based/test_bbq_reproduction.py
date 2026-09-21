"""Deterministic parity tests for private BBQ released-artifact scoring."""

import csv
import hashlib
import json
import os
from pathlib import Path

import pytest

from bias_scope.prompts_based._bbq_reproduction import (
    OFFICIAL_CATEGORIES,
    add_correct_answer_alignment,
    artifact_provenance,
    attach_predictions,
    disambiguated_accuracy_by_alignment,
    group_records,
    join_additional_metadata,
    load_additional_metadata,
    load_official_data,
    normalize_reference_answer,
    normalize_reference_prediction,
    normalize_unifiedqa_prediction,
    roberta_deberta_predictions,
    score_predictions,
    strict_max_option,
    unifiedqa_predictions,
    unknown_option_index,
)


def row(**changes):
    base = {
        "example_id": 1,
        "question_index": "1",
        "question_polarity": "neg",
        "context_condition": "ambig",
        "category": "Gender_identity",
        "context": "ctx",
        "question": "question",
        "ans0": "The man",
        "ans1": "Unknown",
        "ans2": "The woman",
        "label": 1,
        "answer_info": {
            "ans0": ["The man", "man"],
            "ans1": ["Unknown", "unknown"],
            "ans2": ["The woman", "woman"],
        },
        "additional_metadata": {},
        "target_loc": 2,
        "model": "m",
    }
    base.update(changes)
    return base


def metadata(record, **changes):
    base = {
        "example_id": str(record["example_id"]),
        "category": record["category"],
        "question_index": str(record["question_index"]),
        "target_loc": str(record["target_loc"]),
    }
    base.update(changes)
    return base


def predicted(record, option):
    result = dict(record)
    result["predicted_option_index"] = option
    return result


class TestOfficialData:
    def test_loads_all_categories_and_preserves_order(self, tmp_path):
        for category in OFFICIAL_CATEGORIES:
            record = row(category=category)
            (tmp_path / f"{category}.jsonl").write_text(
                json.dumps(record) + "\n", encoding="utf-8"
            )
        rows, provenance = load_official_data(tmp_path)
        assert len(rows) == len(OFFICIAL_CATEGORIES)
        assert rows[0]["ans0"] == "The man"
        assert set(provenance) == set(OFFICIAL_CATEGORIES)
        assert (
            provenance["Age"]["source_status"]
            == "caller_supplied_unverified_official_identity"
        )

    @pytest.mark.parametrize("unknown", (0, 1, 2))
    def test_unknown_metadata_can_be_any_option(self, unknown):
        record = row()
        record["answer_info"] = {
            f"ans{i}": [record[f"ans{i}"], "unknown" if i == unknown else "label"]
            for i in range(3)
        }
        assert unknown_option_index(record) == unknown

    def test_metadata_join_uses_three_part_key_and_filters_missing_target(self):
        first, second = row(example_id=1), row(example_id=2)
        joined, removed = join_additional_metadata(
            [first, second], [metadata(first, label_type="name")]
        )
        assert len(joined) == 1 and joined[0]["label_type"] == "name"
        assert removed == 1

    def test_metadata_csv_and_sha(self, tmp_path):
        path = tmp_path / "additional_metadata.csv"
        path.write_text(
            "example_id,category,question_index,target_loc\n1,Gender_identity,1,2\n",
            encoding="utf-8",
        )
        loaded, provenance = load_additional_metadata(path)
        assert loaded[0]["target_loc"] == "2"
        assert provenance["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


class TestScoring:
    def test_target_loc_is_direct_for_both_polarities(self):
        neg = predicted(row(question_polarity="neg", target_loc=2), 2)
        nonneg = predicted(row(question_polarity="nonneg", target_loc=0), 0)
        result = score_predictions([neg, nonneg])[0]
        assert result["count_target_selected"] == 2
        assert result["raw_bias_score"] == 1.0

    @pytest.mark.parametrize(
        ("answers", "expected"), (([2, 2], 1.0), ([2, 0], 0.0), ([0, 0], -1.0))
    )
    def test_raw_signed_scores(self, answers, expected):
        records = [
            predicted(row(example_id=i), answer) for i, answer in enumerate(answers)
        ]
        assert score_predictions(records)[0]["raw_bias_score"] == expected

    def test_unknown_is_excluded_and_zero_denominator_is_explicit(self):
        records = [predicted(row(example_id=1), 1), predicted(row(example_id=2), 1)]
        result = score_predictions(records)[0]
        assert result["count_unknown_prediction"] == 2
        assert result["raw_bias_score"] is None and result["bias_score"] is None

    def test_unresolved_prediction_is_reported_but_not_scored(self):
        unresolved = predicted(row(example_id=1), None)
        resolved = predicted(row(example_id=2), 2)
        result = score_predictions([unresolved, resolved])[0]
        assert result["count_total"] == 2
        assert result["count_scored"] == 1
        assert result["count_unresolved_prediction"] == 1
        assert result["count_unknown_prediction"] == 0
        assert result["accuracy"] == 0.0

    def test_ambiguous_scaling_and_disambiguated_unscaled(self):
        ambig = [predicted(row(example_id=1), 1), predicted(row(example_id=2), 2)]
        result = score_predictions(ambig)[0]
        assert (
            result["accuracy"] == 0.5
            and result["raw_bias_score"] == 1.0
            and result["bias_score"] == 0.5
        )
        disambig = predicted(row(context_condition="disambig", label=2), 2)
        assert score_predictions([disambig])[0]["bias_score"] == 1.0

    def test_name_and_intersectional_metadata_are_retained(self):
        name = predicted(row(label_type="name"), 2)
        gender = predicted(
            row(
                example_id=2,
                category="Race_x_gender",
                Known_stereotyped_race="Black",
                Known_stereotyped_var2="F",
                corr_ans_aligns_race="1",
                corr_ans_aligns_var2="0",
                full_cond="Mismatch Race\n Match Gender",
            ),
            2,
        )
        ses = predicted(
            row(
                example_id=3,
                category="Race_x_SES",
                Known_stereotyped_race="Black",
                Known_stereotyped_var2="lowSES",
                full_cond="Mismatch Race\n Match SES",
            ),
            2,
        )
        results = score_predictions([name, gender, ses])
        assert any(item["category"] == "Gender_identity (names)" for item in results)
        assert len(group_records([gender, ses], "full_cond")) == 2

    def test_disambiguated_alignment_helper_uses_provided_field(self):
        records = [
            predicted(
                row(context_condition="disambig", label=2, corr_ans_aligns_race="1"), 2
            ),
            predicted(
                row(
                    example_id=2,
                    context_condition="disambig",
                    label=2,
                    corr_ans_aligns_race="0",
                ),
                0,
            ),
        ]
        assert disambiguated_accuracy_by_alignment(records, "corr_ans_aligns_race") == {
            "1": 1.0,
            "0": 0.0,
        }

    def test_ordinary_alignment_compares_gold_to_final_metadata_target(self):
        aligned = row(label=2, target_loc=2)
        nonaligned = row(example_id=2, label=0, target_loc=2)
        annotated = add_correct_answer_alignment([aligned, nonaligned])
        assert [item["correct_answer_aligns_target"] for item in annotated] == [1, 0]


class TestReleasedParsers:
    def test_unifiedqa_normalization_and_question_only(self):
        record = row(
            **{
                "unifiedqa-t5-11b_pred_arc": "The woman.",
                "unifiedqa-t5-11b_pred_race": "The man",
                "unifiedqa-t5-11b_pred_qonly": "Unknown",
            }
        )
        assert normalize_unifiedqa_prediction("The woman.", record) == 2
        assert (
            unifiedqa_predictions([record], "unifiedqa-t5-11b_pred_arc")[0][
                "predicted_option_index"
            ]
            == 2
        )
        assert (
            unifiedqa_predictions([record], "unifiedqa-t5-11b_pred_qonly")[0][
                "predicted_option_index"
            ]
            == 1
        )
        assert normalize_unifiedqa_prediction("unmatched", record) is None

    def test_unifiedqa_reference_transforms_are_asymmetric(self):
        answer_brace = row(ans2="The woman}")
        assert normalize_unifiedqa_prediction("The woman", answer_brace) == 2
        assert normalize_reference_answer("The woman}.") == "The woman"
        pantsuit = row(ans0="some pantsuit")
        assert normalize_unifiedqa_prediction("some pantsu", pantsuit) == 0
        period = row(ans0="The man.")
        assert normalize_unifiedqa_prediction("The man.", period) == 0
        obrien = row(ans0="obrien")
        assert normalize_unifiedqa_prediction("o'brien", obrien) == 0
        assert normalize_reference_prediction("some pantsu") == "some pantsuit"

    def test_unifiedqa_fallback_uses_first_two_words_in_option_order(self):
        record = row(
            ans0="first choice text",
            ans1="second choice text",
            ans2="third choice text",
            answer_info={
                "ans0": ["first choice text", "a"],
                "ans1": ["second choice text", "b"],
                "ans2": ["third choice text", "c"],
            },
        )
        assert (
            normalize_unifiedqa_prediction("first choice and second choice", record)
            == 0
        )

    def test_roberta_strict_max_and_tie(self, tmp_path):
        assert strict_max_option([3, 2, 1]) == 0
        assert strict_max_option([1, 3, 2]) == 1
        assert strict_max_option([1, 2, 3]) == 2
        assert strict_max_option([3, 3, 1]) is None
        path = tmp_path / "df_bbq.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=("index", "cat", "model", "ans0", "ans1", "ans2")
            )
            writer.writeheader()
            writer.writerow(
                {
                    "index": 1,
                    "cat": "Gender_identity",
                    "model": "roberta",
                    "ans0": 1,
                    "ans1": 2,
                    "ans2": 2,
                }
            )
        predictions, _ = roberta_deberta_predictions(path)
        assert predictions[0]["predicted_option_index"] is None

    def test_csv_predictions_attach_without_guessing_question_index(self):
        records, unresolved = attach_predictions(
            [row()],
            [
                {
                    "example_id": 1,
                    "category": "Gender_identity",
                    "model": "r",
                    "predicted_option_index": 0,
                }
            ],
        )
        assert not unresolved and records[0]["predicted_option_index"] == 0

    def test_csv_predictions_preserve_multiple_models_for_one_example(self):
        records, unresolved = attach_predictions(
            [row()],
            [
                {
                    "example_id": 1,
                    "category": "Gender_identity",
                    "model": "roberta-base",
                    "predicted_option_index": 0,
                },
                {
                    "example_id": 1,
                    "category": "Gender_identity",
                    "model": "deberta-large",
                    "predicted_option_index": 2,
                },
            ],
        )
        assert unresolved == 0
        assert {
            (item["model"], item["predicted_option_index"]) for item in records
        } == {
            ("roberta-base", 0),
            ("deberta-large", 2),
        }
        assert {item["model"] for item in score_predictions(records)} == {
            "roberta-base",
            "deberta-large",
        }

    def test_target_json_is_schema_shaped(self):
        target = Path("src/bias_scope/prompts_based/bbq_targets.json")
        content = json.loads(target.read_text(encoding="utf-8"))
        assert set(content) == {
            "paper_published_targets",
            "official_result_parity_targets",
        }

    def test_artifact_provenance_never_authenticates_caller_path(self, tmp_path):
        artifact = tmp_path / "artifact"
        artifact.write_text("x", encoding="utf-8")
        assert (
            artifact_provenance(artifact)["source_status"]
            == "caller_supplied_unverified_official_identity"
        )

    def test_official_root_is_opt_in(self):
        """Normal tests do not fetch or require external BBQ artifacts."""
        root = os.environ.get("BBQ_OFFICIAL_ROOT")
        if root:
            assert (Path(root) / "data").is_dir()
