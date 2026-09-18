import hashlib

import pytest

from bias_scope.prompts_based import OpinionConsistencyAcrossPersonas
from bias_scope.prompts_based._opinionqa_parity import (
    NOTEBOOK_RECONSTRUCTION,
    PAPER_FORMULA,
    build_manifest,
    file_description,
    load_all_combined,
    reconstruct,
    reference_alignment,
)


def row(q, topic, group, dm, dh, order, index, attribute="PARTY", wd=None, path="r"):
    return {
        "row_index": index,
        "source_row_id": index,
        "survey_wave": 1,
        "original_qkey": q,
        "question_id": q,
        "question": q,
        "topic": topic,
        "attribute": attribute,
        "group": group,
        "group_order": order,
        "distribution_model": dm,
        "distribution_human": dh,
        "ordinal": [0, 2],
        "model_identity": {
            "model_name": "m",
            "context_type": "default",
            "results_path": path,
            "model_order": "0",
        },
        "notebook_wd": 1 - reference_alignment(dm, dh, [0, 2])["alignment"] if wd is None else wd,
    }


def test_reference_wasserstein_examples_match_public_metric():
    assert reference_alignment([1, 0], [1, 0], [0, 2])["alignment"] == 1
    assert reference_alignment([1, 0], [0, 1], [0, 2])["alignment"] == 0
    assert reference_alignment([0.5, 0.5], [1, 0], [0, 2])["alignment"] == 0.5
    assert reference_alignment([1, 0, 0], [0, 0, 1], [1, 2, 1.5])["alignment"] == pytest.approx(0.5)
    assert OpinionConsistencyAcrossPersonas.alignment([0.5, 0.5], [1, 0], [0, 2]) == 0.5


def test_paper_formula_equal_topics_excludes_overall_and_notebook_is_question_weighted():
    rows = [
        row("q1", "large", "A", [1, 0], [1, 0], 0, 0),
        row("q1", "large", "B", [1, 0], [0, 1], 1, 1),
        row("q2", "large", "A", [1, 0], [1, 0], 0, 2),
        row("q2", "large", "B", [1, 0], [0, 1], 1, 3),
        row("q3", "small", "A", [0, 1], [1, 0], 0, 4),
        row("q3", "small", "B", [0, 1], [0, 1], 1, 5),
        row("q1", "large", "Overall", [1, 0], [1, 0], 0, 6, "Overall"),
        row("q2", "large", "Overall", [1, 0], [1, 0], 0, 7, "Overall"),
        row("q3", "small", "Overall", [0, 1], [0, 1], 0, 8, "Overall"),
    ]
    paper, notebook = (
        reconstruct(rows, mode=PAPER_FORMULA),
        reconstruct(rows, mode=NOTEBOOK_RECONSTRUCTION),
    )
    assert (
        paper["attributes"]["PARTY"]["overall_best_subgroup"] == "B"
    )  # equal topic tie, high order
    assert notebook["attributes"]["PARTY"]["overall_best_subgroup"] == "A"  # two large questions
    assert paper["included_attributes"] == ["PARTY"]
    assert "Overall" in notebook["included_attributes"]


def test_manifest_description_has_hash_rows_and_schema(tmp_path):
    p = tmp_path / "official.csv"
    p.write_text("a,b\n1,2\n", encoding="utf-8")
    description = file_description(p)
    assert description["row_count"] == 1 and description["columns"] == ["a", "b"]
    assert description["sha256"] == hashlib.sha256(p.read_bytes()).hexdigest()


def test_private_helpers_are_not_public_exports():
    import bias_scope.prompts_based as prompts

    assert "_opinionqa_parity" not in prompts.__all__


def test_missing_group_topic_is_excluded_not_zero_filled():
    rows = [
        row("q", "one", "A", [1, 0], [1, 0], 0, 0),
        row("q", "one", "B", [1, 0], [0, 1], 1, 1),
        row("x", "missing", "A", [1, 0], [1, 0], 0, 2),
    ]
    result = reconstruct(rows, mode=PAPER_FORMULA)
    assert "missing" not in result["attributes"]["PARTY"]["topics"]
    assert any(x["reason"] == "missing_subgroup_topic" for x in result["excluded"])


def test_notebook_requires_csv_wd():
    with pytest.raises(ValueError, match="requires normalized WD"):
        reconstruct([row("q", "t", "A", [1, 0], [1, 0], 0, 0, wd="")], mode=NOTEBOOK_RECONSTRUCTION)


def test_all_waves_use_explicit_official_order(tmp_path):
    import numpy as np

    root = tmp_path / "data"
    distributions = root / "distributions"
    distributions.mkdir(parents=True)
    np.save(root / "topics.npy", {"Q26": {"cg": ["t"]}, "Q27": {"cg": ["t"]}})
    header = "qkey,question,D_M,D_H,ordinal,attribute,group,group_order,model_name,WD\n"
    for wave in (27, 26):
        (distributions / f"American_Trends_Panel_W{wave}_default_combined.csv").write_text(
            header + f'Q{wave},Q{wave},"[1 0]","[1 0]","[0 1]",PARTY,A,0,m,0\n', encoding="utf-8"
        )
    mapping = {"Q26": {"cg": ["t"]}, "Q27": {"cg": ["t"]}}
    records, descriptions = load_all_combined(root, mapping, model="m", allow_partial=True)
    assert [item["survey_wave"] for item in descriptions] == [26, 27]
    assert [item["question_id"] for item in records] == ["W26:Q26", "W27:Q27"]
    assert [item["original_qkey"] for item in records] == ["Q26", "Q27"]


def test_per_wave_results_paths_are_accepted_but_mixed_models_fail():
    rows = [
        row("q1", "t", "A", [1, 0], [1, 0], 0, 0, path="wave1"),
        row("q2", "t", "A", [1, 0], [1, 0], 0, 1, path="wave2"),
    ]
    assert reconstruct(rows, mode=PAPER_FORMULA)["model_identity"]["results_paths"] == [
        "wave1",
        "wave2",
    ]
    rows[1]["model_identity"]["model_name"] = "other"
    with pytest.raises(ValueError, match="model_name/model_order"):
        reconstruct(rows, mode=PAPER_FORMULA)


def test_public_comparison_uses_one_stable_identity_across_wave_paths():
    from bias_scope.prompts_based._opinionqa_parity import compare_with_public_metric

    rows = [
        row("q1", "t", "A", [1, 0], [1, 0], 0, 0, path="wave1"),
        row("q2", "t", "A", [1, 0], [1, 0], 0, 1, path="wave2"),
    ]
    reference = reconstruct(rows, mode=PAPER_FORMULA)
    assert compare_with_public_metric(rows, reference)["absolute_difference"] == 0


def test_public_comparison_deduplicates_multi_topic_humans():
    from bias_scope.prompts_based._opinionqa_parity import compare_with_public_metric

    rows = [
        row("q", "t1", "A", [1, 0], [1, 0], 0, 0),
        row("q", "t1", "B", [1, 0], [0, 1], 1, 1),
        row("q", "t2", "A", [1, 0], [1, 0], 0, 0),
        row("q", "t2", "B", [1, 0], [0, 1], 1, 1),
    ]
    reference = reconstruct(rows, mode=PAPER_FORMULA)
    assert compare_with_public_metric(rows, reference)["absolute_difference"] == 0


def test_notebook_overall_uses_each_source_row_not_collapsed_qkey():
    rows = [
        row("same", "t", "A", [1, 0], [1, 0], 0, 0),
        row("same", "t", "A", [1, 0], [1, 0], 0, 1),
        row("other", "t", "A", [1, 0], [0, 1], 0, 2),
    ]
    result = reconstruct(rows, mode=NOTEBOOK_RECONSTRUCTION)
    assert result["attributes"]["PARTY"]["overall_representativeness"]["A"] == pytest.approx(2 / 3)


def test_manifest_uses_explicit_file_selected_and_expanded_counts(tmp_path):
    import numpy as np

    topic = tmp_path / "topic_mapping.npy"
    np.save(topic, {"q": {"cg": ["a", "b"]}})
    source = tmp_path / "combined.csv"
    source.write_text("x\n1\n2\n", encoding="utf-8")
    records = [row("q", "a", "A", [1, 0], [1, 0], 0, 0), row("q", "b", "A", [1, 0], [1, 0], 0, 0)]
    result = reconstruct(records, mode=PAPER_FORMULA)
    manifest = build_manifest(
        combined=[
            {
                "path": str(source),
                "row_count": 2,
                "columns": ["x"],
                "sha256": "x",
                "survey_wave": 26,
            }
        ],
        topic_mapping_path=topic,
        result=result,
        records=records,
    )
    assert manifest["total_file_rows"] == 2
    assert manifest["selected_model_source_rows"] == 1
    assert manifest["total_topic_expanded_records"] == 2
