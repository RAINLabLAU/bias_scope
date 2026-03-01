import pytest

from bias_scope.generated_text_based.fgb import FGB


def test_fgb_equation():
    """
    FGB = mean_{t,k} |s_a - s_b|
    """
    metric = FGB()

    group_a_completions = [["a1", "a2"], ["a3", "a4"]]
    group_b_completions = [["b1", "b2"], ["b3", "b4"]]

    # gaps: [0.4, 0.1, 0.0, 0.4] => mean = 0.225
    group_a_scores = [[0.6, 0.1], [0.0, 0.8]]
    group_b_scores = [[0.2, 0.2], [0.0, 0.4]]

    score = metric.evaluate(
        group_a_completions=group_a_completions,
        group_b_completions=group_b_completions,
        group_a_scores=group_a_scores,
        group_b_scores=group_b_scores,
    )
    assert score == pytest.approx(0.225)


def test_fgb_raises_on_unpaired_completions_shape():
    metric = FGB()

    group_a_completions = [["a1", "a2"], ["a3", "a4"]]
    group_b_completions = [["b1", "b2"], ["b3"]]
    group_a_scores = [[0.0, 0.1], [0.1, 0.2]]
    group_b_scores = [[0.0, 0.1], [0.1]]

    with pytest.raises(ValueError):
        metric.evaluate(
            group_a_completions=group_a_completions,
            group_b_completions=group_b_completions,
            group_a_scores=group_a_scores,
            group_b_scores=group_b_scores,
        )


def test_fgb_raises_on_score_shape_mismatch():
    metric = FGB()

    with pytest.raises(ValueError):
        metric.evaluate(
            group_a_completions=[["a1", "a2"]],
            group_b_completions=[["b1", "b2"]],
            group_a_scores=[[0.1]],
            group_b_scores=[[0.2, 0.3]],
        )


def test_fgb_raises_on_out_of_range_scores():
    metric = FGB()

    with pytest.raises(ValueError):
        metric.evaluate(
            group_a_completions=[["a"]],
            group_b_completions=[["b"]],
            group_a_scores=[[1.1]],
            group_b_scores=[[0.0]],
        )


def test_fgb_details():
    metric = FGB()

    result = metric.evaluate(
        group_a_completions=[["a1", "a2"]],
        group_b_completions=[["b1", "b2"]],
        group_a_scores=[[0.5, -0.2]],
        group_b_scores=[[0.3, -0.2]],
        return_details=True,
    )

    assert result["fgb_score"] == pytest.approx(0.1)
    assert result["signed_mean_difference"] == pytest.approx(0.1)
    assert result["num_templates"] == 1.0
    assert result["k"] == 2.0
    assert result["num_pairs"] == 2.0
    assert result["mean_group_a_score"] == pytest.approx(0.15)
    assert result["mean_group_b_score"] == pytest.approx(0.05)
    assert result["max_pair_gap"] == pytest.approx(0.2)
    assert result["min_pair_gap"] == pytest.approx(0.0)


def test_fgb_category_property():
    metric = FGB()
    assert metric.category == "generated_text"
