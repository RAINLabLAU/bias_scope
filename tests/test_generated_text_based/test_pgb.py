import pytest

from bias_scope.generated_text_based.pgb import PGB


def test_pgb_equation():
    """
    PGB = mean_{t,k} max(0, s_a - s_b)
    """
    metric = PGB()

    group_a_completions = [["a1", "a2"], ["a3", "a4"]]
    group_b_completions = [["b1", "b2"], ["b3", "b4"]]

    # deltas: [0.4, -0.1, 0.0, 0.4] => positive part: [0.4, 0.0, 0.0, 0.4] => mean=0.2
    group_a_scores = [[0.6, 0.1], [0.0, 0.8]]
    group_b_scores = [[0.2, 0.2], [0.0, 0.4]]

    score = metric.evaluate(
        group_a_completions=group_a_completions,
        group_b_completions=group_b_completions,
        group_a_scores=group_a_scores,
        group_b_scores=group_b_scores,
    )
    assert score == pytest.approx(0.2)


def test_pgb_raises_on_unpaired_completions_shape():
    metric = PGB()

    with pytest.raises(ValueError):
        metric.evaluate(
            group_a_completions=[["a1", "a2"], ["a3", "a4"]],
            group_b_completions=[["b1", "b2"], ["b3"]],
            group_a_scores=[[0.1, 0.2], [0.2, 0.3]],
            group_b_scores=[[0.1, 0.2], [0.3]],
        )


def test_pgb_raises_on_score_shape_mismatch():
    metric = PGB()

    with pytest.raises(ValueError):
        metric.evaluate(
            group_a_completions=[["a1", "a2"]],
            group_b_completions=[["b1", "b2"]],
            group_a_scores=[[0.1]],
            group_b_scores=[[0.1, 0.2]],
        )


def test_pgb_raises_on_out_of_range_scores():
    metric = PGB()

    with pytest.raises(ValueError):
        metric.evaluate(
            group_a_completions=[["a"]],
            group_b_completions=[["b"]],
            group_a_scores=[[1.2]],
            group_b_scores=[[0.0]],
        )


def test_pgb_details():
    metric = PGB()

    result = metric.evaluate(
        group_a_completions=[["a1", "a2"]],
        group_b_completions=[["b1", "b2"]],
        group_a_scores=[[0.5, -0.2]],
        group_b_scores=[[0.3, -0.2]],
        return_details=True,
    )

    # deltas: [0.2, 0.0] => pgb = (0.2 + 0.0)/2 = 0.1
    assert result["pgb_score"] == pytest.approx(0.1)
    assert result["num_templates"] == 1.0
    assert result["k"] == 2.0
    assert result["num_pairs"] == 2.0
    assert result["mean_group_a_score"] == pytest.approx(0.15)
    assert result["mean_group_b_score"] == pytest.approx(0.05)
    assert result["signed_mean_difference"] == pytest.approx(0.1)
    assert result["pct_pairs_positive_direction"] == pytest.approx(0.5)
    assert result["pct_pairs_non_positive"] == pytest.approx(0.5)


def test_pgb_category_property():
    metric = PGB()
    assert metric.category == "generated_text"
