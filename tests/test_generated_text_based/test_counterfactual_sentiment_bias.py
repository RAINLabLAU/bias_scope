import pytest

from bias_scope.generated_text_based.counterfactual_sentiment_bias import (
    CounterfactualSentimentBias,
)


def test_counterfactual_sentiment_bias_equation():
    """
    CSB = mean_{t,k}(s_a - s_b)
    """
    metric = CounterfactualSentimentBias()

    group_a_completions = [["a1", "a2"], ["a3", "a4"]]
    group_b_completions = [["b1", "b2"], ["b3", "b4"]]

    # deltas: [0.4, -0.1, 0.0, 0.4] => mean = 0.175
    group_a_scores = [[0.6, 0.1], [0.0, 0.8]]
    group_b_scores = [[0.2, 0.2], [0.0, 0.4]]

    score = metric.evaluate(
        group_a_completions=group_a_completions,
        group_b_completions=group_b_completions,
        group_a_sentiment_scores=group_a_scores,
        group_b_sentiment_scores=group_b_scores,
    )
    assert score == pytest.approx(0.175)


def test_counterfactual_sentiment_bias_raises_on_unpaired_completions_shape():
    metric = CounterfactualSentimentBias()

    group_a_completions = [["a1", "a2"], ["a3", "a4"]]
    group_b_completions = [["b1", "b2"], ["b3"]]  # K mismatch in template 2
    group_a_scores = [[0.0, 0.1], [0.1, 0.2]]
    group_b_scores = [[0.0, 0.1], [0.1]]

    with pytest.raises(ValueError):
        metric.evaluate(
            group_a_completions=group_a_completions,
            group_b_completions=group_b_completions,
            group_a_sentiment_scores=group_a_scores,
            group_b_sentiment_scores=group_b_scores,
        )


def test_counterfactual_sentiment_bias_raises_on_score_shape_mismatch():
    metric = CounterfactualSentimentBias()

    group_a_completions = [["a1", "a2"]]
    group_b_completions = [["b1", "b2"]]
    group_a_scores = [[0.1]]  # wrong K
    group_b_scores = [[0.1, 0.2]]

    with pytest.raises(ValueError):
        metric.evaluate(
            group_a_completions=group_a_completions,
            group_b_completions=group_b_completions,
            group_a_sentiment_scores=group_a_scores,
            group_b_sentiment_scores=group_b_scores,
        )


def test_counterfactual_sentiment_bias_raises_on_out_of_range_scores():
    metric = CounterfactualSentimentBias()

    with pytest.raises(ValueError):
        metric.evaluate(
            group_a_completions=[["a"]],
            group_b_completions=[["b"]],
            group_a_sentiment_scores=[[1.5]],
            group_b_sentiment_scores=[[0.0]],
        )


def test_counterfactual_sentiment_bias_details():
    metric = CounterfactualSentimentBias()

    group_a_completions = [["a1", "a2"]]
    group_b_completions = [["b1", "b2"]]
    group_a_scores = [[0.5, -0.2]]
    group_b_scores = [[0.3, -0.2]]
    # deltas: [0.2, 0.0]

    result = metric.evaluate(
        group_a_completions=group_a_completions,
        group_b_completions=group_b_completions,
        group_a_sentiment_scores=group_a_scores,
        group_b_sentiment_scores=group_b_scores,
        return_details=True,
    )

    assert result["csb_score"] == pytest.approx(0.1)
    assert result["absolute_csb_score"] == pytest.approx(0.1)
    assert result["num_templates"] == 1.0
    assert result["k"] == 2.0
    assert result["num_pairs"] == 2.0
    assert result["mean_group_a_sentiment"] == pytest.approx(0.15)
    assert result["mean_group_b_sentiment"] == pytest.approx(0.05)
    assert result["pct_pairs_group_a_higher"] == pytest.approx(0.5)
    assert result["pct_pairs_group_b_higher"] == pytest.approx(0.0)
    assert result["pct_pairs_equal"] == pytest.approx(0.5)


def test_counterfactual_sentiment_bias_category_property():
    metric = CounterfactualSentimentBias()
    assert metric.category == "generated_text"
