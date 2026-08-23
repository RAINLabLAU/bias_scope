import pytest

from bias_scope.generated_text_based.counterfactual_sentiment_bias import (
    CounterfactualSentimentBias,
)


def test_counterfactual_sentiment_bias_equation():
    """CSB = mean over templates of W1(P_A, P_B)  (Huang et al. eq. 1-2).

    Changed in 0.2.0. v0.1.1 computed `mean_{t,k}(s_a - s_b)` = 0.175 on this
    input, the signed mean of paired differences. That is not the paper's
    statistic; it is retained as `signed_mean_difference`. See
    docs/fidelity/huang_metrics.md.

    Template 1: A = [0.6, 0.1] -> sorted [0.1, 0.6]
                B = [0.2, 0.2] -> sorted [0.2, 0.2]
                W1 = mean(|0.1-0.2|, |0.6-0.2|) = mean(0.1, 0.4) = 0.25
    Template 2: A = [0.0, 0.8] -> sorted [0.0, 0.8]
                B = [0.0, 0.4] -> sorted [0.0, 0.4]
                W1 = mean(0.0, 0.4) = 0.2
    CSB = (0.25 + 0.2) / 2 = 0.225
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
    assert score == pytest.approx(0.225)

    details = metric.evaluate(
        group_a_completions=group_a_completions,
        group_b_completions=group_b_completions,
        group_a_sentiment_scores=group_a_scores,
        group_b_sentiment_scores=group_b_scores,
        return_details=True,
    )
    # The v0.1.1 statistic is still available, and still 0.175.
    assert details["signed_mean_difference"] == pytest.approx(0.175)
    assert details["per_item"] == pytest.approx([0.25, 0.2])


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
