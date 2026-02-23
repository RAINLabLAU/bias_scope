import pytest

from bias_scope.generated_text_based.emt import EMT


def test_emt_equation():
    """
    EMT = mean_t max_k toxicity(t, k)
    """
    metric = EMT()
    completions = [
        ["a", "b", "c"],
        ["d", "e", "f"],
    ]
    toxicity_scores = [
        [0.1, 0.8, 0.2],  # max=0.8
        [0.3, 0.4, 0.9],  # max=0.9
    ]  # EMT=(0.8+0.9)/2=0.85

    score = metric.evaluate(completions, toxicity_scores)
    assert score == pytest.approx(0.85)


def test_emt_raises_on_shape_mismatch():
    metric = EMT()
    completions = [
        ["a", "b"],
        ["c", "d"],
    ]
    toxicity_scores = [
        [0.1, 0.2],
        [0.3],  # wrong K for second template
    ]

    with pytest.raises(ValueError):
        metric.evaluate(completions, toxicity_scores)


def test_emt_raises_on_out_of_range_score():
    metric = EMT()
    completions = [["a"]]
    toxicity_scores = [[1.2]]

    with pytest.raises(ValueError):
        metric.evaluate(completions, toxicity_scores)


def test_emt_details():
    metric = EMT()
    completions = [["a", "b"], ["c", "d"]]
    toxicity_scores = [[0.2, 0.7], [0.1, 0.4]]

    result = metric.evaluate(completions, toxicity_scores, return_details=True)

    assert result["emt_score"] == pytest.approx(0.55)
    assert result["num_templates"] == 2.0
    assert result["k"] == 2.0
    assert result["num_candidates"] == 4.0
    assert result["mean_toxicity"] == pytest.approx(0.35)
    assert result["max_toxicity"] == pytest.approx(0.7)
    assert result["min_toxicity"] == pytest.approx(0.1)


def test_emt_category_property():
    metric = EMT()
    assert metric.category == "generated_text"
