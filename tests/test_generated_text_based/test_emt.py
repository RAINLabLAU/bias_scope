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
    assert result["bias_score"] == result["emt_score"]
    assert result["per_item"] == pytest.approx([0.7, 0.4])
    # std of [0.7, 0.4] with ddof=1 (pandas/reference convention)
    assert result["std"] == pytest.approx(0.21213203435596426)


def test_std_matches_reference_notebook_convention():
    """RL-022: Gehman et al. report EMT "with a mean and standard deviation".
    The reference notebook computes std_max = max_toxicities.std() (pandas
    default ddof=1). Verify against a hand-computed sample std."""
    metric = EMT()
    completions = [["a"], ["b"], ["c"], ["d"]]
    toxicity_scores = [[0.1], [0.3], [0.5], [0.9]]  # template_maxima == these

    result = metric.evaluate(completions, toxicity_scores, return_details=True)

    import statistics

    expected_std = statistics.stdev([0.1, 0.3, 0.5, 0.9])
    assert result["std"] == pytest.approx(expected_std)


def test_std_is_zero_for_a_single_template():
    """ddof=1 divides by zero for n=1; must not raise or return NaN."""
    metric = EMT()
    completions = [["a", "b"]]
    toxicity_scores = [[0.2, 0.7]]

    result = metric.evaluate(completions, toxicity_scores, return_details=True)
    assert result["std"] == 0.0


def test_run_exposes_std_in_details():
    metric = EMT()
    completions = [["a", "b"], ["c", "d"]]
    toxicity_scores = [[0.2, 0.7], [0.1, 0.4]]

    result = metric.run(completions, toxicity_scores, ci="none")
    assert result.details["std"] == pytest.approx(0.21213203435596426)


def test_run_no_longer_crashes():
    """Before the fix, run() always raised BiasScopeError: evaluate()'s
    dict had no key _split_result recognised (only 'emt_score', etc.)."""
    metric = EMT()
    completions = [["a", "b"], ["c", "d"]]
    toxicity_scores = [[0.2, 0.7], [0.1, 0.4]]

    result = metric.run(completions, toxicity_scores, ci="none")
    assert result.score == metric.evaluate(completions, toxicity_scores)
    assert result.n == 2


def test_run_bootstrap_produces_a_ci():
    metric = EMT()
    completions = [["a", "b"], ["c", "d"], ["e", "f"], ["g", "h"]]
    toxicity_scores = [[0.2, 0.7], [0.1, 0.4], [0.9, 0.1], [0.3, 0.3]]

    result = metric.run(completions, toxicity_scores)  # default bootstrap
    assert result.ci is not None
    assert result.ci_method == "bootstrap"
    lo, hi = result.ci
    assert lo <= result.score <= hi


def test_emt_category_property():
    metric = EMT()
    assert metric.category == "generated_text"
