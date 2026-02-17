import pytest

from bias_scope.generated_text_based.gender_polarity import GenderPolarity


def test_gender_polarity_equation():
    """
    GP(c) = (m - f) / (m + f), GP = mean_c GP(c)
    """
    metric = GenderPolarity()

    completions = [
        ["he is strong", "she is strong"],   # +1, -1
        ["he and she work"],                 # 0
    ]  # K mismatch would fail validation, so keep equal K:
    completions = [
        ["he is strong", "she is strong"],
        ["he and she work", "he leads"],
    ]  # per-completion GP: +1, -1, 0, +1 => mean=0.25

    result = metric.evaluate(
        completions=completions,
        masculine_terms={"he"},
        feminine_terms={"she"},
    )
    assert result == pytest.approx(0.25)


def test_gender_polarity_neutral_policy_zero():
    metric = GenderPolarity()
    completions = [["unknown token", "he speaks"]]  # GP: 0 and +1 => 0.5

    score = metric.evaluate(
        completions=completions,
        masculine_terms={"he"},
        feminine_terms={"she"},
        neutral_policy="zero",
    )
    assert score == pytest.approx(0.5)


def test_gender_polarity_neutral_policy_skip():
    metric = GenderPolarity()
    completions = [["unknown token", "he speaks"]]  # skip first => +1

    score = metric.evaluate(
        completions=completions,
        masculine_terms={"he"},
        feminine_terms={"she"},
        neutral_policy="skip",
    )
    assert score == pytest.approx(1.0)


def test_gender_polarity_neutral_policy_error():
    metric = GenderPolarity()
    completions = [["unknown token"]]

    with pytest.raises(ValueError):
        metric.evaluate(
            completions=completions,
            masculine_terms={"he"},
            feminine_terms={"she"},
            neutral_policy="error",
        )


def test_gender_polarity_overlap_terms_raise():
    metric = GenderPolarity()

    with pytest.raises(ValueError):
        metric.evaluate(
            completions=[["he she"]],
            masculine_terms={"he"},
            feminine_terms={"he", "she"},
        )


def test_gender_polarity_details():
    metric = GenderPolarity()
    completions = [["he is good", "she is good"]]

    result = metric.evaluate(
        completions=completions,
        masculine_terms={"he"},
        feminine_terms={"she"},
        return_details=True,
    )

    assert result["gender_polarity_score"] == pytest.approx(0.0)
    assert result["num_completions"] == 2.0
    assert result["num_scored_completions"] == 2.0
    assert result["completion_coverage_rate"] == pytest.approx(1.0)
    assert result["pct_masculine_leaning"] == pytest.approx(0.5)
    assert result["pct_feminine_leaning"] == pytest.approx(0.5)
    assert result["pct_balanced"] == pytest.approx(0.0)


def test_gender_polarity_category_property():
    metric = GenderPolarity()
    assert metric.category == "generated_text"
