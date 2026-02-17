import pytest

from bias_scope.generated_text_based.honest import HONEST


def test_honest_score_matches_equation():
    """
    HONEST = hurtful_count / (|T| * K)
    """
    metric = HONEST()

    completions = [
        ["doctor", "whore", "nurse"],
        ["teacher", "thief", "kind"],
    ]  # |T|=2, K=3, hurtful={whore, thief} => 2/6
    hurtlex = {"whore", "thief"}

    score = metric.evaluate(completions, hurtlex)
    assert score == pytest.approx(2.0 / 6.0)


def test_honest_multiword_candidate_detects_hurt_term():
    metric = HONEST()

    completions = [
        ["is a kind person", "is an idiot", "is calm"],
    ]
    hurtlex = {"idiot"}

    score = metric.evaluate(completions, hurtlex)
    assert score == pytest.approx(1.0 / 3.0)


def test_honest_raises_on_unequal_k():
    metric = HONEST()
    completions = [
        ["a", "b", "c"],
        ["x", "y"],
    ]

    with pytest.raises(ValueError):
        metric.evaluate(completions, {"x"})


def test_honest_raises_on_empty_hurtlex():
    metric = HONEST()

    with pytest.raises(ValueError):
        metric.evaluate([["a"]], set())


def test_honest_details_with_categories():
    metric = HONEST()

    completions = [
        ["good", "whore", "thief"],
        ["smart", "idiot", "kind"],
    ]  # total=6, hurtful=3
    hurtlex = {
        "whore": "prostitution",
        "thief": "crime",
        "idiot": "derogatory",
    }

    result = metric.evaluate(completions, hurtlex, return_details=True)

    assert result["honest_score"] == pytest.approx(0.5)
    assert result["hurtful_percentage"] == pytest.approx(50.0)
    assert result["num_templates"] == 2.0
    assert result["k"] == 3.0
    assert result["num_candidates"] == 6.0
    assert result["num_hurtful_candidates"] == 3.0
    assert result["category_rate::prostitution"] == pytest.approx(1.0 / 6.0)
    assert result["category_rate::crime"] == pytest.approx(1.0 / 6.0)
    assert result["category_rate::derogatory"] == pytest.approx(1.0 / 6.0)


def test_honest_category_property():
    metric = HONEST()
    assert metric.category == "generated_text"
