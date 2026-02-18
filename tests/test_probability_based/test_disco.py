import pytest

from bias_scope.probability_based.disco import DisCoMetric


def test_disco_score_symmetric_difference():
    """
    Unit test for the DisCo scoring function (no model required).

    Validates the symmetric difference definition:
        T_A = {engineer, doctor, manager}
        T_B = {doctor, teacher, nurse}
        overlap = {doctor}
        sym diff = {engineer, manager, teacher, nurse}
        score = 4
    """
    a = ["engineer", "doctor", "manager"]
    b = ["doctor", "teacher", "nurse"]

    score, overlap = DisCoMetric._disco_score(a, b)

    assert score == 4
    assert overlap == ["doctor"]


def test_template_validation_missing_mask():
    """
    Template without a mask token should raise an error.

    DisCo assumes exactly one masked position.
    """
    disco = DisCoMetric("bert-base-uncased")

    with pytest.raises(ValueError):
        # Has {attr} but no [MASK]
        disco.evaluate("The {attr} works as a doctor.", "man", "woman", k=3)


def test_template_validation_multiple_masks():
    """
    Template with more than one mask token should raise an error.
    """
    disco = DisCoMetric("bert-base-uncased")
    mask = disco.mask_token

    with pytest.raises(ValueError):
        disco.evaluate(
            f"The {{attr}} is a {mask} and also a {mask}.", "man", "woman", k=3
        )


@pytest.mark.slow
def test_integration_returns_k_tokens():
    """
    Integration test: evaluate returns a result with exactly k tokens per side.
    Marked slow because it loads a transformer model.
    """
    disco = DisCoMetric("bert-base-uncased")
    mask = disco.mask_token

    result = disco.evaluate(f"The {{attr}} works as a {mask}.", "man", "woman", k=3)

    assert len(result.topk_a) == 3
    assert len(result.topk_b) == 3
    assert all(isinstance(tok, str) for tok in result.topk_a)
    assert all(isinstance(tok, str) for tok in result.topk_b)
