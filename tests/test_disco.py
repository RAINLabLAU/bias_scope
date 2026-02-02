import pytest

from src.bias_scope.probability_based.disco import DisCoMetric


def test_disco_score_symmetric_difference():
    """
    Pure unit test (no model needed).
    Checks the math of symmetric difference.

    Example:
      A = {engineer, doctor, manager}
      B = {doctor, teacher, nurse}
      overlap = {doctor}
      symmetric diff = {engineer, manager, teacher, nurse} => size 4
    """
    a = ["engineer", "doctor", "manager"]
    b = ["doctor", "teacher", "nurse"]

    score, overlap = DisCoMetric.disco_score(a, b)
    assert score == 4
    assert overlap == ["doctor"]


def test_prompt_validation_missing_mask():
    """
    DisCo should enforce that prompt contains exactly one [MASK].
    """
    disco = DisCoMetric("bert-base-uncased")
    with pytest.raises(ValueError):
        disco.top_k_predictions("No mask token here.", k=3)


def test_prompt_validation_multiple_masks():
    """
    DisCo should reject prompts with multiple masks.
    """
    disco = DisCoMetric("bert-base-uncased")
    mask = disco.mask_token
    with pytest.raises(ValueError):
        disco.top_k_predictions(f"This {mask} has two {mask}.", k=3)


@pytest.mark.slow
def test_integration_returns_k_tokens():
    """
    Integration test: actually runs the model and verifies top-k length.
    Marked slow because it downloads/loads a transformer model.

    You can skip it locally with:
      pytest -m "not slow"
    """
    disco = DisCoMetric("bert-base-uncased")
    mask = disco.mask_token

    tokens = disco.top_k_predictions(f"The doctor is a {mask}.", k=3)
    assert len(tokens) == 3
    assert all(isinstance(t, str) for t in tokens)

