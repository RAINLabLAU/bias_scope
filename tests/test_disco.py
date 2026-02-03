import pytest

from src.bias_scope.probability_based.disco import DisCoMetric


def test_disco_score_symmetric_difference():
    """
    Unit test for the DisCo scoring function (no model required).

    This test validates the mathematical definition of the DisCo score,
    which is based on the symmetric difference between two top-k sets.

    Example:
        T_A = {engineer, doctor, manager}
        T_B = {doctor, teacher, nurse}

        overlap = {doctor}
        symmetric difference = {engineer, manager, teacher, nurse}
        => DisCo score = 4

    This test ensures that:
        - the score is computed correctly
        - the overlap is identified correctly
    """
    a = ["engineer", "doctor", "manager"]
    b = ["doctor", "teacher", "nurse"]

    score, overlap = DisCoMetric.disco_score(a, b)

    assert score == 4
    assert overlap == ["doctor"]


def test_prompt_validation_missing_mask():
    """
    Validation test: prompt without a mask token should raise an error.

    DisCo assumes a controlled setting with exactly one masked position.
    Prompts without a mask token violate this assumption and must be rejected.
    """
    disco = DisCoMetric("bert-base-uncased")

    with pytest.raises(ValueError):
        disco.top_k_predictions("No mask token here.", k=3)


def test_prompt_validation_multiple_masks():
    """
    Validation test: prompt with more than one mask token should raise an error.

    Multiple masks introduce ambiguity and break the controlled comparison
    required by the DisCo metric, so such prompts must be rejected.
    """
    disco = DisCoMetric("bert-base-uncased")
    mask = disco.mask_token

    with pytest.raises(ValueError):
        disco.top_k_predictions(f"This {mask} has two {mask}.", k=3)


@pytest.mark.slow
def test_integration_returns_k_tokens():
    """
    Integration test for top-k prediction extraction.

    This test runs the full DisCo pipeline step that:
        - tokenizes a prompt
        - runs the masked language model
        - extracts the top-k predicted tokens at the mask

    The test checks:
        - exactly k tokens are returned
        - all returned elements are strings

    Marked as 'slow' because it requires loading a transformer model.
    It can be skipped locally using:
        pytest -m "not slow"
    """
    disco = DisCoMetric("bert-base-uncased")
    mask = disco.mask_token

    tokens = disco.top_k_predictions(f"The doctor is a {mask}.", k=3)

    assert len(tokens) == 3
    assert all(isinstance(token, str) for token in tokens)
