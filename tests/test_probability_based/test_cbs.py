import pytest

from bias_scope.probability_based.cbs import CBS


def test_template_validation_missing_mask():
    """
    Validation test: template without a mask token should raise an error.

    CBS requires exactly one masked position for the TARGET category slot.
    """
    cbs = CBS("bert-base-uncased")

    with pytest.raises(ValueError):
        cbs.evaluate(
            templates=["People from X are {attr}."],
            target_words=["American", "Chinese"],
            attribute_words=["smart"],
        )


def test_template_validation_missing_placeholder():
    """
    Validation test: template without {attr} placeholder should raise an error.

    CBS must be able to insert attribute words into templates.
    """
    cbs = CBS("bert-base-uncased")
    mask = cbs.mask_token

    with pytest.raises(ValueError):
        cbs.evaluate(
            templates=[f"People from {mask} are smart."],
            target_words=["American", "Chinese"],
            attribute_words=["smart"],
        )


def test_target_word_multi_token_rejected_by_default():
    """
    Validation test: multi-token target words should be rejected by default.

    CBS indexes probabilities by token id; in strict mode we require each target
    word to map to exactly one token under the tokenizer.
    """
    cbs = CBS("bert-base-uncased")
    mask = cbs.mask_token

    with pytest.raises(ValueError):
        cbs.evaluate(
            templates=[f"People from {mask} are {{attr}}."],
            target_words=["New York"],  # very likely multi-token
            attribute_words=["smart"],
            # allow_multi_token_targets defaults to False
        )


@pytest.mark.slow
def test_multi_token_targets_allowed_when_enabled():
    """
    Integration test (slow): multi-token targets should work when enabled.

    This confirms the Part 4 feature:
        allow_multi_token_targets=True
    """
    cbs = CBS("bert-base-uncased")
    mask = cbs.mask_token

    score = cbs.evaluate(
        templates=[f"People from {mask} are {{attr}}."],
        target_words=["American", "New York", "Chinese"],  # includes multi-token
        attribute_words=["smart"],
        allow_multi_token_targets=True,
    )

    assert score >= 0.0


@pytest.mark.slow
def test_cbs_integration_runs_and_details_shape():
    """
    Integration test (slow): run CBS end-to-end and validate output format.

    After Part 2, details entries should include:
        - variance (float)
        - top_target (str)
    """
    cbs = CBS("bert-base-uncased")
    mask = cbs.mask_token

    result = cbs.evaluate(
        templates=[f"People from {mask} are {{attr}}."],
        target_words=["American", "Chinese", "Arab"],
        attribute_words=["smart", "lazy"],
        return_details=True,
    )

    assert "cbs" in result
    assert "details" in result
    assert result["cbs"] >= 0.0

    # Check one details entry has the expected structure
    first_key = next(iter(result["details"]))
    first_val = result["details"][first_key]

    assert isinstance(first_val, dict)
    assert "variance" in first_val and isinstance(first_val["variance"], float)
    assert "top_target" in first_val and isinstance(first_val["top_target"], str)
