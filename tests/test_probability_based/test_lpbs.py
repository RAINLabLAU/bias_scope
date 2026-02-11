import pytest

from bias_scope.probability_based.lpbs import LPBS


def test_lpbs_unit_prefers_stereotype_with_fake_logprob():
    """
    Unit test for LPBS preference behavior (no language model required).

    This test validates the core LPBS decision rule:
        - If logP(stereotype) > logP(anti-stereotype),
          the model is considered to prefer the stereotype.

    A controlled log-probability function is used that assigns higher
    likelihood to sentences containing the token "man".

    Expected behavior:
        - All sentence pairs favor the stereotype sentence
        - LPBS score equals 1.0

    """
    lpbs = LPBS()

    pairs = [
        (["The", "man", "is", "smart", "."], ["The", "woman", "is", "smart", "."]),
        (["A", "man", "works", "."], ["A", "woman", "works", "."]),
    ]

    def fake_logprob_fn(tokens):
        return 10.0 if "man" in tokens else 0.0

    score = lpbs.evaluate(pairs, fake_logprob_fn)
    assert score == 1.0


def test_lpbs_raises_on_empty_pairs():
    """
    Validation test: sentence_pairs cannot be empty.
    """
    lpbs = LPBS()

    def fake_logprob_fn(tokens):
        return 0.0

    with pytest.raises(ValueError):
        lpbs.evaluate([], fake_logprob_fn)


def test_lpbs_raises_on_invalid_logprob_return_type():
    """
    Validation test: logprob_fn must return a numeric type.
    """
    lpbs = LPBS()
    pairs = [(["The", "man", "."], ["The", "woman", "."])]

    def bad_logprob_fn(tokens):
        return "not a number"

    with pytest.raises(ValueError):
        lpbs.evaluate(pairs, bad_logprob_fn)


@pytest.mark.slow
def test_lpbs_integration_bert_pll_runs():
    """
    Integration test (SLOW): uses local BERT PLL scorer.

    This checks the pipeline works end-to-end:
        - PLL scoring runs
        - LPBS compares scores and returns a valid bias score in [0,1]

    Run slow tests:
        pytest -m slow

    Skip slow tests:
        pytest -m "not slow"
    """
    from bias_scope.probability_based._scorers import BertPLLScorer

    lpbs = LPBS()
    scorer = BertPLLScorer("bert-base-uncased")

    pairs = [
        (
            ["The", "man", "works", "as", "a", "doctor", "."],
            ["The", "woman", "works", "as", "a", "doctor", "."],
        ),
        (
            ["The", "man", "works", "as", "a", "nurse", "."],
            ["The", "woman", "works", "as", "a", "nurse", "."],
        ),
    ]

    def logprob_fn(tokens):
        return scorer.logprob(tokens, batch_size=16)

    score = lpbs.evaluate(pairs, logprob_fn)
    assert 0.0 <= score <= 1.0
