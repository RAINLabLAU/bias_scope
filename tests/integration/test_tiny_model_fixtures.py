"""The tiny-model fixtures load and drive a real metric end to end.

An unexercised fixture is an unverified one, so this file uses each of
``tiny_encoder`` and ``tiny_causal`` for the thing it exists for: scoring text
through a real (if absurdly small) model, on CPU, in well under a second.

These skip rather than fail when transformers is absent or the download cannot
happen; see the message in ``tests/conftest.py``.
"""

import math

import pytest
from tests.conftest import TINY_ENCODER_ID

from bias_scope.probability_based import CrowSPairs


def test_tiny_encoder_scores_a_masked_token(tiny_encoder):
    """The tiny encoder gives a usable log-probability for a masked position."""
    torch = pytest.importorskip("torch")
    tokenizer, model = tiny_encoder

    text = f"The doctor finished {tokenizer.mask_token} shift."
    inputs = tokenizer(text, return_tensors="pt")
    mask_index = (inputs["input_ids"][0] == tokenizer.mask_token_id).nonzero().item()

    with torch.no_grad():
        logits = model(**inputs).logits

    log_probs = torch.log_softmax(logits[0, mask_index], dim=-1)
    assert log_probs.shape[0] == len(tokenizer)
    assert math.isfinite(float(log_probs.max()))
    assert float(log_probs.max()) <= 0.0


def test_tiny_causal_scores_a_continuation(tiny_causal):
    """tiny-gpt2 gives a finite loss for a short sequence."""
    torch = pytest.importorskip("torch")
    tokenizer, model = tiny_causal

    inputs = tokenizer("The nurse said that", return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])

    assert math.isfinite(float(outputs.loss))


def test_crows_pairs_runs_against_the_tiny_encoder(tiny_encoder):
    """A probability-family metric end to end on a real model.

    Two hand-written pairs, so the assertion is on the shape of the result
    rather than on a value a randomly-initialised tiny BERT has no business
    getting right.
    """
    torch = pytest.importorskip("torch")
    tokenizer, model = tiny_encoder

    def predict_masked_token(tokens, index):
        """P(tokens[index] | rest), with that position masked.

        CrowS-Pairs validates that this callback returns a probability in
        (0, 1], not a log-probability, so the softmax output is used directly.
        """
        masked = list(tokens)
        target = masked[index]
        masked[index] = tokenizer.mask_token

        encoded = tokenizer(" ".join(masked), return_tensors="pt")
        positions = (encoded["input_ids"][0] == tokenizer.mask_token_id).nonzero()
        if positions.numel() == 0:
            return 1e-12

        with torch.no_grad():
            logits = model(**encoded).logits
        probs = torch.softmax(logits[0, positions[0].item()], dim=-1)

        target_ids = tokenizer.encode(target, add_special_tokens=False)
        if not target_ids:
            return 1e-12
        # Clamp away from exactly 0, which the metric rejects as invalid.
        return max(float(probs[target_ids[0]]), 1e-12)

    pairs = [
        (["The", "man", "was", "good", "at", "math"],
         ["The", "woman", "was", "good", "at", "math"]),
        (["The", "old", "man", "was", "slow"],
         ["The", "young", "man", "was", "slow"]),
    ]

    # `mode="whitespace"` is explicit because the default is now the faithful
    # WordPiece path, which takes a scorer object rather than this callback.
    # The wordpiece path gets its own end-to-end test below.
    score = CrowSPairs(mode="whitespace").evaluate(
        sentence_pairs=pairs,
        predict_masked_token=predict_masked_token,
    )

    # CrowS-Pairs reports the percentage of pairs where the stereotyping
    # sentence is preferred; 50 is the neutral value.
    assert 0.0 <= score <= 100.0
    assert math.isfinite(score)


def test_crows_pairs_wordpiece_path_runs_against_the_tiny_encoder():
    """The faithful path, end to end on a real model.

    This is the one that matters: `mode="wordpiece"` is the default from 0.2.0
    and is what `MetricInfo.fidelity == "faithful"` refers to. It scores
    WordPiece pseudo-log-likelihoods, so it accepts sentence *strings* and
    handles targets that split into several subwords — which the whitespace
    path cannot.
    """
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from bias_scope.probability_based.scorers import WordPieceBertScorer

    try:
        scorer = WordPieceBertScorer(model_name=TINY_ENCODER_ID)
    except (OSError, ImportError) as error:  # offline
        pytest.skip(f"{TINY_ENCODER_ID} unavailable: {error}")

    pairs = [
        ("The man was good at mathematics.", "The woman was good at mathematics."),
        ("The old programmer was slow.", "The young programmer was slow."),
    ]
    score = CrowSPairs().evaluate(sentence_pairs=pairs, predict_masked_token=scorer)

    assert 0.0 <= score <= 100.0
    assert math.isfinite(score)
