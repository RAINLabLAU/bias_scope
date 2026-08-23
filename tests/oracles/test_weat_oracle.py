"""Differential test: the library's WEAT must agree with the naive oracle.

Seed example for the oracle-agreement criterion (PLAN.md Section 1, criterion 2):
200 random inputs, agreement to 1e-8.
"""

import math

import numpy as np
import pytest
from tests.oracles.weat_oracle import effect_size

from bias_scope.embeddings_based import WEAT

TOLERANCE = 1e-8
N_RANDOM_INPUTS = 200


def test_agrees_with_the_oracle_on_random_inputs():
    """Library and oracle agree to 1e-8 across 200 seeded random draws."""
    rng = np.random.default_rng(20260822)
    weat = WEAT()

    worst = 0.0
    for _ in range(N_RANDOM_INPUTS):
        n = int(rng.integers(3, 9))
        dim = int(rng.integers(4, 17))
        X, Y, A, B = (rng.standard_normal((n, dim)) for _ in range(4))

        library = weat.evaluate((X, Y), (A, B))
        oracle = effect_size(X.tolist(), Y.tolist(), A.tolist(), B.tolist())
        worst = max(worst, abs(library - oracle))

    assert worst < TOLERANCE, f"largest disagreement {worst:.3e} exceeds {TOLERANCE:.0e}"


def test_oracle_reproduces_a_hand_computed_case():
    """Orthogonal, perfectly separated sets give a hand-checkable answer.

    X = {e1}, Y = {e2}, A = {e1}, B = {e2}:
      s(e1) = cos(e1,e1) - cos(e1,e2) = 1 - 0 =  1
      s(e2) = cos(e2,e1) - cos(e2,e2) = 0 - 1 = -1
      pooled {1, -1}, mean 0
        sample variance (ddof=1) = (1 + 1) / (2 - 1) = 2, so std = sqrt(2)
        d = (1 - (-1)) / sqrt(2) = sqrt(2)
      population variance (ddof=0) = (1 + 1) / 2 = 1, so std = 1
        d = (1 - (-1)) / 1 = 2
    """
    e1, e2 = [1.0, 0.0], [0.0, 1.0]
    assert effect_size([e1], [e2], [e1], [e2], ddof=1) == pytest.approx(math.sqrt(2))
    assert effect_size([e1], [e2], [e1], [e2], ddof=0) == pytest.approx(2.0)


def test_library_uses_the_sample_standard_deviation():
    """Pins the library's ddof=1 convention on the hand-computed case.

    Not a fidelity claim: whether Caliskan eq. 3 means ddof=0 or ddof=1 is an
    open Phase 1 question (RL-008). This test exists so that changing the
    convention is a deliberate act with a visible diff.
    """
    X = np.array([[1.0, 0.0]])
    Y = np.array([[0.0, 1.0]])
    assert WEAT().evaluate((X, Y), (X, Y)) == pytest.approx(math.sqrt(2))
