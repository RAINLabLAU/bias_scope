"""Differential test: SEAT's effect size must equal the naive WEAT oracle.

SEAT (May et al. 2019, App. A) computes the effect size "identically" to
Caliskan et al. — same ``s(w, A, B)``, same standardised mean difference with
an unbiased (ddof=1) standard deviation. So the WEAT oracle in
``tests/oracles/weat_oracle.py`` is the reference for SEAT's magnitude too; only
the permutation p-value convention differs (``>=`` vs ``>``), which is covered
in ``tests/test_embeddings/test_seat.py``.
"""

import numpy as np
from tests.oracles.weat_oracle import effect_size

from bias_scope.embeddings_based import SEAT

TOLERANCE = 1e-8
N_RANDOM_INPUTS = 200


def test_seat_effect_size_agrees_with_the_weat_oracle():
    rng = np.random.default_rng(20260909)
    seat = SEAT()

    worst = 0.0
    for _ in range(N_RANDOM_INPUTS):
        n = int(rng.integers(3, 9))
        dim = int(rng.integers(4, 17))
        X, Y, A, B = (rng.standard_normal((n, dim)) for _ in range(4))

        library = seat.evaluate((X, Y), (A, B))
        oracle = effect_size(X.tolist(), Y.tolist(), A.tolist(), B.tolist())
        worst = max(worst, abs(library - oracle))

    assert worst < TOLERANCE, f"largest disagreement {worst:.3e} exceeds {TOLERANCE:.0e}"
