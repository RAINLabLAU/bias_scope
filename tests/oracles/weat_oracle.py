"""Naive reference implementation of WEAT, for differential testing.

Section 1, criterion 2: every metric gets a deliberately naive, loop-based
implementation of the paper's formula — 10-30 lines, no numpy tricks — that the
library version must agree with to 1e-8 on random inputs. The point is that the
two implementations are written differently enough that a shared bug is
unlikely.

Caliskan, Bryson & Narayanan 2017 (Science 356:6334), eq. 1-3:

    s(w, A, B)      = mean_{a in A} cos(w, a) - mean_{b in B} cos(w, b)
    effect size d   = (mean_{x in X} s(x, A, B) - mean_{y in Y} s(y, A, B))
                      / stddev_{w in X u Y} s(w, A, B)

OPEN QUESTION (Phase 1, REVIEW_LATER RL-008): the paper writes "std-dev" without
saying whether it is the population (ddof=0) or sample (ddof=1) standard
deviation, and the two differ by sqrt(n/(n-1)) — 7% at the n=8 of Caliskan's own
word sets. `src/bias_scope/embeddings_based/weat.py:244` uses ddof=1. This
oracle takes `ddof` explicitly and defaults to 1 so it agrees with the library
today; it is NOT evidence that ddof=1 is what Caliskan meant. PLAN.md Section
4.0 forbids settling that from memory — the WEAT audit must read eq. 3 and the
reference implementation and then fix whichever of the two is wrong.

This is the seed example for ``tests/oracles/``; Phases 1-3 add one per metric.
"""

import math
from typing import List, Sequence


def _cosine(u: Sequence[float], v: Sequence[float]) -> float:
    """Cosine similarity, written out with explicit loops."""
    dot = 0.0
    norm_u = 0.0
    norm_v = 0.0
    for ui, vi in zip(u, v):
        dot += ui * vi
        norm_u += ui * ui
        norm_v += vi * vi
    if norm_u == 0.0 or norm_v == 0.0:
        return 0.0
    return dot / (math.sqrt(norm_u) * math.sqrt(norm_v))


def association(w: Sequence[float], A: Sequence[Sequence[float]],
                B: Sequence[Sequence[float]]) -> float:
    """s(w, A, B): mean cosine to A minus mean cosine to B."""
    mean_a = sum(_cosine(w, a) for a in A) / len(A)
    mean_b = sum(_cosine(w, b) for b in B) / len(B)
    return mean_a - mean_b


def effect_size(X: Sequence[Sequence[float]], Y: Sequence[Sequence[float]],
                A: Sequence[Sequence[float]], B: Sequence[Sequence[float]],
                ddof: int = 1) -> float:
    """WEAT effect size d (Caliskan eq. 3). See the module docstring on `ddof`."""
    s_x: List[float] = [association(x, A, B) for x in X]
    s_y: List[float] = [association(y, A, B) for y in Y]

    mean_x = sum(s_x) / len(s_x)
    mean_y = sum(s_y) / len(s_y)

    pooled = s_x + s_y
    mean_pooled = sum(pooled) / len(pooled)
    variance = sum((s - mean_pooled) ** 2 for s in pooled) / (len(pooled) - ddof)
    std = math.sqrt(variance)

    if std == 0.0:
        return 0.0
    return (mean_x - mean_y) / std
