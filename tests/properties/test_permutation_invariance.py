"""Metamorphic property 4 (PLAN.md 6.2): item order must not change a score.

This is the seed example for ``tests/properties/``. Phase 3 parametrises all
five properties over every metric via ``validation/registry.yaml``; until then
this file exists so the layout and the shape of a property test are fixed.
"""

import numpy as np
import pytest

from bias_scope.embeddings_based import WEAT

# Reordering items reassociates the floating-point sums inside the means and
# the pooled standard deviation, so invariance is exact in real arithmetic but
# only holds to roughly machine epsilon in float64. Observed drift on the
# fixture below is ~1e-16; 1e-12 leaves headroom without hiding a real change.
TOLERANCE = 1e-12


def test_weat_is_invariant_to_target_item_order(tiny_embeddings):
    """Shuffling words within a target set leaves the effect size unchanged.

    WEAT's effect size is built from per-word association means and their
    pooled standard deviation, both of which are order-independent.
    """
    weat = WEAT()
    targets = (tiny_embeddings["target_a"], tiny_embeddings["target_b"])
    attributes = (tiny_embeddings["attribute_a"], tiny_embeddings["attribute_b"])

    baseline = weat.evaluate(targets, attributes)

    rng = np.random.default_rng(0)
    shuffled = (
        targets[0][rng.permutation(len(targets[0]))],
        targets[1][rng.permutation(len(targets[1]))],
    )
    assert weat.evaluate(shuffled, attributes) == pytest.approx(baseline, abs=TOLERANCE)


def test_weat_is_invariant_to_attribute_item_order(tiny_embeddings):
    """The same holds for the attribute sets."""
    weat = WEAT()
    targets = (tiny_embeddings["target_a"], tiny_embeddings["target_b"])
    attributes = (tiny_embeddings["attribute_a"], tiny_embeddings["attribute_b"])

    baseline = weat.evaluate(targets, attributes)

    rng = np.random.default_rng(1)
    shuffled = (
        attributes[0][rng.permutation(len(attributes[0]))],
        attributes[1][rng.permutation(len(attributes[1]))],
    )
    assert weat.evaluate(targets, shuffled) == pytest.approx(baseline, abs=TOLERANCE)
