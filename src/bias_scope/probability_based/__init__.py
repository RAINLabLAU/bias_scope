"""Probability-based bias metrics."""

# Metrics that don't require torch (numpy only)
from bias_scope.probability_based.aul import AUL
from bias_scope.probability_based.aula import AULA
from bias_scope.probability_based.cat import CAT
from bias_scope.probability_based.crows_pairs import CrowSPairs
from bias_scope.probability_based.icat import ICAT
from bias_scope.probability_based.lmb import LMB
from bias_scope.probability_based.lpbs import LPBS
from bias_scope.probability_based.pairwise_likelihood_preference import (
    PairwiseLikelihoodPreference,
)
from bias_scope.probability_based.scorers import BertPLLScorer, TokenPredictionScorer


def _torch_dependency_stub(class_name: str, original_error: ImportError):
    """Create a class-like placeholder for optional torch-backed metrics."""

    class _MissingTorchDependency:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                f"{class_name} requires optional torch dependencies. "
                "Please install bias-scope[torch] to use this metric."
            ) from original_error

    _MissingTorchDependency.__name__ = class_name
    _MissingTorchDependency.__qualname__ = class_name
    _MissingTorchDependency.__module__ = __name__
    _MissingTorchDependency.__doc__ = (
        f"Placeholder for {class_name}; install bias-scope[torch] to use it."
    )
    return _MissingTorchDependency


# Metrics that require torch (optional dependency)
try:
    from bias_scope.probability_based.cbs import CBS
except ImportError as exc:
    CBS = _torch_dependency_stub("CBS", exc)

# DisCoMetric is pure Python since 0.2.0 — the caller supplies the fills, so it
# needs no torch. TopKFillDivergence (the v0.1.1 statistic) still loads a model
# itself, so it keeps the optional-dependency stub.
from bias_scope.probability_based.disco import DisCoMetric

try:
    from bias_scope.probability_based.topk_fill_divergence import TopKFillDivergence
except ImportError as exc:
    TopKFillDivergence = _torch_dependency_stub("TopKFillDivergence", exc)

# Public API - classes only
__all__ = [
    "CrowSPairs",
    "CAT",
    "AUL",
    "ICAT",
    "AULA",
    "LMB",
    "LPBS",
    "PairwiseLikelihoodPreference",
    "CBS",
    "DisCoMetric",
    "TopKFillDivergence",
    "BertPLLScorer",
    "TokenPredictionScorer",
]
