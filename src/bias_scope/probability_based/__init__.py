"""Probability-based bias metrics."""

# Metrics that don't require torch (numpy only)
from bias_scope.probability_based.aul import AUL
from bias_scope.probability_based.aula import AULA
from bias_scope.probability_based.cat import CAT
from bias_scope.probability_based.crows_pairs import CrowSPairs
from bias_scope.probability_based.icat import ICAT
from bias_scope.probability_based.lmb import LMB
from bias_scope.probability_based.lpbs import LPBS
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

try:
    from bias_scope.probability_based.disco import DisCoMetric
except ImportError as exc:
    DisCoMetric = _torch_dependency_stub("DisCoMetric", exc)

# Public API - classes only
__all__ = [
    "CrowSPairs",
    "CAT",
    "AUL",
    "ICAT",
    "AULA",
    "LMB",
    "LPBS",
    "CBS",
    "DisCoMetric",
    "BertPLLScorer",
    "TokenPredictionScorer",
]
