"""Probability-based bias metrics."""

# Metrics that don't require torch (numpy only)
from bias_scope.probability_based.aul import AUL
from bias_scope.probability_based.aula import AULA
from bias_scope.probability_based.cat import CAT
from bias_scope.probability_based.crows_pairs import CrowSPairs
from bias_scope.probability_based.icat import ICAT
from bias_scope.probability_based.lmb import LMB

# Metrics that require torch (make optional)
try:
    from bias_scope.probability_based.cbs import CBS
    from bias_scope.probability_based.disco import DisCoMetric
    from bias_scope.probability_based.lpbs import LPBS

    _TORCH_METRICS_AVAILABLE = True
except ImportError:
    _TORCH_METRICS_AVAILABLE = False
    CBS = DisCoMetric = LPBS = None

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
]
