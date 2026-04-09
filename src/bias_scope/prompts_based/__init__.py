"""Prompt-based bias metrics."""

try:
    from bias_scope.prompts_based.analogical_reasoning_bias import AnalogicalReasoningBias
except ImportError:
    AnalogicalReasoningBias = None

try:
    from bias_scope.prompts_based.bbq import BBQMetric
except ImportError:
    BBQMetric = None

try:
    from bias_scope.prompts_based.counterfactual_fairness import CounterfactualFairness
except ImportError:
    CounterfactualFairness = None

try:
    from bias_scope.prompts_based.demographic_representation_bias import (
        DemographicRepresentationBias,
    )
except ImportError:
    DemographicRepresentationBias = None

try:
    from bias_scope.prompts_based.stereoset import StereoSetMetric
except ImportError:
    StereoSetMetric = None

try:
    from bias_scope.prompts_based.tof_nof import TofNof
except ImportError:
    TofNof = None

try:
    from bias_scope.prompts_based.unqover import UnQoverMetric
except ImportError:
    UnQoverMetric = None

try:
    from bias_scope.prompts_based.bold import BOLD
except ImportError:
    BOLD = None

try:
    from bias_scope.prompts_based.opinion_consistency_across_personas import (
        OpinionConsistencyAcrossPersonas,
    )
except ImportError:
    OpinionConsistencyAcrossPersonas = None

try:
    from bias_scope.prompts_based.realtoxicityprompts import RealToxicityPrompts
except ImportError:
    RealToxicityPrompts = None

try:
    from bias_scope.prompts_based.truthfulqa import TruthfulQA
except ImportError:
    TruthfulQA = None

__all__ = [
    "AnalogicalReasoningBias",
    "BBQMetric",
    "BOLD",
    "CounterfactualFairness",
    "DemographicRepresentationBias",
    "OpinionConsistencyAcrossPersonas",
    "RealToxicityPrompts",
    "StereoSetMetric",
    "TofNof",
    "TruthfulQA",
    "UnQoverMetric",
]
