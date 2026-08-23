"""Generated text-based bias metrics."""

from bias_scope.generated_text_based.cooccurrence_bias_score import (
    CoOccurrenceBiasScore,
)
from bias_scope.generated_text_based.counterfactual_sentiment_bias import (
    CounterfactualSentimentBias,
)
from bias_scope.generated_text_based.demographic_representation import (
    DemographicRepresentation,
)
from bias_scope.generated_text_based.emt import EMT
from bias_scope.generated_text_based.fgb import FGB
from bias_scope.generated_text_based.gender_polarity import GenderPolarity
from bias_scope.generated_text_based.honest import HONEST
from bias_scope.generated_text_based.marked_persons import MarkedPersons
from bias_scope.generated_text_based.mean_score_gap import MeanScoreGap
from bias_scope.generated_text_based.perspective_api import PerspectiveAPIClient
from bias_scope.generated_text_based.pgb import PGB
from bias_scope.generated_text_based.psycholinguistic_norms import PsycholinguisticNorms
from bias_scope.generated_text_based.regard_score import RegardScore
from bias_scope.generated_text_based.social_group_substitution import (
    SocialGroupSubstitution,
)
from bias_scope.generated_text_based.stereotype_rule_hit_rate import (
    StereotypeRuleHitRate,
)
from bias_scope.generated_text_based.stereotypical_associations import (
    StereotypicalAssociations,
)
from bias_scope.generated_text_based.toxicity_fraction import ToxicityFraction
from bias_scope.generated_text_based.toxicity_probability import ToxicityProbability

# Public API - classes only
__all__ = [
    "SocialGroupSubstitution",
    "CoOccurrenceBiasScore",
    "CounterfactualSentimentBias",
    "DemographicRepresentation",
    "StereotypeRuleHitRate",
    "StereotypicalAssociations",
    "MarkedPersons",
    "EMT",
    "FGB",
    "PGB",
    "ToxicityFraction",
    "ToxicityProbability",
    "RegardScore",
    "MeanScoreGap",
    "ScoreParity",
    "GenderPolarity",
    "HONEST",
    "PerspectiveAPIClient",
    "PsycholinguisticNorms",
]


#: Renamed in 0.2.0 (pure rename; the statistic is unchanged). The old name
#: stays importable until 0.3.0 and warns. See bias_scope/_deprecation.py.
_RENAMED = {
    "ScoreParity": "MeanScoreGap",
    "StereotypicalAssociationsRules": "StereotypeRuleHitRate",
}


def __getattr__(name: str):
    if name in _RENAMED:
        from bias_scope._deprecation import deprecated_alias

        value = deprecated_alias(globals()[_RENAMED[name]], name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
