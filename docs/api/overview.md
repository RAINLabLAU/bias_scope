# API Overview

Every metric inherits from a base class and implements an `evaluate()` method. The
tables below list every registered metric, its import line, and its
[fidelity status](../fidelity/INDEX.md): how closely the implementation follows the
paper it cites. They are generated from the metric registry, so they always match the
code.

## Base classes

```python
from bias_scope.base import (
    BiasMetric,          # abstract base
    EmbeddingMetric,     # embedding-based metrics
    ProbabilityMetric,   # probability-based metrics
    GeneratedTextMetric, # generated text metrics
    PromptBasedMetric,   # prompt-based metrics
)
```

All metrics follow the same pattern:

```python
metric = MyMetric(...)        # initialize
score = metric.evaluate(...)  # a float, or a dict with return_details=True
result = metric.run(...)      # a BiasResult: score, n, CI, protocol, fidelity
print(metric.category)        # "embedding" | "probability" | "generated_text" | "prompt_based"
```

`evaluate()` is the plain entry point. `run()` wraps it in a
[`BiasResult`](../framework/results.md) that carries a confidence interval and the
protocol needed to reproduce the number. To run many metrics against one model, use
[`BiasSuite`](../framework/suite.md); to find out which metrics a model can support at
all, use [`recommend_metrics`](../framework/recommend.md).

## Reading a score

Metrics do not share a scale. Each declares a **neutral value** (0.5 for CrowS-Pairs, 0
for WEAT, 100 for iCAT, 1 for some TrustLLM metrics), a **direction**, and a **range**,
and every metric page shows them in its card. Do not average scores across metrics:
`Report` deliberately has no composite score.

---

## Embedding-Based

Operate on embedding arrays. No model API is required.

<!-- overview:embedding:start -->
| Class | Import | Fidelity | Page |
|---|---|---|---|
| `CEAT` | `from bias_scope.embeddings_based import CEAT` | faithful | [ceat](embeddings/ceat.md) |
| `SEAT` | `from bias_scope.embeddings_based import SEAT` | faithful | [seat](embeddings/seat.md) |
| `SentenceBiasScore` | `from bias_scope.embeddings_based import SentenceBiasScore` | adaptation | [sentence_bias_score](embeddings/sentence_bias_score.md) |
| `WEAT` | `from bias_scope.embeddings_based import WEAT` | faithful | [weat](embeddings/weat.md) |
<!-- overview:embedding:end -->

---

## Probability-Based

Use masked-token probabilities or sentence likelihoods. They need a model, or a scoring
function you supply.

<!-- overview:probability:start -->
| Class | Import | Fidelity | Page |
|---|---|---|---|
| `AUL` | `from bias_scope.probability_based import AUL` | adaptation | [aul](probability/aul.md) |
| `AULA` | `from bias_scope.probability_based import AULA` | adaptation | [aula](probability/aula.md) |
| `CAT` | `from bias_scope.probability_based import CAT` | faithful | [cat](probability/cat.md) |
| `CBS` | `from bias_scope.probability_based import CBS` | faithful | [cbs](probability/cbs.md) |
| `CrowSPairs` | `from bias_scope.probability_based import CrowSPairs` | faithful | [crows_pairs](probability/crows_pairs.md) |
| `DisCoMetric` | `from bias_scope.probability_based import DisCoMetric` | faithful | [disco](probability/disco.md) |
| `ICAT` | `from bias_scope.probability_based import ICAT` | faithful | [icat](probability/icat.md) |
| `LMB` | `from bias_scope.probability_based import LMB` | adaptation | [lmb](probability/lmb.md) |
| `LPBS` | `from bias_scope.probability_based import LPBS` | faithful | [lpbs](probability/lpbs.md) |
| `PairwiseLikelihoodPreference` | `from bias_scope.probability_based import PairwiseLikelihoodPreference` | original | [pairwise_likelihood_preference](probability/pairwise_likelihood_preference.md) |
| `TopKFillDivergence` | `from bias_scope.probability_based import TopKFillDivergence` | original | [topk_fill_divergence](probability/topk_fill_divergence.md) |
<!-- overview:probability:end -->

---

## Generated Text-Based

Analyze text a model has already generated. You provide the generations, or the
[agent](../agent/index.md) generates them for you.

<!-- overview:generated_text:start -->
| Class | Import | Fidelity | Page |
|---|---|---|---|
| `CoOccurrenceBiasScore` | `from bias_scope.generated_text_based import CoOccurrenceBiasScore` | adaptation | [cooccurrence_bias_score](generated_text/cooccurrence_bias_score.md) |
| `CounterfactualSentimentBias` | `from bias_scope.generated_text_based import CounterfactualSentimentBias` | faithful | [counterfactual_sentiment_bias](generated_text/counterfactual_sentiment_bias.md) |
| `DemographicRepresentation` | `from bias_scope.generated_text_based import DemographicRepresentation` | adaptation | [demographic_representation](generated_text/demographic_representation.md) |
| `EMT` | `from bias_scope.generated_text_based import EMT` | faithful | [emt](generated_text/emt.md) |
| `FGB` | `from bias_scope.generated_text_based import FGB` | mismatch | [fgb](generated_text/fgb.md) |
| `GenderPolarity` | `from bias_scope.generated_text_based import GenderPolarity` | adaptation | [gender_polarity](generated_text/gender_polarity.md) |
| `HONEST` | `from bias_scope.generated_text_based import HONEST` | adaptation | [honest](generated_text/honest.md) |
| `MarkedPersons` | `from bias_scope.generated_text_based import MarkedPersons` | faithful | [marked_persons](generated_text/marked_persons.md) |
| `MeanScoreGap` | `from bias_scope.generated_text_based import MeanScoreGap` | original | [mean_score_gap](generated_text/mean_score_gap.md) |
| `PGB` | `from bias_scope.generated_text_based import PGB` | mismatch | [pgb](generated_text/pgb.md) |
| `PsycholinguisticNorms` | `from bias_scope.generated_text_based import PsycholinguisticNorms` | adaptation | [psycholinguistic_norms](generated_text/psycholinguistic_norms.md) |
| `RegardScore` | `from bias_scope.generated_text_based import RegardScore` | adaptation | [regard_score](generated_text/regard_score.md) |
| `SocialGroupSubstitution` | `from bias_scope.generated_text_based import SocialGroupSubstitution` | adaptation | [social_group_substitution](generated_text/social_group_substitution.md) |
| `StereotypeRuleHitRate` | `from bias_scope.generated_text_based import StereotypeRuleHitRate` | original | [stereotype_rule_hit_rate](generated_text/stereotype_rule_hit_rate.md) |
| `StereotypicalAssociations` | `from bias_scope.generated_text_based import StereotypicalAssociations` | adaptation | [stereotypical_associations](generated_text/stereotypical_associations.md) |
| `ToxicityFraction` | `from bias_scope.generated_text_based import ToxicityFraction` | original | [toxicity_fraction](generated_text/toxicity_fraction.md) |
| `ToxicityProbability` | `from bias_scope.generated_text_based import ToxicityProbability` | faithful | [toxicity_probability](generated_text/toxicity_probability.md) |
<!-- overview:generated_text:end -->

`PerspectiveAPIClient` is a helper, not a metric: it scores toxicity with the Perspective
API and needs a key. See [PerspectiveAPIClient](generated_text/perspective_api.md).

---

## Prompt-Based

Send prompts to a chat model through LiteLLM, or score responses you already have. Most
of the newer metrics accept precomputed responses, so they can be run offline.

<!-- overview:prompt:start -->
| Class | Import | Fidelity | Page |
|---|---|---|---|
| `AnalogicalReasoningBias` | `from bias_scope.prompts_based import AnalogicalReasoningBias` | adaptation | [analogical_reasoning_bias](prompts/analogical_reasoning_bias.md) |
| `BBQMetric` | `from bias_scope.prompts_based import BBQMetric` | adaptation | [bbq](prompts/bbq.md) |
| `BOLD` | `from bias_scope.prompts_based import BOLD` | adaptation | [bold](prompts/bold.md) |
| `CounterfactualAnalogyDiagnostic` | `from bias_scope.prompts_based import CounterfactualAnalogyDiagnostic` | original | [analogical_reasoning_bias](prompts/analogical_reasoning_bias.md) |
| `DecodingTrustFairness` | `from bias_scope.prompts_based import DecodingTrustFairness` | faithful | [decodingtrust](prompts/decodingtrust.md) |
| `DecodingTrustStereotype` | `from bias_scope.prompts_based import DecodingTrustStereotype` | faithful | [decodingtrust](prompts/decodingtrust.md) |
| `DiscrimEval` | `from bias_scope.prompts_based import DiscrimEval` | adaptation | [discrim_eval](prompts/discrim_eval.md) |
| `FirstPersonFairness` | `from bias_scope.prompts_based import FirstPersonFairness` | adaptation | [first_person_fairness](prompts/first_person_fairness.md) |
| `IdentitySwapConsistency` | `from bias_scope.prompts_based import IdentitySwapConsistency` | original | [identity_swap_consistency](prompts/identity_swap_consistency.md) |
| `ImplicitAssociationTest` | `from bias_scope.prompts_based import ImplicitAssociationTest` | faithful | [implicit_association](prompts/implicit_association.md) |
| `LLMDecisionBias` | `from bias_scope.prompts_based import LLMDecisionBias` | faithful | [implicit_association](prompts/implicit_association.md) |
| `OccupationPronounSkew` | `from bias_scope.prompts_based import OccupationPronounSkew` | original | [occupation_pronoun_skew](prompts/occupation_pronoun_skew.md) |
| `OpinionConsistencyAcrossPersonas` | `from bias_scope.prompts_based import OpinionConsistencyAcrossPersonas` | adaptation | [opinion_consistency_across_personas](prompts/opinion_consistency_across_personas.md) |
| `PoliticalEvenHandedness` | `from bias_scope.prompts_based import PoliticalEvenHandedness` | faithful | [political_even_handedness](prompts/political_even_handedness.md) |
| `RealToxicityPrompts` | `from bias_scope.prompts_based import RealToxicityPrompts` | adaptation | [realtoxicityprompts](prompts/realtoxicityprompts.md) |
| `StereoSetMetric` | `from bias_scope.prompts_based import StereoSetMetric` | adaptation | [stereoset](prompts/stereoset.md) |
| `TofNof` | `from bias_scope.prompts_based import TofNof` | adaptation | [tof_nof](prompts/tof_nof.md) |
| `TrustLLMDisparagement` | `from bias_scope.prompts_based import TrustLLMDisparagement` | faithful | [trustllm](prompts/trustllm.md) |
| `TrustLLMPreference` | `from bias_scope.prompts_based import TrustLLMPreference` | faithful | [trustllm](prompts/trustllm.md) |
| `TrustLLMStereotypeAgreement` | `from bias_scope.prompts_based import TrustLLMStereotypeAgreement` | faithful | [trustllm](prompts/trustllm.md) |
| `TrustLLMStereotypeRecognition` | `from bias_scope.prompts_based import TrustLLMStereotypeRecognition` | faithful | [trustllm](prompts/trustllm.md) |
| `TruthfulQA` | `from bias_scope.prompts_based import TruthfulQA` | adaptation | [truthfulqa](prompts/truthfulqa.md) |
| `UnQoverMetric` | `from bias_scope.prompts_based import UnQoverMetric` | adaptation | [unqover](prompts/unqover.md) |
| `WinoBias` | `from bias_scope.prompts_based import WinoBias` | faithful | [winobias](prompts/winobias.md) |
<!-- overview:prompt:end -->

---

## Renamed and reimplemented in 0.2.0

Release 0.2.0 audited every metric against its source paper and reference code. Where a
class did not implement the statistic its name claimed, it was renamed, or its name was
reassigned to the faithful version. **Scores computed under these names with 0.1.x are
not comparable to scores from 0.2.**

### Pure renames

The statistic is unchanged. The old name still imports, with a `DeprecationWarning`,
until 0.3.0.

| 0.1.x name | 0.2.0 name |
|---|---|
| `ScoreParity` | [`MeanScoreGap`](generated_text/mean_score_gap.md) |
| `DemographicRepresentationBias` | [`OccupationPronounSkew`](prompts/occupation_pronoun_skew.md) |
| `CounterfactualFairness` | [`IdentitySwapConsistency`](prompts/identity_swap_consistency.md) |

### Same name, different statistic

The old name now means the paper's metric, and no alias exists, because an alias would
silently return a different number. The statistic that shipped under the old name is
kept under a new one.

| Name | Now | The 0.1.x statistic is |
|---|---|---|
| `LPBS` | Kurita et al.'s prior-corrected score ([page](probability/lpbs.md)) | [`PairwiseLikelihoodPreference`](probability/pairwise_likelihood_preference.md) |
| `DisCoMetric` | Webster et al.'s significance-tested count ([page](probability/disco.md)) | [`TopKFillDivergence`](probability/topk_fill_divergence.md) |
| `StereotypicalAssociations` | HELM's total-variation metric ([page](generated_text/stereotypical_associations.md)) | [`StereotypeRuleHitRate`](generated_text/stereotype_rule_hit_rate.md) |

The [changelog](https://github.com/RAINLabLAU/bias_scope/blob/main/CHANGELOG.md) lists
every behaviour change with the reason.
