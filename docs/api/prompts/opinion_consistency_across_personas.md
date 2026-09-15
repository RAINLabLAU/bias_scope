# OpinionConsistencyAcrossPersonas

`OpinionConsistencyAcrossPersonas` implements the Consistency analysis in
Santurkar et al., [*Whose Opinions Do Language Models Reflect?* (ICML
2023)](https://proceedings.mlr.press/v202/santurkar23a.html). It is **not**
agreement among answers generated under different persona prompts.

For model option distribution \(D_M\), human subgroup distribution \(D_H\),
and actual response-option coordinates \(o\), alignment is
\(1 - W_1(D_M,D_H)/W_1(\delta_{\min o},\delta_{\max o})\). The metric averages
that alignment across questions for each subgroup/topic (representativeness).
Coordinates are passed directly to the one-dimensional Wasserstein calculation:
they need not be sorted or unique (for example, an official midpoint hedging
option is valid), but their range must be non-zero. The metric then
selects each topic's best subgroup, and compares it with the overall best
subgroup. Overall subgroup representativeness weights valid topics equally.
Consistency is the fraction of matching topics, averaged across demographic
attributes. The paper does not specify tie handling. BiasScope deterministically
uses the highest `subgroup_order` and reports all ties; this is a BiasScope
extension. The official notebook's sort/groupby result is ordering-dependent
and is represented separately as notebook-incidental behavior.

The required distributions and metadata come from the official OpinionQA
release, not from persona strings: its `human_resp` responses and metadata,
`model_input` question/topic mapping, and model-run outputs. The official code
uses first-token option log probabilities (including both leading-space and
unspaced token forms), imputes unavailable options from residual/minimum
probability, then normalizes the non-refusal options to form `D_M`. Human
responses are survey-weighted within each official demographic subgroup to form
`D_H`. BiasScope accepts those precomputed distributions; it does not make paid
API calls or silently replace missing probabilities with a hard choice.

```python
from bias_scope.prompts_based.opinion_consistency_across_personas import OpinionConsistencyAcrossPersonas

models = [{"question_id": "Q1", "distribution": [0.7, 0.3],
           "ordinal": [0, 1], "topic": "Politics"}]
humans = [
    {"question_id": "Q1", "attribute": "POLPARTY", "subgroup": "Democrat",
     "subgroup_order": 0, "distribution": [0.8, 0.2]},
    {"question_id": "Q1", "attribute": "POLPARTY", "subgroup": "Republican",
     "subgroup_order": 1, "distribution": [0.2, 0.8]},
]
result = OpinionConsistencyAcrossPersonas().evaluate(models, humans, return_details=True)
```

To parse the official processed CSVs, load the official `topic_mapping.npy`
yourself and pass its mapping to `records_from_official_csv`. The official
`human.csv` does not contain group order, so also pass the ordered subgroup
options from each official `metadata.csv` as `subgroup_orders`; a processed
combined CSV may instead carry `group_order`. This parser does not claim that
the dataset is bundled or automatically downloaded.

The former majority-choice statistic remains available as
`PersonaAnswerConsistency`. It is explicitly a custom diagnostic with output
`persona_answer_consistency`, and must not be compared to the paper's result.

## Offline official-data parity

The private research-only parity protocol is pinned to the official repository
and paper-evidence commit `612e5c1592803c1900b43bd832a27cddb3707f60`. It never
downloads data or calls a model: callers supply the official CodaLab artifacts.
It distinguishes the paper's equal-coarse-topic formula from a literal
`consistency.ipynb` reconstruction. The notebook averages the overall best
subgroup over question rows and appears to include `Overall`; normal BiasScope
uses the paper formula, excludes `Overall`, requires a common valid-question
set, and uses deterministic ties. Tie behavior is therefore a BiasScope
extension, not a paper rule. The paper models were historical OpenAI/AI21 APIs,
not released local checkpoints; their released runs are the evidence target.
