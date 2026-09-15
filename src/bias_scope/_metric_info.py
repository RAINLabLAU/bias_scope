"""MetricInfo for every metric in the library (PLAN.md 5.1).

Generated once and then maintained by hand. Kept in its own module so the
metric modules stay focused on their formula, and so the whole table can be
read in one place — which is what makes the audit gap obvious.

`fidelity` is **not** guesswork: it is `unaudited` unless the metric's
`sources/SOURCES.yaml` entry says `read` and a `docs/fidelity/<metric>.md`
records the verdict. `tests/test_metadata.py` enforces exactly that.
"""

from __future__ import annotations

from bias_scope.metadata import MetricInfo

#: Class name -> metadata.
METRIC_INFO: dict[str, MetricInfo] = {
    "AUL": MetricInfo(
        name="AUL",
        family="probability",
        access=('logits',),
        neutral_value=50.0,
        direction="higher_more_biased",
        value_range=(0.0, 100.0),
        fidelity="faithful",
        reference=(
            'Unmasking the Mask -- Evaluating Social Biases in Masked Language Models, AAAI '
            '2022 — https://arxiv.org/abs/2104.07496'
        ),
        reference_impl='https://github.com/kanekomasahiro/evaluate_bias_in_mlm @ 6b10239974a7',
        resource_binding="dataset",
        deviation_note='',
        fidelity_note='docs/fidelity/aul_aula.md',
    ),
    "AULA": MetricInfo(
        name="AULA",
        family="probability",
        access=('logits',),
        neutral_value=50.0,
        direction="higher_more_biased",
        value_range=(0.0, 100.0),
        fidelity="faithful",
        reference=(
            'Unmasking the Mask -- Evaluating Social Biases in Masked Language Models, AAAI '
            '2022 — https://arxiv.org/abs/2104.07496'
        ),
        reference_impl='https://github.com/kanekomasahiro/evaluate_bias_in_mlm @ 6b10239974a7',
        resource_binding="dataset",
        deviation_note='',
        fidelity_note='docs/fidelity/aul_aula.md',
    ),
    "AnalogicalReasoningBias": MetricInfo(
        name="AnalogicalReasoningBias",
        family="prompt",
        access=('chat',),
        neutral_value=0.0,
        direction="higher_more_biased",
        value_range=(0.0, 1.0),
        fidelity="original",
        reference=(
            'BiasScope original. Inspired by Abid et al. 2021 '
            '(https://arxiv.org/abs/2101.05783) and Bolukbasi et al. 2016 '
            '(https://arxiv.org/abs/1607.06520); neither defines a metric of this name.'
        ),
        reference_impl='',
        resource_binding="dataset",
        deviation_note=(
            "BiasScope's own operationalization: prompts a chat model with analogy "
            'completions and scores the answers. Bolukbasi et al. define direct and indirect '
            'bias over a gender subspace; Abid et al. define no reusable score. Neither is '
            'implemented here. See docs/fidelity/originals.md.'
        ),
        fidelity_note='docs/fidelity/originals.md',
    ),
    "BBQMetric": MetricInfo(
        name="BBQMetric",
        family="prompt",
        access=('chat',),
        neutral_value=0.0,
        direction="signed",
        value_range=(-1.0, 1.0),
        fidelity="faithful",
        reference=(
            'BBQ: A Hand-Built Bias Benchmark for Question Answering, '
            'Findings of ACL 2022 — https://arxiv.org/abs/2110.08193'
        ),
        reference_impl='https://github.com/nyu-mll/BBQ @ bea11bd97d79',
        resource_binding="dataset",
        deviation_note='',
        fidelity_note='docs/fidelity/bbq.md',
    ),
    "BOLD": MetricInfo(
        name="BOLD",
        family="prompt",
        access=('chat',),
        neutral_value=0.0,
        direction="higher_more_biased",
        value_range=(0.0, float("inf")),
        fidelity="faithful",
        reference=(
            'BOLD: Dataset and Metrics for Measuring Biases in Open-Ended Language '
            'Generation, FAccT 2021 — https://arxiv.org/abs/2101.11718'
        ),
        reference_impl='https://github.com/amazon-science/bold @ 3ad652c773f5',
        resource_binding="dataset",
        deviation_note='',
        fidelity_note='docs/fidelity/bold.md',
    ),
    "CAT": MetricInfo(
        name="CAT",
        family="probability",
        access=('logits',),
        neutral_value=50.0,
        direction="higher_more_biased",
        value_range=(0.0, 100.0),
        fidelity="faithful",
        reference=(
            'StereoSet: Measuring stereotypical bias in pretrained language models, ACL 2021 '
            '— https://arxiv.org/abs/2004.09456'
        ),
        reference_impl='https://github.com/moinnadeem/StereoSet @ ead7d086a64a',
        resource_binding="dataset",
        deviation_note='',
        fidelity_note='docs/fidelity/stereoset_family.md',
    ),
    "CBS": MetricInfo(
        name="CBS",
        family="probability",
        access=('logits',),
        neutral_value=0.0,
        direction="higher_more_biased",
        value_range=(0.0, float("inf")),
        fidelity="faithful",
        reference=(
            'Mitigating Language-Dependent Ethnic Bias in BERT, EMNLP 2021 — '
            'https://arxiv.org/abs/2109.05704'
        ),
        reference_impl='https://github.com/jaimeenahn/ethnic_bias @ a115eb7c3af7',
        # Twelve, from the twelve keys in the authors' `configuration.py`, each
        # with its own templates, nationality list and BERT checkpoint. The
        # paper's title is "Language-Dependent Ethnic Bias"; declaring English
        # only understated the metric. See bias_scope.multilingual.
        languages=('ko', 'en', 'de', 'fr', 'es', 'zh', 'ja', 'tr', 'ar', 'el',
                   'th', 'vi'),
        resource_binding="lexicon",
        deviation_note='',
        fidelity_note='docs/fidelity/cbs_lmb.md',
    ),
    "CEAT": MetricInfo(
        name="CEAT",
        family="embedding",
        access=('embeddings',),
        neutral_value=0.0,
        direction="signed",
        value_range=(float("-inf"), float("inf")),
        fidelity="faithful",
        reference=(
            'Detecting Emergent Intersectional Biases: Contextualized Word Embeddings Contain '
            'a Distribution of Human-like Biases, AIES 2021 — '
            'https://arxiv.org/abs/2006.03955'
        ),
        reference_impl='https://github.com/weiguowilliam/CEAT @ 497e2958a152',
        resource_binding="dataset",
        deviation_note='',
        fidelity_note='docs/fidelity/ceat.md',
    ),
    "CoOccurrenceBiasScore": MetricInfo(
        name="CoOccurrenceBiasScore",
        family="generated_text",
        access=('completions',),
        neutral_value=0.0,
        direction="signed",
        value_range=(float("-inf"), float("inf")),
        fidelity="adaptation",
        reference=(
            'Identifying and Reducing Gender Bias in Word-Level Language Models, NAACL SRW '
            '2019 — https://arxiv.org/abs/1904.03035'
        ),
        reference_impl='https://github.com/BordiaS/language-model-bias @ 59f6584e8a85',
        resource_binding="lexicon",
        deviation_note=(
            "Omits the paper's sum_i c(w_i, g) normalisation - the total words appearing in "
            "each group's context windows. That is a constant additive offset, so rankings "
            'and word-to-word comparisons are unaffected, but absolute values are shifted: an '
            "equally-co-occurring word does not score 0, which is this metric's neutral "
            "value. Additive smoothing is also ours, not the paper's. See "
            'docs/fidelity/cooccurrence_bias_score.md and REVIEW_LATER RL-023.'
        ),
        fidelity_note='docs/fidelity/cooccurrence_bias_score.md',
    ),
    "IdentitySwapConsistency": MetricInfo(
        name="IdentitySwapConsistency",
        family="prompt",
        access=('chat',),
        neutral_value=0.0,
        direction="higher_more_biased",
        value_range=(0.0, 1.0),
        fidelity="original",
        reference=(
            'BiasScope original. Inspired by Kusner et al. 2017, Counterfactual Fairness, '
            'NeurIPS 2017 — https://arxiv.org/abs/1703.06856'
        ),
        reference_impl='',
        resource_binding="dataset",
        deviation_note=(
            "BiasScope's own operationalization: cosine similarity between response "
            'embeddings under identity swaps. Kusner et al. define a CAUSAL CRITERION on a '
            'predictor, requiring a causal model - not a score, and not this. The class name '
            'is the exact title of that paper and should be renamed IdentitySwapConsistency '
            '(PLAN.md 5.2; REVIEW_LATER RL-024). See docs/fidelity/originals.md.'
        ),
        fidelity_note='docs/fidelity/originals.md',
    ),
    "CounterfactualSentimentBias": MetricInfo(
        name="CounterfactualSentimentBias",
        family="generated_text",
        access=('completions',),
        neutral_value=0.0,
        direction="higher_more_biased",
        value_range=(0.0, float("inf")),
        fidelity="faithful",
        reference=(
            'Reducing Sentiment Bias in Language Models via Counterfactual Evaluation, '
            'Findings of EMNLP 2020 — https://arxiv.org/abs/1911.03064'
        ),
        reference_impl='',
        resource_binding="classifier",
        deviation_note='',
        fidelity_note='docs/fidelity/huang_metrics.md',
    ),
    "CrowSPairs": MetricInfo(
        name="CrowSPairs",
        family="probability",
        access=('logits',),
        neutral_value=50.0,
        direction="higher_more_biased",
        value_range=(0.0, 100.0),
        fidelity="faithful",
        reference=(
            'CrowS-Pairs: A Challenge Dataset for Measuring Social Biases in Masked Language '
            'Models, EMNLP 2020 — https://arxiv.org/abs/2010.00133'
        ),
        reference_impl=(
            'https://github.com/nyu-mll/crows-pairs '
            '@ 8aaac11c4854'
        ),
        resource_binding="dataset",
        deviation_note=(
            'Faithful in the default `mode="wordpiece"`, which is the pseudo-log-likelihood '
            "over WordPiece tokens that the authors' `metric.py` computes. `mode=\"whitespace\"` "
            'scores whole whitespace words instead; it is kept for continuity with v0.1.x but is '
            'NOT the published protocol and must not be reported as CrowS-Pairs. Reproduction on '
            'bert-base-uncased over all 1,508 pairs: 58.62 vs the published 60.5, Wald intervals '
            'overlapping (results/emnlp/crows_pairs.json).'
        ),
        fidelity_note='docs/fidelity/crows_pairs.md',
    ),
    "DemographicRepresentation": MetricInfo(
        name="DemographicRepresentation",
        family="generated_text",
        access=('completions',),
        neutral_value=0.0,
        direction="higher_more_biased",
        value_range=(0.0, 1.0),
        fidelity="adaptation",
        reference=(
            'Holistic Evaluation of Language Models, TMLR 2023 — '
            'https://arxiv.org/abs/2211.09110'
        ),
        reference_impl='https://github.com/stanford-crfm/helm @ 63754d05db6f',
        resource_binding="lexicon",
        deviation_note=(
            "Faithful to HELM's formula. One protocol deviation: HELM tokenises "
            'with NLTK word_tokenize; the default here is a regex word tokenizer so '
            'the core install stays light. Pass tokenizer=nltk.word_tokenize to '
            'match HELM exactly.'
        ),
        fidelity_note='docs/fidelity/demographic_representation.md',
    ),
    "OccupationPronounSkew": MetricInfo(
        name="OccupationPronounSkew",
        family="prompt",
        access=('chat',),
        neutral_value=1.0,
        direction="signed",
        value_range=(0.0, float("inf")),
        fidelity="original",
        reference=(
            'BiasScope original. Inspired by Zhao et al. 2018 (WinoBias), '
            'Gender Bias in Coreference Resolution, NAACL 2018 — '
            'https://arxiv.org/abs/1804.06876'
        ),
        reference_impl='',
        resource_binding="lexicon",
        deviation_note=(
            "BiasScope's own operationalization: the male/female pronoun-count "
            'ratio in generations for occupation templates. Shipped as '
            '`DemographicRepresentationBias` citing WinoBias through v0.1.1, which '
            'was a mismatch — WinoBias is a coreference F1 gap, not a pronoun count. '
            'Renamed in 0.2.0; a faithful WinoBias is a separate Phase 4 item. '
            'See docs/fidelity/demographic_representation_bias.md.'
        ),
        fidelity_note='docs/fidelity/demographic_representation_bias.md',
    ),
    "DecodingTrustStereotype": MetricInfo(
        name="DecodingTrustStereotype",
        family="prompt",
        access=('chat', 'completions'),
        neutral_value=0.0,
        direction="higher_more_biased",
        value_range=(0.0, 1.0),
        fidelity="faithful",
        reference=(
            'DecodingTrust: A Comprehensive Assessment of Trustworthiness in GPT '
            'Models, NeurIPS 2023 Datasets and Benchmarks — '
            'https://arxiv.org/abs/2306.11698'
        ),
        reference_impl='https://github.com/AI-secure/DecodingTrust @ 161ae8321ced',
        resource_binding="dataset",
        deviation_note=(
            'Reports the agreement rate itself (0 neutral, higher more biased) '
            "rather than the benchmark's `1 - round(mean, 2)` leaderboard score, "
            'which is in details["decodingtrust_score"] with the same rounding.'
        ),
        fidelity_note='docs/fidelity/decodingtrust.md',
    ),
    "DecodingTrustFairness": MetricInfo(
        name="DecodingTrustFairness",
        family="prompt",
        access=('chat', 'completions'),
        neutral_value=0.0,
        direction="higher_more_biased",
        value_range=(0.0, 1.0),
        fidelity="faithful",
        reference=(
            'DecodingTrust, NeurIPS 2023 Datasets and Benchmarks, Sec. 8 '
            '(fairness) — https://arxiv.org/abs/2306.11698'
        ),
        reference_impl='https://github.com/AI-secure/DecodingTrust @ 161ae8321ced',
        resource_binding="dataset",
        deviation_note=(
            "Reports the demographic parity difference itself rather than the "
            "benchmark's (1 - DPD) * 100 leaderboard score, which is in details. "
            'An answer naming both classes is dropped rather than resolved by '
            "the reference's unseeded coin flip."
        ),
        fidelity_note='docs/fidelity/decodingtrust.md',
    ),
    "DiscrimEval": MetricInfo(
        name="DiscrimEval",
        family="prompt",
        access=('chat',),
        neutral_value=0.0,
        direction="higher_more_biased",
        value_range=(0.0, float("inf")),
        fidelity="adaptation",
        reference=(
            'Evaluating and Mitigating Discrimination in Language Model Decisions, 2023 — '
            'https://arxiv.org/abs/2312.03689'
        ),
        reference_impl='https://huggingface.co/datasets/Anthropic/discrim-eval',
        resource_binding="dataset",
        deviation_note=(
            'Tamkin et al. estimate the discrimination score by fitting a MIXED EFFECTS '
            'linear regression, with demographics as fixed effects and decision-question '
            'types plus their interactions as random effects. This computes the marginal '
            'difference in mean logit instead. The two agree when every group answers the '
            'same questions with no question-by-demographic interaction, and diverge under '
            'unbalanced coverage, because the mixed model shrinks noisy per-question '
            'estimates and a plain mean does not. Fitting the full model needs statsmodels, '
            'which the light-core rule keeps out of the default install. See '
            'docs/fidelity/discrim_eval.md.'
        ),
        fidelity_note='docs/fidelity/discrim_eval.md',
    ),
    "DisCoMetric": MetricInfo(
        name="DisCoMetric",
        family="probability",
        access=('logits',),
        neutral_value=0.0,
        direction="higher_more_biased",
        value_range=(0.0, float("inf")),
        fidelity="faithful",
        reference=(
            'Measuring and Reducing Gendered Correlations in Pre-trained Models, '
            'arXiv preprint — https://arxiv.org/abs/2010.06032'
        ),
        reference_impl='',
        resource_binding="lexicon",
        deviation_note='',
        fidelity_note='docs/fidelity/disco.md',
    ),
    "TopKFillDivergence": MetricInfo(
        name="TopKFillDivergence",
        family="probability",
        access=('logits',),
        neutral_value=0.0,
        direction="higher_more_biased",
        value_range=(0.0, float("inf")),
        fidelity="original",
        reference=(
            'BiasScope original. The top-k symmetric difference that shipped as '
            'DisCoMetric through v0.1.1; no cited paper defines it.'
        ),
        reference_impl='',
        resource_binding="lexicon",
        deviation_note=(
            "BiasScope's own operationalization: the size of the symmetric difference "
            'between two prompts\' top-k mask fills. It carries no significance test, '
            "so it is NOT Webster et al.'s DisCo, which counts fills whose prediction "
            'rates differ significantly by a Bonferroni-corrected chi-square, averaged '
            'over templates. Renamed in 0.2.0. See docs/fidelity/disco.md.'
        ),
        fidelity_note='docs/fidelity/disco.md',
    ),
    "EMT": MetricInfo(
        name="EMT",
        family="generated_text",
        access=('completions',),
        neutral_value=0.0,
        direction="higher_more_biased",
        value_range=(0.0, 1.0),
        fidelity="faithful",
        reference=(
            'RealToxicityPrompts: Evaluating Neural Toxic Degeneration in Language Models, '
            'Findings of EMNLP 2020 — https://arxiv.org/abs/2009.11462'
        ),
        reference_impl='https://github.com/allenai/real-toxicity-prompts @ 3beff74a01f',
        resource_binding="classifier",
        deviation_note='',
        fidelity_note='docs/fidelity/toxicity_family.md',
    ),
    "FGB": MetricInfo(
        name="FGB",
        family="generated_text",
        access=('completions',),
        neutral_value=0.0,
        direction="higher_more_biased",
        value_range=(0.0, float("inf")),
        fidelity="mismatch",
        reference=(
            '"I\'m sorry to hear that": Finding New Biases in Language Models with a Holistic '
            'Descriptor Dataset, EMNLP 2022 — https://arxiv.org/abs/2205.09209'
        ),
        reference_impl=(
            'https://github.com/facebookresearch/ResponsibleNLP '
            '@ 0ec714eb0842'
        ),
        resource_binding="classifier",
        deviation_note=(
            'Implements a mean absolute paired gap; Smith et al. Sec. A.7 define the variance '
            'across descriptors of a 217-class style vector, summed over styles. Blocked: the '
            'style classifier is not published (REVIEW_LATER RL-013).'
        ),
        fidelity_note='docs/fidelity/fgb.md',
    ),
    "FirstPersonFairness": MetricInfo(
        name="FirstPersonFairness",
        family="prompt",
        access=('chat',),
        neutral_value=0.0,
        direction="signed",
        value_range=(-1.0, 1.0),
        fidelity="adaptation",
        reference=(
            'First-Person Fairness in Chatbots, Eloundou et al. 2024 (OpenAI), '
            'Sec. 3.3 and Figure 3 — https://arxiv.org/abs/2410.19803'
        ),
        reference_impl='',
        resource_binding="judge",
        deviation_note=(
            'The estimator H = E[h_F - h_R] is the paper\'s exactly, including '
            'the swapped second judging pass and the identical-response rule. '
            'The judge template is the paper\'s Figure 3, which the paper marks '
            '"slightly abbreviated" — the production prompt was not published '
            'and no reference implementation was located (search log in '
            'SOURCES.yaml), so the wording cannot be mirrored verbatim and the '
            'status stays `adaptation` rather than `faithful`.'
        ),
        fidelity_note='docs/fidelity/first_person_fairness.md',
    ),
    "GenderPolarity": MetricInfo(
        name="GenderPolarity",
        family="generated_text",
        access=('completions',),
        neutral_value=0.0,
        direction="signed",
        value_range=(-1.0, 1.0),
        fidelity="adaptation",
        reference=(
            'BOLD: Dataset and Metrics for Measuring Biases in Open-Ended Language '
            'Generation, FAccT 2021 — https://arxiv.org/abs/2101.11718'
        ),
        reference_impl='https://github.com/amazon-science/bold @ 3ad652c773f5',
        resource_binding="lexicon",
        deviation_note=(
            'Three differences from BOLD Sec. 4.5. (1) Reports a continuous mean of signed '
            'token-count ratios; BOLD assigns each text a three-way male/female/neutral label '
            "and reports label proportions. (2) Sign is inverted: BOLD's b_i is positive for "
            "FEMALE (g = she - he); this is positive for masculine. (3) BOLD's second gender- "
            'polarity metric - the Bolukbasi projection with Gender-Wavg / Gender-Max on '
            'hard-debiased Word2Vec - is not implemented at all (REVIEW_LATER RL-025). See '
            'docs/fidelity/bold_metrics.md.'
        ),
        fidelity_note='docs/fidelity/bold_metrics.md',
    ),
    "HONEST": MetricInfo(
        name="HONEST",
        family="generated_text",
        access=('completions',),
        neutral_value=0.0,
        direction="higher_more_biased",
        value_range=(0.0, 1.0),
        fidelity="adaptation",
        reference=(
            'HONEST: Measuring Hurtful Sentence Completion in Language Models, '
            'NAACL 2021 — https://aclanthology.org/2021.naacl-main.191/'
        ),
        reference_impl='https://github.com/MilaNLProc/honest @ 6efc7817cbeb',
        # Six binary-template languages in resources/ of the authors' repo;
        # the queer/non-queer set is English only. See
        # bias_scope.multilingual.MULTILINGUAL_DATASETS['HONEST'].
        languages=('en', 'es', 'fr', 'it', 'pt', 'ro'),
        resource_binding="lexicon",
        deviation_note=(
            "The formula is the paper's, unchanged. Two deviations move the number: "
            'for causal LMs the top-K mask fills are replaced by K sampled '
            'single-token generations, and HurtLex has grown since Nozza\'s 1,072-term '
            'snapshot (1,722 terms in the selection used here). Both are recorded in '
            'the protocol and bracketed in results/emnlp/. See docs/fidelity/honest.md.'
        ),
        fidelity_note='docs/fidelity/honest.md',
    ),
    "ICAT": MetricInfo(
        name="ICAT",
        family="probability",
        access=('logits',),
        neutral_value=100.0,
        direction="lower_more_biased",
        value_range=(0.0, 100.0),
        fidelity="faithful",
        reference=(
            'StereoSet: Measuring stereotypical bias in pretrained language models, ACL 2021 '
            '— https://arxiv.org/abs/2004.09456'
        ),
        reference_impl='https://github.com/moinnadeem/StereoSet @ ead7d086a64a',
        resource_binding="dataset",
        deviation_note='',
        fidelity_note='docs/fidelity/stereoset_family.md',
    ),
    "LMB": MetricInfo(
        name="LMB",
        family="probability",
        access=('logits',),
        neutral_value=0.0,
        direction="signed",
        value_range=(float("-inf"), float("inf")),
        fidelity="adaptation",
        reference=(
            'RedditBias: A Real-World Resource for Bias Evaluation and Debiasing of '
            'Conversational Language Models, ACL 2021 — https://arxiv.org/abs/2106.03521'
        ),
        reference_impl='https://github.com/umanlp/RedditBias @ 61f9ae9458e2',
        resource_binding="dataset",
        deviation_note=(
            "Barikeri et al. report the bias effect as the t-value of a Student's two-tailed "
            'test; evaluate(return_details=False) returns mean_diff instead. The t-statistic '
            'and p-value are computed and available in details, so nothing is missing - the '
            "default scalar is simply not the paper's (REVIEW_LATER RL-026). The paper's "
            '3-sigma outlier rule was added as the default in 0.2.0; the percentile variant '
            "is BiasScope's own. See docs/fidelity/cbs_lmb.md."
        ),
        fidelity_note='docs/fidelity/cbs_lmb.md',
    ),
    "LPBS": MetricInfo(
        name="LPBS",
        family="probability",
        access=('logits',),
        neutral_value=0.0,
        direction="signed",
        value_range=(float("-inf"), float("inf")),
        fidelity="faithful",
        reference=(
            'Measuring Bias in Contextualized Word Representations, GeBNLP 2019 '
            '— https://arxiv.org/abs/1906.07337'
        ),
        reference_impl=(
            'https://github.com/keitakurita/contextual_embedding_bias_measure '
            '@ 18044f87e2ff'
        ),
        resource_binding="dataset",
        deviation_note=(
            'Faithful to the paper. One documented judgement: the authors\' code '
            'appears to read p_prior at the attribute mask rather than the target '
            'mask, and its get_index last branch omits the [CLS] offset; this '
            'implementation follows the paper\'s Sec. 2 step 3 instead. See '
            'docs/fidelity/lpbs.md and REVIEW_LATER RL-012.'
        ),
        fidelity_note='docs/fidelity/lpbs.md',
    ),
    "MarkedPersons": MetricInfo(
        name="MarkedPersons",
        family="generated_text",
        access=('completions',),
        neutral_value=0.0,
        direction="higher_more_biased",
        value_range=(0.0, float("inf")),
        fidelity="faithful",
        reference=(
            'Marked Personas: Using Natural Language Prompts to Measure Stereotypes in '
            'Language Models, ACL 2023 — https://arxiv.org/abs/2305.18189'
        ),
        reference_impl='https://github.com/myracheng/markedpersonas @ 9b3ae82ad262',
        resource_binding="lexicon",
        deviation_note='',
        fidelity_note='docs/fidelity/marked_persons.md',
    ),
    "OpinionConsistencyAcrossPersonas": MetricInfo(
        name="OpinionConsistencyAcrossPersonas",
        family="prompt",
        access=('chat',),
        neutral_value=0.0,
        direction="higher_more_biased",
        value_range=(0.0, 1.0),
        fidelity="adaptation",
        reference=(
            'Santurkar et al. (2023), "Whose Opinions Do Language Models Reflect?", '
            'ICML â€” https://proceedings.mlr.press/v202/santurkar23a.html'
        ),
        reference_impl='https://github.com/tatsu-lab/opinions_qa',
        resource_binding="dataset",
        deviation_note=(
            "Implements the published distributional calculation from precomputed "
            "OpinionQA distributions. Live HELM first-token log-probability collection "
            "and automatic OpinionQA download are intentionally not bundled. The public "
            "metric follows the paper's equal-topic formula; private offline parity can "
            "also reconstruct the paper-era notebook's question-weighted overall choice. "
            "The metadata vocabulary requires an access mode, so 'chat' is retained for "
            "compatibility even though this public precomputed-distribution API makes no call."
        ),
        fidelity_note='docs/api/prompts/opinion_consistency_across_personas.md',
    ),
    "PGB": MetricInfo(
        name="PGB",
        family="generated_text",
        access=('completions',),
        neutral_value=0.0,
        direction="higher_more_biased",
        value_range=(0.0, float("inf")),
        fidelity="mismatch",
        reference=(
            '"I\'m sorry to hear that": Finding New Biases in Language Models with a Holistic '
            'Descriptor Dataset, EMNLP 2022 — https://arxiv.org/abs/2205.09209'
        ),
        reference_impl=(
            'https://github.com/facebookresearch/ResponsibleNLP '
            '@ 0ec714eb0842'
        ),
        resource_binding="classifier",
        deviation_note=(
            'Implements a mean rectified paired gap; Smith et al. Sec. A.7 define FGB restricted '
            'to a style cluster. Blocked on the same missing classifier (REVIEW_LATER RL-013).'
        ),
        fidelity_note='docs/fidelity/pgb.md',
    ),
    "PairwiseLikelihoodPreference": MetricInfo(
        name="PairwiseLikelihoodPreference",
        family="probability",
        access=('logits',),
        neutral_value=0.5,
        direction="higher_more_biased",
        value_range=(0.0, 1.0),
        fidelity="original",
        reference=(
            'BiasScope original. The sentence-pair preference rate shipped as LPBS '
            'through v0.1.1; the comparison is the CrowS-Pairs / StereoSet family\'s, '
            'not any single cited paper\'s.'
        ),
        reference_impl='',
        resource_binding="dataset",
        deviation_note=(
            'BiasScope\'s own operationalization: the proportion of sentence pairs '
            'where the stereotype sentence has the higher log-probability. It carries '
            'no prior correction and is NOT Kurita et al. 2019\'s Log Probability Bias '
            'Score, which shipped under this behaviour until v0.2.0. See '
            'docs/fidelity/lpbs.md.'
        ),
        fidelity_note='docs/fidelity/lpbs.md',
    ),
    "PoliticalEvenHandedness": MetricInfo(
        name="PoliticalEvenHandedness",
        family="prompt",
        access=('chat',),
        neutral_value=0.0,
        direction="higher_more_biased",
        value_range=(0.0, 1.0),
        fidelity="faithful",
        reference=(
            'Anthropic 2025, Measuring political bias in Claude — '
            'https://www.anthropic.com/news/political-even-handedness'
        ),
        reference_impl='https://github.com/anthropics/political-neutrality-eval @ c5ed67908b56',
        resource_binding="judge",
        deviation_note='',
        fidelity_note='docs/fidelity/political_even_handedness.md',
    ),
    "PsycholinguisticNorms": MetricInfo(
        name="PsycholinguisticNorms",
        family="generated_text",
        access=('completions',),
        neutral_value=0.0,
        direction="signed",
        value_range=(float("-inf"), float("inf")),
        fidelity="adaptation",
        reference=(
            'BOLD: Dataset and Metrics for Measuring Biases in Open-Ended Language '
            'Generation, FAccT 2021 — https://arxiv.org/abs/2101.11718'
        ),
        reference_impl='https://github.com/amazon-science/bold @ 3ad652c773f5',
        resource_binding="lexicon",
        deviation_note=(
            "Averages caller-supplied norm values per dimension, which matches BOLD's "
            "aggregation, but does NOT apply the paper's rescaling (VAD to [-1,1] with 0 "
            'neutral). With the standard NRC-VAD file the output is on the original 1-9 scale '
            'where 5, not 0, is neutral - so the declared neutral_value does not apply to raw '
            'NRC-VAD input. BE5 emotion norms are not shipped. See '
            'docs/fidelity/bold_metrics.md.'
        ),
        fidelity_note='docs/fidelity/bold_metrics.md',
    ),
    "RealToxicityPrompts": MetricInfo(
        name="RealToxicityPrompts",
        family="prompt",
        access=('chat',),
        neutral_value=0.0,
        direction="higher_more_biased",
        value_range=(0.0, 1.0),
        fidelity="adaptation",
        reference=(
            'RealToxicityPrompts: Evaluating Neural Toxic Degeneration in Language Models, '
            'Findings of EMNLP 2020 — https://arxiv.org/abs/2009.11462'
        ),
        reference_impl='https://github.com/allenai/real-toxicity-prompts @ dd44ab77ed8b',
        resource_binding="classifier",
        deviation_note=(
            "The corrected metric requires Perspective API scoring or an explicitly named "
            'injected scorer adaptation; a substitute scorer changes the numbers. LiteLLM chat '
            'generation is also a modern adaptation of the paper\'s raw causal-LM path, and K is '
            "caller-chosen rather than fixed at the paper's 25. See "
            'docs/fidelity/toxicity_family.md.'
        ),
        fidelity_note='docs/fidelity/toxicity_family.md',
    ),
    "RegardScore": MetricInfo(
        name="RegardScore",
        family="generated_text",
        access=('completions',),
        neutral_value=0.0,
        direction="signed",
        value_range=(-1.0, 1.0),
        fidelity="adaptation",
        reference=(
            'The Woman Worked as a Babysitter: On Biases in Language Generation, EMNLP 2019 — '
            'https://arxiv.org/abs/1909.01326'
        ),
        reference_impl='https://github.com/ewsheng/nlg-bias @ 7f8d08ea4f33',
        resource_binding="classifier",
        deviation_note=(
            "Scores regard with sasha/regardv3, Sheng's published checkpoint, rather than her "
            "own regard1 3-BERT majority-vote ensemble that produced the paper's numbers - a "
            "different checkpoint trained on the v2 dataset with an added 'other' bucket. "
            'v0.1.1 defaulted to a SENTIMENT classifier, which was a mismatch: Sheng et al. '
            'Table 2 gives sentences where sentiment and regard labels have opposite signs. '
            'See docs/fidelity/regard_score.md.'
        ),
        fidelity_note='docs/fidelity/regard_score.md',
    ),
    "SEAT": MetricInfo(
        name="SEAT",
        family="embedding",
        access=('embeddings',),
        neutral_value=0.0,
        direction="signed",
        value_range=(float("-inf"), float("inf")),
        fidelity="faithful",
        reference=(
            'On Measuring Social Biases in Sentence Encoders, NAACL 2019 — '
            'https://arxiv.org/abs/1903.10561'
        ),
        reference_impl='https://github.com/W4ngatang/sent-bias @ e3559fb669ca',
        resource_binding="dataset",
        deviation_note='',
        fidelity_note='docs/fidelity/seat.md',
    ),
    "MeanScoreGap": MetricInfo(
        name="MeanScoreGap",
        family="generated_text",
        access=('completions',),
        neutral_value=0.0,
        direction="signed",
        value_range=(float("-inf"), float("inf")),
        fidelity="original",
        reference=(
            'BiasScope original. Inspired by Borkan et al. 2019, Nuanced Metrics for '
            'Measuring Unintended Bias, WWW 2019 companion — https://arxiv.org/abs/1903.04561'
        ),
        reference_impl='',
        resource_binding="classifier",
        deviation_note=(
            "BiasScope's own operationalization: the difference in mean classifier score "
            "between two groups, with Cohen's d. Borkan et al.'s five metrics are all "
            'threshold-agnostic and label-based (Subgroup/BPSN/BNSP AUC and the two Average '
            'Equality Gaps); a mean-score gap is none of them and needs no labels. Shipped as '
            'ScoreParity citing Borkan through v0.1.1; renamed in 0.2.0. See '
            'docs/fidelity/score_parity.md.'
        ),
        fidelity_note='docs/fidelity/score_parity.md',
    ),
    "SentenceBiasScore": MetricInfo(
        name="SentenceBiasScore",
        family="embedding",
        access=('embeddings',),
        neutral_value=0.0,
        direction="signed",
        value_range=(float("-inf"), float("inf")),
        fidelity="unaudited",
        reference=(
            'Dolci, Azzalini & Tanelli 2023, Data Science and Engineering 8(2), Springer — '
            'paper NOT LOCATED'
        ),
        reference_impl='',
        resource_binding="lexicon",
        deviation_note=(
            'Fidelity cannot be established: the source paper is paywalled and no preprint '
            'was found after the Section 4.0 search, so it has not been read. PLAN.md Section '
            '4.0 forbids assigning a status without reading the paper, and this is the one '
            'metric in the library where that was impossible. Do NOT cite this implementation '
            'as faithful to Dolci et al. See docs/fidelity/sentence_bias_score.md and '
            'REVIEW_LATER RL-029.'
        ),
        fidelity_note='docs/fidelity/sentence_bias_score.md',
    ),
    "SocialGroupSubstitution": MetricInfo(
        name="SocialGroupSubstitution",
        family="generated_text",
        access=('completions',),
        neutral_value=0.0,
        direction="higher_more_biased",
        value_range=(0.0, float("inf")),
        fidelity="adaptation",
        reference=(
            'Reducing Sentiment Bias in Language Models via Counterfactual Evaluation, '
            'Findings of EMNLP 2020 — https://arxiv.org/abs/1911.03064'
        ),
        reference_impl='',
        resource_binding="classifier",
        deviation_note=(
            "Mirrors the paper's Individual and Group Fairness structure but computes both as "
            'a range (max - min) across substituted values rather than the Wasserstein-1 '
            'distance between distributions; a range is set by the two extreme groups and '
            'ignores everything between them. Group disparity also compares subgroups to each '
            'other, whereas the paper compares each subgroup to the entire evaluation set. '
            'See docs/fidelity/huang_metrics.md and REVIEW_LATER RL-027.'
        ),
        fidelity_note='docs/fidelity/huang_metrics.md',
    ),
    "StereoSetMetric": MetricInfo(
        name="StereoSetMetric",
        family="prompt",
        access=('chat',),
        neutral_value=50.0,
        direction="higher_more_biased",
        value_range=(0.0, 100.0),
        fidelity="adaptation",
        reference=(
            'StereoSet: Measuring stereotypical bias in pretrained language models, ACL 2021 '
            '— https://arxiv.org/abs/2004.09456'
        ),
        reference_impl='https://github.com/moinnadeem/StereoSet @ ead7d086a64a',
        resource_binding="dataset",
        deviation_note=(
            'Two deviations from the likelihood protocol: the model makes a forced three-way '
            'A/B/C choice rather than having its likelihoods ranked pairwise, so lms and ss '
            "are analogues rather than the reference's pairwise counts; and scores are "
            'aggregated flat rather than averaged per target term. CAT/ICAT are the faithful '
            'likelihood version. See docs/fidelity/stereoset_family.md.'
        ),
        fidelity_note='docs/fidelity/stereoset_family.md',
    ),
    "StereotypeRuleHitRate": MetricInfo(
        name="StereotypeRuleHitRate",
        family="generated_text",
        access=('completions',),
        neutral_value=0.0,
        direction="higher_more_biased",
        value_range=(0.0, float("inf")),
        fidelity="original",
        reference=(
            'BiasScope original. The rule/window hit-rate matcher that shipped as '
            'StereotypicalAssociations through v0.1.1; no cited paper defines it.'
        ),
        reference_impl='',
        resource_binding="lexicon",
        deviation_note=(
            "BiasScope's own operationalization: how often user-supplied stereotype "
            'rules match a generation, by token window or regex. It is NOT HELM\'s '
            'StereotypicalAssociations, which is a mean TVD over target words. Renamed '
            'in 0.2.0. See docs/fidelity/stereotypical_associations.md.'
        ),
        fidelity_note='docs/fidelity/stereotypical_associations.md',
    ),
    "StereotypicalAssociations": MetricInfo(
        name="StereotypicalAssociations",
        family="generated_text",
        access=('completions',),
        neutral_value=0.0,
        direction="higher_more_biased",
        value_range=(0.0, 1.0),
        fidelity="adaptation",
        reference=(
            'Holistic Evaluation of Language Models, TMLR 2023 — '
            'https://arxiv.org/abs/2211.09110'
        ),
        reference_impl='https://github.com/stanford-crfm/helm @ 63754d05db6f',
        resource_binding="lexicon",
        deviation_note=(
            "Faithful to HELM's formula. One protocol deviation: HELM tokenises "
            'with NLTK word_tokenize; the default here is a regex word tokenizer so '
            'the core install stays light. Pass tokenizer=nltk.word_tokenize to '
            'match HELM exactly.'
        ),
        fidelity_note='docs/fidelity/stereotypical_associations.md',
    ),
    "TofNof": MetricInfo(
        name="TofNof",
        family="prompt",
        access=('chat',),
        neutral_value=0.0,
        direction="higher_more_biased",
        value_range=(0.0, 1.0),
        fidelity="adaptation",
        reference=(
            'Measuring Sycophancy of Language Models in Multi-turn Dialogues, arXiv preprint '
            '2025 — https://arxiv.org/abs/2505.23840'
        ),
        reference_impl='',
        resource_binding="judge",
        deviation_note=(
            'Turn-of-Flip and Number-of-Flip match eqs. 1-2. The deviation is the judge: Hong '
            'et al. obtain the binary stance-alignment labels from GPT-4o, and the label IS '
            'the measurement, so two TofNof numbers from different judges are not comparable. '
            'To be refactored onto the Judge abstraction so judge model and prompt version '
            'reach the protocol block (PLAN.md 7.1). See docs/fidelity/tofnof.md.'
        ),
        fidelity_note='docs/fidelity/tofnof.md',
    ),
    "ToxicityFraction": MetricInfo(
        name="ToxicityFraction",
        family="generated_text",
        access=('completions',),
        neutral_value=0.0,
        direction="higher_more_biased",
        value_range=(0.0, 1.0),
        fidelity="original",
        reference=(
            'BiasScope original. Inspired by Gehman et al. 2020 — '
            'https://arxiv.org/abs/2009.11462'
        ),
        reference_impl='https://github.com/allenai/real-toxicity-prompts @ dd44ab77ed8b',
        resource_binding="classifier",
        deviation_note=(
            "BiasScope's own operationalization: the mean over prompts of the fraction of the "
            'K generations that are toxic. Gehman et al. define exactly two metrics for '
            'prompted generation - expected maximum toxicity (EMT) and the probability of at '
            'least one toxic span (ToxicityProbability) - and this mean-of-fractions is '
            'neither. A model with 1 toxic generation in 25 scores 1.0 on ToxicityProbability '
            'and 0.04 here. See docs/fidelity/toxicity_family.md.'
        ),
        fidelity_note='docs/fidelity/toxicity_family.md',
    ),
    "ToxicityProbability": MetricInfo(
        name="ToxicityProbability",
        family="generated_text",
        access=('completions',),
        neutral_value=0.0,
        direction="higher_more_biased",
        value_range=(0.0, 1.0),
        fidelity="faithful",
        reference=(
            'RealToxicityPrompts: Evaluating Neural Toxic Degeneration in Language Models, '
            'Findings of EMNLP 2020 — https://arxiv.org/abs/2009.11462'
        ),
        reference_impl='https://github.com/allenai/real-toxicity-prompts @ dd44ab77ed8b',
        resource_binding="classifier",
        deviation_note='',
        fidelity_note='docs/fidelity/toxicity_family.md',
    ),
    "TrustLLMStereotypeRecognition": MetricInfo(
        name="TrustLLMStereotypeRecognition",
        family="prompt",
        access=('chat', 'completions'),
        neutral_value=1.0,
        direction="lower_more_biased",
        value_range=(0.0, 1.0),
        fidelity="faithful",
        reference=(
            'TrustLLM: Trustworthiness in Large Language Models, ICML 2024 — '
            'https://arxiv.org/abs/2401.05561 (fairness, stereotype recognition)'
        ),
        reference_impl='https://github.com/HowieHwong/TrustLLM @ 4b864211f326',
        resource_binding="dataset",
        deviation_note=(
            'Reports accuracy directly, as the benchmark does. 1.0 is the ideal, '
            'not 0.0, so `neutral_value` is 1.0 and profile plots measure '
            'deviation downward from it.'
        ),
        fidelity_note='docs/fidelity/trustllm.md',
    ),
    "TrustLLMStereotypeAgreement": MetricInfo(
        name="TrustLLMStereotypeAgreement",
        family="prompt",
        access=('chat', 'completions'),
        neutral_value=0.0,
        direction="higher_more_biased",
        value_range=(0.0, 1.0),
        fidelity="faithful",
        reference=(
            'TrustLLM: Trustworthiness in Large Language Models, ICML 2024 — '
            'https://arxiv.org/abs/2401.05561 (fairness, stereotype agreement)'
        ),
        reference_impl='https://github.com/HowieHwong/TrustLLM @ 4b864211f326',
        resource_binding="dataset",
        deviation_note=(
            'The denominator is every response, so an unclear answer counts '
            'against agreement rather than being excluded — unlike '
            'TrustLLMStereotypeRecognition, which drops its invalid answers. '
            "Both reproduce the benchmark's own code."
        ),
        fidelity_note='docs/fidelity/trustllm.md',
    ),
    "TrustLLMDisparagement": MetricInfo(
        name="TrustLLMDisparagement",
        family="prompt",
        access=('chat', 'completions'),
        neutral_value=1.0,
        direction="lower_more_biased",
        value_range=(0.0, 1.0),
        fidelity="faithful",
        reference=(
            'TrustLLM: Trustworthiness in Large Language Models, ICML 2024 — '
            'https://arxiv.org/abs/2401.05561 (fairness, disparagement)'
        ),
        reference_impl='https://github.com/HowieHwong/TrustLLM @ 4b864211f326',
        resource_binding="dataset",
        deviation_note=(
            'Reports the smallest p-value across the tested attributes rather '
            'than a per-attribute dict, so a mean cannot let an independent '
            'attribute mask a dependent one. Chi-square is computed in-library '
            'rather than via scipy.stats.chi2_contingency; both are the '
            'uncorrected Pearson statistic for these table shapes.'
        ),
        fidelity_note='docs/fidelity/trustllm.md',
    ),
    "TrustLLMPreference": MetricInfo(
        name="TrustLLMPreference",
        family="prompt",
        access=('chat', 'completions'),
        neutral_value=1.0,
        direction="lower_more_biased",
        value_range=(0.0, 1.0),
        fidelity="faithful",
        reference=(
            'TrustLLM: Trustworthiness in Large Language Models, ICML 2024 — '
            'https://arxiv.org/abs/2401.05561 (fairness, preference)'
        ),
        reference_impl='https://github.com/HowieHwong/TrustLLM @ 4b864211f326',
        resource_binding="classifier",
        deviation_note=(
            'The refusal classifier is a caller-supplied callable rather than a '
            'hardwired LibrAI/longformer-action-ro, so the classifier version '
            'is recorded in the protocol instead of being implicit.'
        ),
        fidelity_note='docs/fidelity/trustllm.md',
    ),
    "TruthfulQA": MetricInfo(
        name="TruthfulQA",
        family="prompt",
        access=('logits',),
        neutral_value=1.0,
        direction="lower_more_biased",
        value_range=(0.0, 1.0),
        fidelity="adaptation",
        reference=(
            'TruthfulQA: Measuring How Models Mimic Human Falsehoods, ACL 2022 — '
            'https://arxiv.org/abs/2109.07958'
        ),
        reference_impl='https://github.com/sylinrl/TruthfulQA @ 8b48f682acc3a71cd04e32e46e6da40ddb1a5860 (preserved v0)',
        resource_binding="dataset",
        deviation_note=(
            'Supports source-faithful aggregation of precomputed GPT-judge/GPT-info '
            'probabilities and original MC1/MC2 log-likelihoods. It does not ship the '
            'unavailable fine-tuned judge artifacts; private research helpers collect local '
            'paper-era v0 likelihoods without changing the normal metric API. '
            'Truthfulness is not social bias; it remains excluded from bias profiles by default.'
        ),
        fidelity_note='docs/fidelity/truthfulqa.md',
    ),
    "UnQoverMetric": MetricInfo(
        name="UnQoverMetric",
        family="prompt",
        access=('chat',),
        neutral_value=0.0,
        direction="higher_more_biased",
        value_range=(0.0, 1.0),
        fidelity="adaptation",
        reference=(
            'UnQovering Stereotyping Biases via Underspecified Questions, Findings of EMNLP '
            '2020 — https://arxiv.org/abs/2010.02428'
        ),
        reference_impl='https://github.com/allenai/unqover @ 3e47969b78ac',
        resource_binding="dataset",
        deviation_note=(
            'The four-variant positional/negation design and the gamma, mu (eq. 8) and eta '
            '(eq. 9) aggregations are faithful. The deviation is access mode: Li et al. score '
            "QA answer spans and masked-LM fills, while this reads a chat model's token "
            'logprobs for the A/B option letters. Letter probability is not subject '
            'confidence and is sensitive to prompt formatting - the class of confound UnQover '
            'exists to remove. A masked-LM path would be faithful. See '
            'docs/fidelity/unqover.md.'
        ),
        fidelity_note='docs/fidelity/unqover.md',
    ),
    "ImplicitAssociationTest": MetricInfo(
        name="ImplicitAssociationTest",
        family="prompt",
        access=('chat', 'completions'),
        neutral_value=0.0,
        direction="signed",
        value_range=(-1.0, 1.0),
        fidelity="faithful",
        reference=(
            'Explicitly unbiased large language models still form biased '
            'associations, PNAS 2025 — https://arxiv.org/abs/2402.04105'
        ),
        reference_impl=(
            'https://github.com/baixuechunzi/llm-implicit-bias @ 0d2772e8eb21'
        ),
        resource_binding="dataset",
        deviation_note=(
            "Uses the reference code's 0.01 divide-by-zero guard in both "
            'denominators (analysis/clean.ipynb, `d_score`), which the paper '
            'omits; pass epsilon=0.0 for the published formula exactly. Both '
            "reproduce the paper's worked examples."
        ),
        fidelity_note='docs/fidelity/implicit_association.md',
    ),
    "LLMDecisionBias": MetricInfo(
        name="LLMDecisionBias",
        family="prompt",
        access=('chat', 'completions'),
        neutral_value=0.5,
        direction="signed",
        value_range=(0.0, 1.0),
        fidelity="faithful",
        reference=(
            'Explicitly unbiased large language models still form biased '
            'associations, PNAS 2025, Sec. 2.2 — https://arxiv.org/abs/2402.04105'
        ),
        reference_impl=(
            'https://github.com/baixuechunzi/llm-implicit-bias @ 0d2772e8eb21'
        ),
        resource_binding="judge",
        deviation_note='',
        fidelity_note='docs/fidelity/implicit_association.md',
    ),
    "WinoBias": MetricInfo(
        name="WinoBias",
        family="prompt",
        access=('chat',),
        neutral_value=0.0,
        direction="signed",
        value_range=(-1.0, 1.0),
        fidelity="faithful",
        reference=(
            'Gender Bias in Coreference Resolution: Evaluation and Debiasing Methods, NAACL '
            '2018 — https://arxiv.org/abs/1804.06876'
        ),
        reference_impl='https://github.com/uclanlp/corefBias @ 0bce984dd081',
        resource_binding="dataset",
        deviation_note='',
        fidelity_note='docs/fidelity/winobias.md',
    ),
    "WEAT": MetricInfo(
        name="WEAT",
        family="embedding",
        access=('embeddings',),
        neutral_value=0.0,
        direction="signed",
        value_range=(float("-inf"), float("inf")),
        fidelity="faithful",
        reference=(
            'Semantics derived automatically from language corpora contain '
            'human-like biases, Science 356(6334) — '
            'https://arxiv.org/abs/1608.07187'
        ),
        reference_impl=(
            'https://github.com/W4ngatang/sent-bias '
            '@ e3559fb669ca'
        ),
        resource_binding="language_agnostic",
        deviation_note='',
        fidelity_note='docs/fidelity/weat.md',
    ),
}


def attach() -> int:
    """Attach each `MetricInfo` to its metric class and register it.

    Called once from `bias_scope/__init__.py`. Done here, explicitly, rather
    than by a decorator or an import side effect on each metric module —
    PLAN.md Section 1 rules out plugin registries built that way, and this
    keeps the whole mapping readable in one file.

    Returns the number of classes successfully attached. Classes that are
    absent because an optional extra is missing are skipped silently; the
    stubs `bias_scope/__init__.py` installs are not metrics.
    """
    import importlib

    from bias_scope.base import BiasMetric
    from bias_scope.metadata import register

    modules = (
        "bias_scope.embeddings_based",
        "bias_scope.probability_based",
        "bias_scope.generated_text_based",
        "bias_scope.prompts_based",
    )

    attached = 0
    for module_name in modules:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        for class_name, info in METRIC_INFO.items():
            cls = getattr(module, class_name, None)
            if cls is None or not isinstance(cls, type):
                continue
            if not issubclass(cls, BiasMetric):
                continue
            cls.info = info
            register(class_name, info)
            attached += 1
    return attached
