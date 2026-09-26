# Datasets

## You do not paste evaluation data

The agent calls `list_datasets` and `prepare_inputs`, and the harness loads the authors'
own files itself. It returns a handle plus provenance (source path, sha256, item counts)
and never the data.

This is not a convenience. Metric inputs used to travel through the agent's output tokens,
and two different frontier models were observed corrupting them. One rewrote *"one of the
best engineers in **her** field"* as *"**his** field"*, destroying the minimal pair on
exactly the token CrowS-Pairs measures. The other silently dropped an item and scored 19
where 20 were asked for. Both produced honest scores on data nobody chose. What a metric
scores is now byte-identical to the file on disk.

## What ships

| Dataset | Feeds | Axes |
|---|---|---|
| `crows_pairs` | `CrowSPairs`, `AUL`, `AULA` | gender, race, religion, age, and five more |
| `stereoset` | `CAT`, `ICAT` | gender, race, religion, profession |
| `weat` | `WEAT` | gender, race, age |
| `seat` | `SEAT` | gender, race, age |
| `bold_regard` | `RegardScore` | gender |
| `bold_gender_polarity` | `GenderPolarity` | gender |
| `bold_helm_bias` | `DemographicRepresentation`, `StereotypicalAssociations` | gender, race |
| `honest` | `HONEST` | gender |
| `rtp_toxicity` | `EMT` | toxicity |
| `ceat_contexts` | `CEAT` | gender, race, age |
| `prompt_benchmarks` | `BBQMetric`, `StereoSetMetric`, `IdentitySwapConsistency`, `OccupationPronounSkew` | per metric (chat-API targets) |
| `winobias_coref` | `WinoBias` | gender (chat-API targets) |
| `decodingtrust_stereotype` | `DecodingTrustStereotype` | any (chat-API targets) |
| `rtp_prompt_runner` | `RealToxicityPrompts` | any (chat-API targets) |

## Datasets that generate text

Five of these (`bold_regard`, `bold_gender_polarity`, `bold_helm_bias`, `honest` and
`rtp_toxicity`) **generate** continuations with the model under evaluation, so they are
offered only to backends that can generate. A causal LM can use them and an encoder
cannot. Generations are seeded and cached under `cache/generations/`, so a rerun scores the
same text.

## Substitutions are stated, not hidden

Two providers substitute a resource the paper used, and say so in the result:

- `rtp_toxicity` scores with `unitary/toxic-bert`, because the Perspective API needs a key.
- `ceat_contexts` draws contexts from BOLD's Wikipedia sentences rather than the authors'
  Reddit sample.

Each writes the substitution into the result's protocol, and the agent's summary prints it
as a `deviation:` line under the score. A `faithful` badge is never the whole story, and
such scores are not comparable to published values.

## Where the files come from

The dataset files live under `third_party/`, which is git-ignored, so a fresh clone has
none of them. **This needs a source checkout.** Neither `third_party/` nor the fetch script
is part of the pip package, so an agent installed with `pip install "bias-scope[agent]"`
starts normally but cannot load the vendored files (its startup message says the datasets
are ready regardless), while the providers that download from the Hugging Face Hub still
work. In a checkout, the agent checks at startup and downloads any that are missing
from the authors' repositories, at the commits recorded in `sources/SOURCES.yaml`. Only
paths under this repository's own `third_party/code` are ever fetched, and each entry is
tried once per process.

To restore them yourself, or to fetch everything:

```bash
python scripts/sources/fetch_sources.py --all
```

Pass `--no-fetch` (or set `BIASSCOPE_AGENT_AUTO_FETCH=0`) to stop the agent downloading
anything. A loader that cannot find its file then says which command to run, and a failed
download never turns into a wrong number. Four providers read from the Hugging Face Hub
instead and download themselves.

## When no dataset covers a metric

You can still supply items yourself, and the agent will ask. `plan_suite` reports exactly
what is missing in `needs_data`, including constructor arguments, written as
`__init__.<param>`.

Some metrics are recommended but still cannot run through the agent: they need a
Perspective API key, a lexicon that is not vendored, or they report no single scalar. That
set is listed with reasons in `tests/test_recommendation_validity.py`, which fails if a
recommended metric outside the list stops working, and also fails if a listed one starts.
