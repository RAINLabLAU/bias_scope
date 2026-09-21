# BOLD

**Cited source:** Dhamala et al. (2021), *BOLD: Dataset and Metrics for
Measuring Biases in Open-Ended Language Generation*, FAccT 2021. Official
dataset repository: [amazon-science/bold](https://github.com/amazon-science/bold)
at `3ad652c773f5d1e30d5f6f61657ed934d768ecad`.

## Verdict: adaptation

BOLD is a 23,679-prompt, 43-group dataset and a suite of five evaluation
families: sentiment, toxicity, regard, psycholinguistic norms, and gender
polarity. It is not one scalar metric. The paper reports distributions,
proportions, counts, ratios, and selected group differences.

The public `BOLD` class is a BiasScope orchestration adaptation. It accepts a
caller-provided generation function and scalar scorers. Its generic group means
and max-minus-min `gaps` are not paper-defined BOLD outputs, and it deliberately
does not invent a universal BOLD score.

## Private reproduction protocol

`_bold_reproduction.py` is private infrastructure. It can load external pinned
prompt artifacts, record SHA-256 values, validate Table 1 counts, and apply
explicit caller-supplied `Person`/`XYZ` substitutions. It implements the stated
paper formulas for:

- VADER thresholded sentiment proportions;
- six-label toxicity OR aggregation over supplied labels;
- BOLD-scope regard label proportions for male/female and European/African
  American groups;
- rescaled, weighted VAD/BE5 norms with caller-provided POS tags and lexicon;
- fixed-list unigram and `she - he` embedding gender polarity.

These helpers are **paper-formula faithful** when supplied compatible inputs.
They are not automatically exact historical reproduction. The paper-era BERT
toxicity checkpoint, historical regard artifact, induced FastText lexicon,
POS-tagger choice, and complete BERT/GPT-2/CTRL generation provenance are not
all recoverable from the pinned prompt repository. No Perspective API,
`sasha/regardv3`, arbitrary GPT-2 checkpoint, or modern embedding is silently
substituted for those dependencies.

## Reproduction levels

1. **Official data artifact parity:** external official prompt JSONs can be
   loaded, schema checked, count validated, and hashed.
2. **Paper-formula faithful:** private aggregators implement stated calculations
   over caller-supplied historical-shaped labels and resources.
3. **Historical local reconstruction:** not yet implemented. The paper names
   `bert-large-cased`, but does not fully pin its cited generator runtime;
   GPT-2 and CTRL checkpoint identities remain ambiguous.
4. **Exact end-to-end paper reproduction:** currently unavailable.
