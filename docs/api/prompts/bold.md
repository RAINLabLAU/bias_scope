# BOLD

<!-- metric-card:start -->
| | |
|---|---|
| Family | prompt |
| Model access | `chat` |
| Neutral value | 0 |
| Direction | higher means more biased |
| Range | 0 or more |
| Languages | en |
| Fidelity | **adaptation**: same comparison as the cited paper, but a different access mode or scoring path that can change the numbers. BOLD is a dataset plus five heterogeneous evaluation families, not one scalar. [Audit note](../../fidelity/bold.md). |
| Source | BOLD: Dataset and Metrics for Measuring Biases in Open-Ended Language Generation, FAccT 2021 — https://arxiv.org/abs/2101.11718 |
| Reference code | https://github.com/amazon-science/bold @ 3ad652c773f5 |
<!-- metric-card:end -->


::: bias_scope.prompts_based.bold.BOLD

`BOLD` is a public BiasScope orchestration adaptation. It accepts prompts, a
caller-selected generation function, and caller-selected scalar scorers. Its
generic score means and `gaps` are BiasScope diagnostics, not BOLD paper
outputs, and there is no universal BOLD score.

The private BOLD reproduction protocol uses external, opt-in official prompt
artifacts and paper-shaped labels/resources. It intentionally does not replace
unavailable historical toxicity, regard, lexicon, or generation artifacts with
modern services or models.

## Example

See [the offline BOLD example](https://github.com/RAINLabLAU/bias_scope/blob/main/examples/prompts_based/bold.py) for
the public adaptation. It does not claim paper-table reproduction.
