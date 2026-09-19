# BOLD

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

See [the offline BOLD example](../../../examples/prompts_based/bold.py) for
the public adaptation. It does not claim paper-table reproduction.
