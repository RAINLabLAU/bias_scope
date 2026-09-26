# PoliticalEvenHandedness

<!-- metric-card:start -->
| | |
|---|---|
| Family | prompt |
| Model access | `chat` |
| Neutral value | 0 |
| Direction | higher means more biased |
| Range | 0 to 1 |
| Languages | en |
| Fidelity | **faithful**: same formula and protocol as the cited paper. [Audit note](../../fidelity/political_even_handedness.md). |
| Source | Anthropic 2025, Measuring political bias in Claude — https://www.anthropic.com/news/political-even-handedness |
| Reference code | https://github.com/anthropics/political-neutrality-eval @ c5ed67908b56 |
<!-- metric-card:end -->


Whether a model treats paired prompts with opposing political stances the same way (Anthropic, 2025). Read the three reported rates together.

::: bias_scope.prompts_based.political_even_handedness.PoliticalEvenHandedness

## Example

```python
--8<-- "examples/prompts_based/political_even_handedness.py"
```
