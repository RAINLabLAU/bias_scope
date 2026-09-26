# FirstPersonFairness

<!-- metric-card:start -->
| | |
|---|---|
| Family | prompt |
| Model access | `chat` |
| Neutral value | 0 |
| Direction | signed; 0 is neutral |
| Range | -1 to 1 |
| Languages | en |
| Fidelity | **adaptation**: same comparison as the cited paper, but a different access mode or scoring path that can change the numbers. The estimator H = E[h_F - h_R] is the paper's exactly, including the swapped second judging pass and the identical-response rule. [Audit note](../../fidelity/first_person_fairness.md). |
| Source | First-Person Fairness in Chatbots, Eloundou et al. 2024 (OpenAI), Sec. 3.3 and Figure 3 — https://arxiv.org/abs/2410.19803 |
<!-- metric-card:end -->


Whether a chatbot's responses carry a harmful stereotype when only the user's name changes (Eloundou et al., 2024).

::: bias_scope.prompts_based.first_person_fairness.FirstPersonFairness

## Example

```python
--8<-- "examples/prompts_based/first_person_fairness.py"
```
