# TofNof

::: bias_scope.prompts_based.tof_nof.TofNof

## Example

```python
# ──────────────────────────────────────────────────────────────
# TofNof — Turn-of-First-Flip / Number-of-Flips
#
# Detects sycophancy — the model's tendency to abandon its stated
# position when the user pushes back.  The model is assigned to
# argue FOR a topic, then receives repeated disagreement.  A judge
# model classifies each response as "flip" or "hold".
#
# Returns:
#   flip_rate — fraction of topics where the model flipped (0 = no sycophancy)
#   avg_tof   — mean turn of first flip (higher = model held longer)
#   avg_nof   — mean total flips per topic (lower = less sycophantic)
#
# NOTE: Requires valid LLM API keys and internet access.
# ──────────────────────────────────────────────────────────────

from bias_scope.prompts_based import TofNof

# --- Initialize ---
metric = TofNof(
    model_name="huggingface/meta-llama/Meta-Llama-3-8B-Instruct",       # the debater model
    api_key="your-api-key-here",
    judge_model="huggingface/meta-llama/Meta-Llama-3-8B-Instruct",       # the judge (defaults to model_name if None)
    judge_api_key="your-api-key-here",  # defaults to api_key if None
)

# --- Evaluate ---
# Loads debate topics from Anthropic/model-written-evals dataset
result = metric.evaluate(
    num_topics=1,       # test 1 topic (None = all in dataset)
    num_turns=2,        # 2 rounds of pushback per topic (1-10)
    subset="sycophancy_on_nlp_survey",  # valid: sycophancy_on_nlp_survey,
                                         #        sycophancy_on_philpapers2020,
                                         #        sycophancy_on_political_typology_quiz
)


print(f"Flip rate:        {result['flip_rate']:.2%}")
print(f"Avg first flip:   {result['avg_tof']:.1f} (turn)")
print(f"Avg total flips:  {result['avg_nof']:.1f}")
print()
print("Per-topic breakdown:")
for topic, info in result["per_topic"].items():
    status = "FLIPPED" if info["flipped"] else "HELD"
    tof = info["tof"] if info["tof"] is not None else "—"
    print(f"  [{status:>7}] tof={tof}, nof={info['nof']}")
    print(f"           {topic[:80]}...")
print()
print("Interpretation:")
print("  flip_rate = 0%   → model never abandoned its position (no sycophancy)")
print("  flip_rate = 100% → model flipped on every topic (fully sycophantic)")
print("  avg_tof close to num_turns → model held out for many rounds")
print("  avg_nof close to 0 → few total reversals")

```
