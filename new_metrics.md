
# bias_scope — Generated Text-Based Metrics (Category 3) Specification

## Goal

Add a new metric category to `bias_scope` for measuring bias in **generated text** (outputs), following the same design philosophy as existing embedding- and probability-based metrics:
- clean modular design
- consistent class interface: `evaluate(...)`
- strong validation + clear errors
- testable, deterministic toy tests
- JSON-serializable returns
- minimal dependencies

This spec covers implementation + **comprehensive tests** for:
1. Social Group Substitution
2. Co-Occurrence Bias Score
3. Demographic Representation
4. Stereotypical Associations
5. Marked Persons

---

## Project structure

Create a new package:

```

bias_scope/
generated_text_based/
**init**.py
_helpers.py
social_group_substitution.py
cooccurrence_bias_score.py
demographic_representation.py
stereotypical_associations.py
marked_persons.py

```

Tests:

```

tests/
generated_text_based/
test_social_group_substitution.py
test_cooccurrence_bias_score.py
test_demographic_representation.py
test_stereotypical_associations.py
test_marked_persons.py

```

Update exports:
- `bias_scope/generated_text_based/__init__.py`
- `bias_scope/__init__.py`

---

## Base class changes (`bias_scope/base.py`)

Add:

### `class GeneratedTextMetric(BiasMetric)`
- Category: `"generated_text"`
- Provide shared validation helpers for:
  - sequences of strings
  - callable checks
  - numeric scalar checks (finite float)

#### Required behavior
- `category` property returns `"generated_text"`.
- Validation helpers should raise `ValueError` or `TypeError` with helpful messages.
- Keep consistent patterns with `EmbeddingMetric` and `ProbabilityMetric`.

#### Helper methods (recommended)
- `_validate_texts(texts: Sequence[str], name: str) -> list[str]`
  - must be sequence (not a single string)
  - non-empty
  - all elements are `str`
  - strip not required, but if empty strings exist: raise `ValueError`
- `_validate_callable(fn, name: str) -> None`
- `_validate_finite_float(x, name: str) -> float`
  - cast to float and ensure finite (not NaN/Inf)

---

## Shared helpers (`bias_scope/generated_text_based/_helpers.py`)

### Tokenization
Add a simple tokenizer:
- default: lowercase
- split on non-alphanumeric boundaries
- keep apostrophes inside tokens optional (document)
- MUST be deterministic and dependency-free

Provide:
- `tokenize(text: str) -> list[str]`

### Lexicon utilities
- `normalize_lexicon(lex: Sequence[str]) -> set[str]`
- `count_lexicon_mentions(tokens: list[str], lex: set[str]) -> int`
- `find_token_positions(tokens: list[str], lex: set[str]) -> list[int]`

### Windows / co-occurrence
- `count_cooccurrence_in_window(tokens, anchor_positions, target_lex, window_size) -> Counter[str]`
  - counts occurrences of target tokens within ±window around anchor positions
  - MUST avoid double-counting the same token position multiple times when anchors overlap:
    - define behavior explicitly: if a token position is within window of multiple anchors, count it once per document OR once per anchor.
    - Choose one and document. Recommended for stability: **count once per anchor** (simpler, aligns with “opportunities” scaling), but then tests must match.
  - Keep behavior consistent across metrics.

### Numeric stability
- Provide a small epsilon constant for logs where needed
- Ensure all returned floats are Python floats (not numpy scalars)

---

# Metric 1 — Social Group Substitution

## Research grounding
Use counterfactual evaluation framing:
- Huang et al. (2020), counterfactual fairness in sentiment bias (ACL/Findings/EMNLP).
Implementation is generalized to any score function (sentiment/toxicity/regard/etc).

## Definition

Inputs:
- `prompts: Sequence[str]`
  - each prompt is a template containing placeholders like `{group}` `{religion}`.
- `substitutions: Mapping[str, Sequence[str]]`
  - keys are placeholder names (without braces)
  - values are candidate substitution strings for that placeholder
- `generate_fn`
  - callable generating text from prompt(s)
  - support:
    - `Callable[[str], str]` (single)
    - `Callable[[Sequence[str]], Sequence[str]]` (batch)
- `score_fn: Callable[[str], float]`
  - maps generated text to scalar; must be finite float

### API
```

evaluate(
prompts,
substitutions,
generate_fn,
score_fn,
*,
num_samples: int = 1,
decoding_kwargs: Optional[dict] = None,
aggregation: Literal["mean","median"] = "mean",
) -> dict

```

### Generation protocol
For each prompt template `p_i`, for each placeholder key `k`, for each substitution value `a` in substitutions[k]:
- form counterfactual prompt `p_i(k=a)`
- generate `num_samples` outputs with the same decoding kwargs
- score each output with `score_fn`
- aggregate scores per (i,k,a) across samples using aggregation

### Metrics
Let `S_{i,k,a}` be the aggregated score for prompt i, placeholder k, substitution value a.

#### Individual counterfactual unfairness (per prompt i, placeholder k)
Average pairwise absolute difference across substitution values:
```

IF_{i,k} = mean_{a != a'} | S_{i,k,a} - S_{i,k,a'} |

```
- If fewer than 2 values, raise ValueError.

#### Overall individual unfairness
```

IF_overall = mean_{i,k} IF_{i,k}

```

#### Per-group mean score
For each (k,a):
```

GroupMean_{k,a} = mean_i S_{i,k,a}

```

#### Group disparity (range)
For each placeholder k:
```

GD_k = max_a GroupMean_{k,a} - min_a GroupMean_{k,a}

```
Overall:
- either return per-k and also `GD_overall = mean_k GD_k`

### Return schema
Must be JSON-serializable and include:

```

{
"metric": "SocialGroupSubstitution",
"category": "generated_text",
"num_prompts": int,
"num_samples": int,
"placeholders": [str],
"aggregation": "mean"|"median",

"scores": {
"<placeholder>": {
"<value>": [float per prompt i]   # length = num_prompts
}
},

"individual_unfairness": {
"<placeholder>": [float per prompt i]  # IF_{i,k}
},
"individual_unfairness_overall": float,

"group_means": {
"<placeholder>": { "<value>": float }
},
"group_disparity": {
"<placeholder>": float,
"_overall": float
},

"metadata": {
"supports_batched_generate_fn": bool,
"decoding_kwargs_used": dict
}
}

```

### Validation requirements
- prompts: non-empty list[str]
- substitutions: non-empty mapping[str -> non-empty list[str]]
- placeholder keys must appear in prompts at least once OR allow but warn? Choose strict:
  - Recommended strict: if a key never appears in any prompt, raise ValueError.
- all prompts must be format-able with substitutions: validate by attempting `.format(**{k: v0})`
- generate_fn callable; score_fn callable
- num_samples >= 1
- aggregation in {"mean","median"}
- score_fn output finite float

### Tests (must cover all cases)
At minimum:

1. **Happy path (single placeholder, 2 values, deterministic)**
   - generate_fn returns f"OUT:{prompt}"
   - score_fn: returns 1.0 if contains "A", else 3.0, etc
   - assert exact IF and GD values

2. **Multiple placeholders**
   - prompts include `{group}` and `{role}`
   - substitutions for both
   - ensure scores collected per placeholder independently

3. **Batch generate_fn support**
   - provide generate_fn that accepts list[str] and returns list[str]
   - confirm code path used and output correct

4. **Non-batched generate_fn**
   - accepts str only
   - still works

5. **num_samples > 1 aggregation mean**
   - generate_fn returns alternating outputs based on call count
   - score_fn maps outputs to known numbers
   - assert aggregated mean

6. **aggregation median**
   - confirm median calculation correct

7. **Validation errors**
   - empty prompts -> ValueError
   - prompts provided as a single string -> TypeError
   - substitutions empty -> ValueError
   - substitutions value list has <2 values -> ValueError
   - placeholder key not in any prompt -> ValueError
   - invalid prompt formatting -> ValueError
   - generate_fn not callable -> TypeError
   - score_fn not callable -> TypeError
   - score_fn returns NaN/Inf -> ValueError

8. **Determinism**
   - same inputs should yield identical outputs for deterministic functions

---

# Metric 2 — Co-Occurrence Bias Score

## Research grounding
Bordia & Bowman (2019): co-occurrence bias in word-level LMs, generalized to generated text.

## Definition
Count association of neutral words with group lexicons within a context window.

### API
```

evaluate(
generations: Sequence[str],
group_lexicons: Mapping[str, Sequence[str]],
neutral_vocab: Optional[Sequence[str]] = None,
*,
window_size: int = 10,
smoothing: float = 1.0,
return_top_k: int = 50,
multi_group_mode: Literal["pairwise","vs_mean"] = "pairwise",
) -> dict

```

### Tokenization
Use shared `tokenize`.

### Counting
Let groups be G.
For each generation:
- tokens = tokenize(text)
- for each group g:
  - anchor positions = positions of tokens in lexicon(g)
  - count neutral-word co-occurrences within ±window_size of each anchor

Define:
- `c[w,g]` = total co-occurrence count between word w and group g anchors across corpus
- `C[g]` = total number of group anchor occurrences across corpus (sum of anchor positions)

If `neutral_vocab` is None:
- use all tokens not in any group lexicon (or all tokens except group tokens), but document clearly
- Recommended: **neutral_vocab = all tokens excluding any group lexicon tokens**.

### Score
For 2 groups g1,g2:
```

score(w; g1,g2) =
log( (c[w,g1] + s) / (C[g1] + s) )

* log( (c[w,g2] + s) / (C[g2] + s) )

```

For >2 groups:
- `pairwise`: compute for every unordered pair (g_i, g_j)
- `vs_mean`: score(w,g) = log P(w|g) - mean_{g'!=g} log P(w|g')

Return must include whichever mode used.

### Return schema
```

{
"metric": "CoOccurrenceBiasScore",
"category": "generated_text",
"window_size": int,
"smoothing": float,
"groups": [str],
"vocab_size": int,

"counts": {
"group_anchors": { "<g>": int },
"cooccurrence": { "<g>": { "<w>": int } }
},

"scores": {
"mode": "pairwise"|"vs_mean",
"pairwise": {
"<g1>|<g2>": { "<w>": float }
},
"vs_mean": {
"<g>": { "<w>": float }
}
},

"summary": {
"mean_abs_score": float,
"top_terms": {
"<g1>|<g2>": {
"most_associated_with_g1": [ [w, score], ... ],
"most_associated_with_g2": [ [w, score], ... ]
}
}
}
}

```

### Validation
- generations: non-empty list[str]
- group_lexicons: >= 2 groups; each lexicon non-empty
- window_size >= 1
- smoothing > 0
- return_top_k >= 1
- multi_group_mode allowed

### Tests (must cover all cases)
1. **Happy path 2 groups, hand-computable counts**
   - small texts; compute c and C exactly; assert scores exactly

2. **neutral_vocab provided**
   - restrict scoring to provided list; assert others excluded

3. **neutral_vocab None uses derived vocab**
   - ensure group lexicon tokens excluded from neutral set

4. **window_size effect**
   - same corpus with window=1 vs window=10 produce different counts

5. **smoothing effect**
   - smoothing=1 vs smoothing=0.5 changes result; assert expected numeric value

6. **>2 groups pairwise mode**
   - assert keys and scores produced for all pairs

7. **>2 groups vs_mean mode**
   - assert vs_mean output shape correct

8. **Edge cases**
   - group anchors exist but no neutral words nearby -> counts zero; scores must still be finite due to smoothing

9. **Validation errors**
   - <2 groups -> ValueError
   - empty lexicon -> ValueError
   - empty generations -> ValueError
   - window_size invalid -> ValueError
   - smoothing <=0 -> ValueError
   - return_top_k invalid -> ValueError

---

# Metric 3 — Demographic Representation

## Research grounding
Representation and diversity auditing of demographic presence (inspired by Lahoti et al., 2023), generalized to lexicon-based detection.

## API
```

evaluate(
generations: Sequence[str],
group_lexicons: Mapping[str, Sequence[str]],
*,
normalize: Literal["mentions","tokens"] = "mentions",
compare_to: Optional[Mapping[str, float]] = None,
smoothing: float = 1e-12
) -> dict

```

### Counting
For each group g:
- count mentions = sum over generations of count of tokens in lexicon(g)

Normalization modes:
- `"mentions"`: p(g)=count(g)/sum_g count(g)
  - If total mentions == 0: raise ValueError (or return zeros + warning). Choose strict: raise ValueError.
- `"tokens"`: p(g)=count(g)/total_tokens_corpus
  - total_tokens must be >0 else ValueError.

### Diversity metrics
For distribution p over K groups:
- entropy: `H = - sum_g p(g) * log(p(g)+eps)`
- normalized entropy: `H / log(K)` if K>1 else 0.0
- gini impurity: `1 - sum_g p(g)^2`

### Reference comparison (optional)
Given `compare_to` q:
- validate q has same group keys; sums ~ 1 (allow small tolerance)
- compute:
  - KL(p||q) with smoothing
  - Jensen-Shannon divergence JS(p,q)

### Return schema
```

{
"metric": "DemographicRepresentation",
"category": "generated_text",
"groups": [str],
"counts": { "<g>": int },
"total_mentions": int,
"distribution": { "<g>": float },
"diversity": {
"entropy": float,
"normalized_entropy": float,
"gini_impurity": float
},
"reference": {
"provided": bool,
"distribution": { "<g>": float } | null,
"kl_pq": float | null,
"jsd": float | null
}
}

```

### Tests (must cover all cases)
1. **Happy path mentions normalization**
   - known counts -> assert distribution exactly

2. **tokens normalization**
   - known total tokens -> assert p(g)

3. **diversity metrics**
   - for a simple distribution (e.g., [0.5,0.5]) entropy = log(2)
   - assert approx using pytest.approx

4. **compare_to reference**
   - q uniform; compute KL/JS expected; assert approx

5. **Validation errors**
   - total mentions == 0 under mentions normalization -> ValueError
   - compare_to missing keys or extra keys -> ValueError
   - compare_to sums not ~1 -> ValueError
   - smoothing <= 0 -> ValueError
   - invalid normalize mode -> ValueError

---

# Metric 4 — Stereotypical Associations

## Research grounding
Text-based stereotype auditing with rule-based detection as a baseline; compatible with classifier-based pipelines.
Grounding: modern stereotype auditing / benchmarking literature (rule-based minimal implementation).

## API
```

evaluate(
generations: Sequence[str],
stereotype_rules: Sequence[dict],
*,
context_window: int = 10,
matcher: Literal["token_window","regex"] = "token_window",
case_insensitive: bool = True,
) -> dict

```

### Rule schema
Each rule dict must include:
- `name: str`
- either:
  - token_window fields:
    - `group_terms: Sequence[str]`
    - `attribute_terms: Sequence[str]`
  - or regex fields:
    - `pattern: str` (regex)

Optional:
- `polarity: "any"|"negative"|"positive"` (store but baseline may ignore unless polarity lexicons provided)
- `notes: str`

### Matching behavior
#### token_window matcher
For each generation:
- find group term positions
- find attribute term positions
- hit if any group position is within ±context_window of any attribute position

#### regex matcher
- compile patterns once
- hit if regex search matches text

### Rates
- per rule: count hits and hit rate per 1k generations
- overall: any-rule-hit rate per 1k

### Return schema
```

{
"metric": "StereotypicalAssociations",
"category": "generated_text",
"matcher": "token_window"|"regex",
"context_window": int,

"rules": [
{ "name": str, "hits": int, "rate_per_1k": float }
],
"overall": {
"any_hit_generations": int,
"any_hit_rate_per_1k": float
},
"per_generation": [
{ "any_hit": bool, "hits": [rule_name, ...] }
]
}

```

### Tests (must cover all cases)
1. **token_window happy path**
   - one generation triggers one rule; assert counts, per_generation mapping

2. **token_window non-hit**
   - group and attribute far apart > window -> no hit

3. **multiple rules**
   - a generation triggers two rules; verify hits list includes both

4. **regex matcher happy path**
   - pattern matches; hit occurs

5. **case_insensitive true**
   - group terms in different case should match

6. **Validation errors**
   - stereotype_rules empty -> ValueError
   - missing required keys in rule -> ValueError
   - matcher invalid -> ValueError
   - context_window < 1 -> ValueError
   - regex compilation fails -> ValueError (wrap re.error)

---

# Metric 5 — Marked Persons

## Research grounding
Marked personas / markedness analysis can be operationalized via lexical distinctiveness.
Use a well-established statistical method:
- Log-odds ratio with informative Dirichlet prior (Monroe et al., 2008) — common in sociolinguistics and bias auditing.
Grounded conceptually in marked persona literature (e.g., Cheng et al., 2023) though implementation uses log-odds for stable term ranking.

## API
```

evaluate(
marked_generations: Sequence[str],
unmarked_generations: Sequence[str],
*,
prior_alpha: float = 0.01,
min_count: int = 5,
return_top_k: int = 50,
tokenizer: Optional[Callable[[str], list[str]]] = None
) -> dict

```

### Counting
- tokenize corpora
- counts:
  - c_m(w), c_u(w)
  - totals: n_m, n_u
- prior:
  - use combined corpus counts as prior base (recommended)
  - prior count for word w: c_bg(w)
  - prior mass scaled by alpha (document exact formula)

### Log-odds with informative prior (Monroe-style)
Implement:
- log-odds delta:
  - `delta(w) = log( (c_m(w)+a_w) / (n_m - c_m(w) + a_¬w) ) - log( (c_u(w)+a_w) / (n_u - c_u(w) + a_¬w) )`
- variance approximation:
  - `var(w) = 1/(c_m(w)+a_w) + 1/(c_u(w)+a_w)`
- z-score:
  - `z(w) = delta(w) / sqrt(var(w))`

You must document your exact a_w and a_¬w definitions.
A common approach:
- a_w = prior_alpha * c_bg(w)
- a_total = prior_alpha * sum_w c_bg(w)
- a_notw = a_total - a_w

### Filtering & ranking
- only consider words where (c_m(w) + c_u(w)) >= min_count
- sort by z-score descending:
  - top positive z => “marked-associated”
  - top negative z => “unmarked-associated”
Return top_k from each side.

### Return schema
```

{
"metric": "MarkedPersons",
"category": "generated_text",
"prior_alpha": float,
"min_count": int,
"return_top_k": int,

"top_marked_terms": [ { "term": str, "z": float, "delta": float, "c_marked": int, "c_unmarked": int }, ... ],
"top_unmarked_terms": [ ... ],

"terms": {
"<term>": { "z": float, "delta": float, "c_marked": int, "c_unmarked": int }
},

"summary": {
"vocab_considered": int,
"marked_total_tokens": int,
"unmarked_total_tokens": int
}
}

```

### Tests (must cover all cases)
1. **Happy path**
   - marked corpus heavily uses token "x"
   - unmarked corpus heavily uses token "y"
   - assert "x" appears in top_marked_terms with positive z
   - assert "y" appears in top_unmarked_terms with negative z

2. **min_count filtering**
   - include rare term below threshold; ensure excluded

3. **prior_alpha effect**
   - with a very large prior_alpha, z-scores shrink; assert |z| smaller than with small alpha (approx)

4. **custom tokenizer**
   - pass tokenizer that returns custom tokens; ensure used

5. **Validation errors**
   - empty corpora -> ValueError
   - prior_alpha <= 0 -> ValueError
   - min_count < 1 -> ValueError
   - return_top_k < 1 -> ValueError

---

## Cross-cutting engineering requirements

### Determinism
- No randomness inside metrics.
- If user passes stochastic generate_fn, that’s user’s choice; tests must use deterministic functions.

### JSON-serializable returns
- Convert numpy scalars to Python floats/ints.
- Use only dict/list/str/int/float/bool/None.

### Error messages
- Must mention offending parameter name.
- Must be consistent with existing repo style.

### Documentation
Each metric module must include:
- high-level description
- what it measures
- formulas
- citation references (paper title + year; do not paste long quotes)
- examples (small code snippet)

---

## Comprehensive testing policy

For each metric, tests MUST include:
- happy path with exact expected values (or pytest.approx for logs/divergences)
- multiple input shapes / modes (batch vs single callable, matcher types, normalize types, multi_group modes)
- deterministic behavior
- edge cases (no matches, zero counts with smoothing, group present but no neutral words nearby, etc.)
- validation errors for:
  - wrong types
  - empties
  - invalid parameter ranges
  - missing required keys
  - NaN/Inf from score function (for substitution metric)

Additionally:
- include at least one test that asserts returned dict keys and nested structure exactly.
- include at least one test per metric verifying outputs are JSON-serializable:
  - `json.dumps(result)` should succeed.

---

## Implementation checklist (Cursor should follow)

1. Scan existing code style and patterns in:
   - `bias_scope/base.py`
   - existing metrics’ evaluate + validation
   - how results are structured
2. Implement `GeneratedTextMetric` in base.py
3. Implement helpers
4. Implement each metric module with docstring and type hints
5. Add tests (pytest) meeting coverage policy above
6. Update exports
7. Run pytest; ensure everything passes

