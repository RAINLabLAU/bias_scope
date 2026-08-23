# RegardScore

**Cited source:** Sheng, Chang, Natarajan & Peng 2019, *The Woman Worked as a
Babysitter: On Biases in Language Generation*, EMNLP 2019.
[arXiv:1909.01326](https://arxiv.org/abs/1909.01326).
**Reference implementation:**
[ewsheng/nlg-bias](https://github.com/ewsheng/nlg-bias) @ `7f8d08ea4f33` —
**no license file** (RL-015).
**Sections and files read:** §1 (the introduction of *regard* as distinct from
sentiment); **Table 2** (worked counterexamples where the two labels disagree);
§"we then use the annotations to build a classifier for regard"; repo layout
(`scripts/ensemble.py`, the `regard1` 3-BERT majority-vote ensemble).
**Family / access:** generated_text / `completions`

## Definition in the paper

Sheng et al. **introduce regard specifically because sentiment is the wrong
proxy**. That is the paper's contribution, not a footnote to it:

> we address the appropriateness of sentiment scores as a proxy for measuring
> bias ... We then use the annotations to build a classifier for regard, and use
> both sentiment and regard to present biases found in NLG systems.

Table 2 lists sentences where the two labels have **opposite signs**:

| Sentence | sentiment | regard |
|---|---|---|
| "XYZ was a pimp and her friend was happy." | + | − |
| "XYZ, known for his kindness, had died alone." | − | + |

The metric is the **distribution of regard labels per demographic group**, and
the bias is the difference between groups' distributions.

## Verdict

**adaptation**, as of v0.2.0.

**History: v0.1.1 was a `mismatch`,** and a clear one. It

- defaulted to `siebert/sentiment-roberta-large-english` — a **sentiment**
  classifier;
- described itself in its own docstring as measuring "sentiment differences"
  using "a sentiment classifier";
- and folded every unrecognised label into `neutral`, so a four-bucket regard
  checkpoint's `other` class silently became "neutral regard".

A class named `RegardScore`, citing Sheng et al., computing sentiment, is the
exact conflation the cited paper was written to refute. PLAN.md 4.2 listed this
metric as `unchecked` with the action "audit classifier checkpoint" — the audit
found the checkpoint was the wrong *kind* of classifier.

## Resolution (v0.2.0)

1. Default changed to **`sasha/regardv3`**, Sheng's published regard checkpoint.
2. The four regard buckets are declared (`REGARD_LABELS`) and `other` is
   reported separately from `neutral`, because "the classifier could not place
   this on the regard scale" is not "this is mid-scale".
3. `LABEL_0..3` mapped explicitly rather than by substring guessing.
4. The constructor docstring carries an explicit warning that passing a
   sentiment classifier does not compute regard.

Status is **adaptation**, not `faithful`, for one honest reason: Sheng's own
`regard1` is a **3-BERT majority-vote ensemble** (`ewsheng/nlg-bias`
`scripts/ensemble.py`), and `sasha/regardv3` is a single later checkpoint
trained on the v2 dataset with the added `other` bucket. It is a regard
classifier, but it is not the classifier that produced the paper's numbers.

`scripts/experiments/repro_regard_fig2_rescore.py` already loads the real
`regard1` ensemble, so the faithful path exists — it is just not the library
default, because the ensemble is three checkpoints totalling several GB.

## Required action

- Expose the `regard1` ensemble as an opt-in scorer so a user can choose the
  faithful path without leaving the library.
- Record the classifier checkpoint in the protocol block. Two RegardScore
  numbers from different checkpoints are not comparable, and nothing currently
  forces that to be visible.

## Validation possible

- **Tier 1:** Sheng's Figure 2 and Table 3 give regard distributions for GPT-2
  across six demographics. `results/emnlp/regard_full/` already contains a
  6,000-completion reproduction scored **both** ways — `sasha/regardv3` and the
  real `regard1` ensemble — which is exactly the evidence needed to quantify the
  checkpoint deviation. That comparison should go in the paper's
  protocol-sensitivity section (PLAN.md 10.2).
- **Tier 2:** the repo states **no license** (RL-015), so its code may be run
  locally but not vendored.
- **Tier 3:** null (identical groups → all differences 0) and swap antisymmetry
  (swapping the groups negates every difference) both apply and are tested.

## Known limitations of the metric itself

- Entirely dependent on the regard classifier, which is trained on a small
  annotated set and inherits its annotators' judgements.
- The `other` bucket absorbs anything the classifier cannot place, and its size
  varies by model — a large `other` share makes the remaining distribution hard
  to interpret.
- Regard is defined toward a *demographic referent*, so the metric assumes the
  generation is actually about the intended group.
