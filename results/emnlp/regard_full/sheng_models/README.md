# Sheng et al. regard classifiers — configs only

These three directories hold the tokenizer and config for the BERT regard
classifiers used to reproduce Sheng et al. 2019 Figure 2
(`scripts/experiments/repro_regard_fig2_rescore.py`).

**The weights are not here.** `pytorch_model.bin` and the optimizer state are
git-ignored: they are ~1.3 GB each and are the authors' artefact, not ours to
redistribute. What is committed is what identifies the classifier — the vocab,
the label map in `config.json`, and the tokenizer settings — so a rerun can be
checked against the same configuration.

To rerun, fetch the checkpoints from `github.com/ewsheng/nlg-bias` (the
`regard1` release) into these directories, then run the script. Labels are
`0 = negative, 1 = neutral, 2 = positive`.
