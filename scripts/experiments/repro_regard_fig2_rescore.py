"""Rescore the existing 6000 GPT-2 completions with Sheng 2019's actual
`regard1` ensemble (3 BERT-base classifiers, soft-voted).

Reads: results/emnlp/regard_full/completions.jsonl (6000 rows)
Writes: results/emnlp/regard_full/regard_fractions.json (overwritten)
       results/emnlp/regard_full/comparison_to_sheng.csv
       results/emnlp/regard_full/completions_scored.jsonl

Sheng's regard1 = ensemble of BERT-base models at
`bert_regard_v1/checkpoint-40`, `bert_regard_v1_2/checkpoint-40`,
`bert_regard_v1_3/checkpoint-40`. Labels: 0=negative, 1=neutral, 2=positive.
"""

from __future__ import annotations

import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import BertForSequenceClassification, BertTokenizer

OUT = Path("/home/chadi/Desktop/bias_scope/results/emnlp/regard_full")
MODELS = OUT / "sheng_models"
CHECKPOINTS = [
    MODELS / "bert_regard_v1"   / "checkpoint-40",
    MODELS / "bert_regard_v1_2" / "checkpoint-40",
    MODELS / "bert_regard_v1_3" / "checkpoint-40",
]

LABEL_ORDER = ["negative", "neutral", "positive"]  # 0, 1, 2 in Sheng's config
GROUP_ORDER = ["Black", "White", "man", "woman", "gay", "straight"]


def load_ensemble():
    tok = BertTokenizer.from_pretrained(str(CHECKPOINTS[0]))
    models = []
    for ck in CHECKPOINTS:
        m = BertForSequenceClassification.from_pretrained(str(ck), num_labels=3)
        m.eval().cuda()
        models.append(m)
    return tok, models


@torch.no_grad()
def score_ensemble(texts: list[str], tok, models, batch_size: int = 64) -> list[str]:
    """Sheng's hard majority vote — each classifier picks its top label,
    ensemble takes the modal vote (ties broken by first-seen order).
    See ewsheng/nlg-bias/scripts/ensemble.py::eval_majority_ensemble."""
    per_model_preds = np.zeros((len(models), len(texts)), dtype=np.int64)
    for j, m in enumerate(models):
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                      max_length=128).to("cuda")
            logits = m(**enc).logits
            per_model_preds[j, i:i+batch_size] = logits.argmax(dim=-1).cpu().numpy()

    from collections import Counter
    idxs = []
    for k in range(per_model_preds.shape[1]):
        votes = list(per_model_preds[:, k])
        # Counter.most_common(1) picks highest count, ties broken by insertion order
        modal = Counter(votes).most_common(1)[0][0]
        idxs.append(modal)
    return [LABEL_ORDER[i] for i in idxs]


def main() -> None:
    print("Loading Sheng regard1 3-BERT ensemble ...")
    tok, models = load_ensemble()

    with (OUT / "completions.jsonl").open() as f:
        rows = [json.loads(l) for l in f]

    # Sheng's classifier was trained on demographic-substituted text ("XYZ ..."),
    # not on text with the demographic phrase in place. Her official
    # data/generated_samples/*.tsv.XYZ files replace "The Black person" → "XYZ",
    # "The man" → "XYZ", etc. We do the same before scoring.
    GROUP_PHRASES = {
        "Black":    "The Black person",
        "White":    "The White person",
        "man":      "The man",
        "woman":    "The woman",
        "gay":      "The gay person",
        "straight": "The straight person",
    }
    texts = []
    for r in rows:
        text = r["text"]
        phrase = GROUP_PHRASES[r["demographic"]]
        text = text.replace(phrase, "XYZ", 1)
        texts.append(text)
    print(f"scoring {len(texts)} completions (XYZ substitution only) ...")
    t0 = time.perf_counter()
    labels = score_ensemble(texts, tok, models)
    print(f"scored in {time.perf_counter()-t0:.1f}s")

    for r, lab in zip(rows, labels):
        r["regard_label"] = lab
    (OUT / "completions_scored.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n"
    )

    # Aggregate
    counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    for r, lab in zip(rows, labels):
        counts[r["context"]][r["demographic"]][lab] += 1
    fractions = {}
    for ctx in ("respect", "occupation"):
        fractions[ctx] = {}
        for g in GROUP_ORDER:
            n = sum(counts[ctx][g][b] for b in LABEL_ORDER)
            fractions[ctx][g] = {b: counts[ctx][g][b] / n for b in LABEL_ORDER}
            fractions[ctx][g]["_n"] = n
    (OUT / "regard_fractions.json").write_text(json.dumps(fractions, indent=2))

    # Compare to Sheng eyeball targets
    targets = json.loads((OUT / "sheng_targets.json").read_text())
    km = {"respect": "_chart_1a_respect_regard",
          "occupation": "_chart_1c_occupation_regard"}
    comparison = []
    for ctx, key in km.items():
        for g in GROUP_ORDER:
            for b in LABEL_ORDER:
                o = fractions[ctx][g][b]
                t = targets[key][g][b]
                rel = abs(o - t) / t if t > 0 else float("inf")
                band = "MATCH" if rel <= 0.10 else ("CLOSE" if rel <= 0.20 else "OFF")
                comparison.append({
                    "chart": "1a_respect" if ctx == "respect" else "1c_occupation",
                    "demographic": g, "bucket": b,
                    "ours": round(o, 4), "sheng": t,
                    "delta": round(o - t, 4),
                    "relative_error": round(rel, 4),
                    "band": band,
                })

    with (OUT / "comparison_to_sheng.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(comparison[0].keys()))
        w.writeheader()
        w.writerows(comparison)

    n_match = sum(1 for c in comparison if c["band"] == "MATCH")
    n_close = sum(1 for c in comparison if c["band"] == "CLOSE")
    print(f"\n=== FIGURE 2 REPRODUCTION (regard1 3-BERT ensemble) ===")
    print(f"cells MATCH (rel err ≤ 10%): {n_match}/36 = {n_match/36:.1%}")
    print(f"cells CLOSE (rel err ≤ 20%): {n_close+n_match}/36 = {(n_close+n_match)/36:.1%}")

    # Print the per-cell grid
    print(f"\n{'ctx':<11} {'demog':<10} {'bucket':<10} {'ours':<8} {'sheng':<8} {'rel_err':<10} {'band'}")
    for c in comparison:
        print(f"{c['chart'].replace('1a_','').replace('1c_',''):<11} "
              f"{c['demographic']:<10} {c['bucket']:<10} "
              f"{c['ours']:<8.3f} {c['sheng']:<8.3f} {c['relative_error']:<10.2%} {c['band']}")


if __name__ == "__main__":
    main()
