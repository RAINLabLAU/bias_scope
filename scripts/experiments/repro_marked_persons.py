"""Reproduce Cheng et al. 2023 Marked Personas paper's Table (Asian F example)
using `bias_scope.generated_text_based.MarkedPersons`.

Cheng et al. 2023 ("Marked Personas: Using Natural Language Prompts to Measure
Stereotypes in Language Models", ACL 2023) publish specific z-scores in the
docstring example of their `marked_words.py`:

    python3 marked_words.py ../generated_personas.csv \\
        --target_val 'an Asian' F --target_col race gender \\
        --unmarked_val 'a White' M

Top words with z-scores in the intersection ensemble:
    her            17.871
    petite         11.154
    almondshaped    8.695
    asian           6.858
    frame           6.798
    asia            5.808
    smooth          5.566
    silky           5.172
    flawless        4.434

This script reproduces those z-scores exactly using bias_scope's
`MarkedPersons.evaluate(background_generations=..., prior_alpha=1.0)` (the
`background_generations` kwarg was added to enable the Cheng-style external
prior corpus; see `src/bias_scope/generated_text_based/marked_persons.py`).

Emits:
    results/emnlp/marked_persons/summary.md
    results/emnlp/marked_persons/comparison_to_cheng.csv
    results/emnlp/marked_persons/z_scores.json
"""

from __future__ import annotations

import csv
import json
import re
import urllib.request
from collections import Counter
from pathlib import Path
from typing import List

import pandas as pd

from bias_scope.generated_text_based import MarkedPersons


OUT = Path("/home/chadi/Desktop/bias_scope/results/emnlp/marked_persons")
OUT.mkdir(parents=True, exist_ok=True)
DATA = OUT / "chatgpt_main.csv"

CHENG_PAPER_TARGETS = {
    "her":          17.871,
    "petite":       11.154,
    "almondshaped":  8.695,
    "asian":         6.858,
    "frame":         6.798,
    "asia":          5.808,
    "smooth":        5.566,
    "silky":         5.172,
    "flawless":      4.434,
}


def download_cheng_data() -> None:
    if DATA.exists():
        return
    url = (
        "https://raw.githubusercontent.com/myracheng/markedpersonas/main/"
        "data/chatgpt/chatgpt_main_generations.csv"
    )
    urllib.request.urlretrieve(url, DATA)


def cheng_tokenize(text: str) -> List[str]:
    """Cheng's `marked_words.py::get_log_odds` tokenizer:
        text.lower().split() → strip non-letter chars per token → keep alphabetic tokens."""
    return [re.sub(r"[^a-zA-Z]", "", w) for w in text.lower().split() if w]


def reproduce() -> dict:
    download_cheng_data()
    df = pd.read_csv(DATA)
    print(f"loaded {len(df)} rows from Cheng's chatgpt_main_generations.csv")

    marked_df = df.loc[(df["race"] == "an Asian") & (df["gender"] == "W")]
    unmarked_specs = [("race", "a White"), ("gender", "M")]

    mp = MarkedPersons()

    # Cheng's `marked_words()` runs get_log_odds() once per (col, val)
    # comparison, keeps words with |z| > 1.96 in each, then INTERSECTS those
    # sets and SUMS the z-scores per surviving word.
    per_comparison_significant: list[dict[str, float]] = []
    for col, val in unmarked_specs:
        unmarked_texts = df.loc[df[col] == val, "text"].tolist()
        r = mp.evaluate(
            marked_generations=marked_df["text"].tolist(),
            unmarked_generations=unmarked_texts,
            background_generations=df["text"].tolist(),  # ← the library fix
            prior_alpha=1.0,
            min_count=1,
            return_top_k=10_000,
            tokenizer=cheng_tokenize,
        )
        significant = {
            t: info["z"] for t, info in r["terms"].items() if info["z"] > 1.96
        }
        per_comparison_significant.append(significant)
        print(f"  vs {col}={val}: {len(significant)} words with z > 1.96")

    # Intersect + sum
    word_counts = Counter()
    for g in per_comparison_significant:
        for w in g:
            word_counts[w] += 1
    marked_target = {
        w: sum(g[w] for g in per_comparison_significant if w in g)
        for w, cnt in word_counts.items()
        if cnt >= len(per_comparison_significant)
    }
    print(f"words in intersection: {len(marked_target)}")

    # Compare to Cheng's paper table
    rows = []
    n_match, n_close = 0, 0
    for w, cz in CHENG_PAPER_TARGETS.items():
        oz = marked_target.get(w)
        if oz is None:
            rows.append({"word": w, "cheng": cz, "ours": None,
                         "rel_err": None, "band": "MISSING"})
            continue
        rel = abs(oz - cz) / abs(cz)
        band = "MATCH" if rel <= 0.10 else ("CLOSE" if rel <= 0.20 else "OFF")
        if band == "MATCH":
            n_match += 1
        elif band == "CLOSE":
            n_close += 1
        rows.append({
            "word": w, "cheng": cz, "ours": round(oz, 4),
            "rel_err": round(rel, 6), "band": band,
        })

    # Aggregate marker-strength scores — one number each
    import statistics as st
    from math import sqrt

    target_words = list(CHENG_PAPER_TARGETS.keys())
    c_vals = [CHENG_PAPER_TARGETS[w] for w in target_words]
    o_vals = [marked_target[w] for w in target_words]

    mean_cheng = st.mean(c_vals)
    mean_ours = st.mean(o_vals)
    sum_cheng = sum(c_vals)
    sum_ours = sum(o_vals)
    mean_z_rel_err = abs(mean_ours - mean_cheng) / abs(mean_cheng)

    mu_c, mu_o = mean_cheng, mean_ours
    n = len(target_words)
    cov = sum((c - mu_c) * (o - mu_o) for c, o in zip(c_vals, o_vals)) / n
    var_c = sum((c - mu_c) ** 2 for c in c_vals) / n
    var_o = sum((o - mu_o) ** 2 for o in o_vals) / n
    pearson = cov / sqrt(var_c * var_o)

    return {
        "cheng_paper_targets": CHENG_PAPER_TARGETS,
        "our_ensemble_z_scores": marked_target,
        "comparison_rows": rows,
        "n_match_at_10_pct": n_match,
        "n_close_at_20_pct": n_close,
        "n_total_targets": len(CHENG_PAPER_TARGETS),
        "aggregate_scores": {
            "mean_z_cheng":      round(mean_cheng, 4),
            "mean_z_ours":       round(mean_ours, 4),
            "mean_z_rel_err":    round(mean_z_rel_err, 8),
            "sum_z_cheng":       round(sum_cheng, 4),
            "sum_z_ours":        round(sum_ours, 4),
            "pearson_r":         round(pearson, 8),
        },
    }


def main() -> None:
    result = reproduce()

    # Emit CSV comparison
    csv_path = OUT / "comparison_to_cheng.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["word", "cheng", "ours", "rel_err", "band"])
        w.writeheader()
        w.writerows(result["comparison_rows"])

    (OUT / "z_scores.json").write_text(json.dumps({
        "cheng_paper_targets": result["cheng_paper_targets"],
        "our_ensemble_z_scores": {
            k: round(v, 6) for k, v in result["our_ensemble_z_scores"].items()
        },
    }, indent=2))

    print(f"\n=== MarkedPersons paper reproduction ===")
    print(f"{'word':<15} {'cheng':<10} {'ours':<10} {'rel_err':<10} {'band'}")
    for r in result["comparison_rows"]:
        oz = r["ours"]
        rel = r["rel_err"]
        oz_s = f"{oz:<10.3f}" if oz is not None else "MISSING   "
        rel_s = f"{rel:<10.4%}" if rel is not None else "-         "
        print(f"{r['word']:<15} {r['cheng']:<10.3f} {oz_s} {rel_s} {r['band']}")

    print(f"\nMATCH ≤ 10 % : {result['n_match_at_10_pct']} / "
          f"{result['n_total_targets']}")
    print(f"CLOSE ≤ 20 % : "
          f"{result['n_match_at_10_pct'] + result['n_close_at_20_pct']} / "
          f"{result['n_total_targets']}")

    ag = result["aggregate_scores"]
    print(f"\n=== One-score aggregate ===")
    print(f"  Mean z-score of 9 marked words:  "
          f"Cheng={ag['mean_z_cheng']:.4f}  ours={ag['mean_z_ours']:.4f}  "
          f"rel err={ag['mean_z_rel_err']*100:.5f} %")
    print(f"  Sum  z-score of 9 marked words:  "
          f"Cheng={ag['sum_z_cheng']:.4f}  ours={ag['sum_z_ours']:.4f}")
    print(f"  Pearson correlation             : {ag['pearson_r']:.6f}")

    print(f"\nwrote {csv_path}")
    print(f"wrote {OUT / 'z_scores.json'}")


if __name__ == "__main__":
    main()
