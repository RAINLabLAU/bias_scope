"""Run finalize_emnlp.py's exact HONEST generation code path and dump every
first-word to disk so we can cross-score them with the reference package.

We copy the code verbatim from finalize_emnlp.py::rerun_honest_full so
whatever RNG state finalize sees, we see. Only difference: after building
`predicted_words`, we also save it, plus the full completions.
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import pipeline
from unidecode import unidecode

OUT = Path("/home/chadi/Desktop/bias_scope/results/emnlp/honest_diagnosis")
OUT.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path("/home/chadi/Desktop/bias_scope/results/emnlp/data")


def _seed(s: int = 42):
    random.seed(s)
    np.random.seed(s)
    import os

    os.environ["PYTHONHASHSEED"] = str(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def main():
    _seed(42)

    # ── Templates: same as finalize_emnlp.py ─────────────────────────
    tsv_path = hf_hub_download(
        "MilaNLProc/honest",
        "data/en/en_binary_template.tsv",
        repo_type="dataset",
    )
    with open(tsv_path) as f:
        templates_raw = []
        for row in csv.DictReader(f, delimiter="\t"):
            t = row["template_masked"]
            for suf in (" [M].", "[M].", " [M]", "[M]"):
                t = t.replace(suf, "")
            t = t.rstrip(" .")
            if t:
                templates_raw.append(t)
    print(f"templates: {len(templates_raw)}")

    # ── HurtLex: same as finalize ────────────────────────────────────
    hurtlex_words: set[str] = set()
    nozza_path = DATA_DIR / "hurtlex_EN_nozza.tsv"
    src = nozza_path if nozza_path.exists() else (DATA_DIR / "hurtlex_EN.tsv")
    with src.open() as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row.get("level") != "conservative":
                continue
            lemma = unidecode(str(row["lemma"]))
            hurtlex_words.add(lemma)
    print(f"hurtlex: {len(hurtlex_words)} lemmas")

    # ── Generation: EXACT copy of finalize block ─────────────────────
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    gen = pipeline("text-generation", model="gpt2", device=0)

    predicted_words: list[list[str]] = []
    full_completions: list[list[str]] = []
    _uni = unidecode
    K = 5
    for tmpl in templates_raw:
        outs = gen(
            tmpl,
            max_new_tokens=1,
            num_return_sequences=K,
            do_sample=True,
            top_p=0.95,
            top_k=0,
            temperature=1.0,
            pad_token_id=gen.tokenizer.eos_token_id,
        )
        words = []
        fulls = []
        for o in outs:
            full = o["generated_text"]
            fulls.append(full)
            cont = full[len(tmpl):].strip().split()
            w = cont[0] if cont else ""
            w = _uni(w.lower().strip(".,:;!?\"'()[]{}"))
            words.append(w)
        predicted_words.append(words)
        full_completions.append(fulls)

    # in-script scoring, verbatim from finalize
    hurt_count = 0
    for word_list in predicted_words:
        for w in word_list:
            if w in hurtlex_words:
                hurt_count += 1
    total = len(predicted_words) * K
    print(f"HONEST (finalize-style, self-scored): {hurt_count / total:.4f} "
          f"({hurt_count}/{total})")

    # ── Save the exact completions finalize produced ─────────────────
    path = OUT / "finalize_completions_seed42.jsonl"
    with path.open("w") as f:
        for i, (tmpl, fulls, words) in enumerate(
            zip(templates_raw, full_completions, predicted_words)
        ):
            f.write(json.dumps({
                "template": tmpl,
                "completions": fulls,
                "first_words": words,
            }) + "\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
