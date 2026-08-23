"""Generate a canonical set of GPT-2 completions for HONEST diagnosis.

Produces:
  - honest_completions.jsonl : {template, seed, completions:[str,...]}
  - honest_first_words.jsonl : {template, seed, words:[str,...]}   (max_new_tokens=1 path)

Saves for seeds 0, 1, 2, 3, 42. Uses the exact protocol we settled on
(top_p=0.95, temperature=1.0, max_new_tokens=1, K=5).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import pipeline
from unidecode import unidecode as _uni

OUT = Path("/home/chadi/Desktop/bias_scope/results/emnlp/honest_diagnosis")
OUT.mkdir(parents=True, exist_ok=True)

SEEDS = [0, 1, 2, 3, 42]
K = 5


def load_templates() -> list[str]:
    """Nozza's en_binary templates (810), stripped of trailing '[M].'"""
    import csv

    tsv_path = hf_hub_download(
        "MilaNLProc/honest",
        "data/en/en_binary_template.tsv",
        repo_type="dataset",
    )
    templates: list[str] = []
    with open(tsv_path) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            t = row["template_masked"]
            for suf in (" [M].", "[M].", " [M]", "[M]"):
                t = t.replace(suf, "")
            t = t.rstrip(" .")
            if t:
                templates.append(t)
    return templates


def gen_for_seed(gen, templates: list[str], seed: int) -> list[dict]:
    """Return a list, one entry per template, each entry has:
    template, full_completions (str, prompt+continuation), first_words.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    out = []
    for tmpl in templates:
        # Same decoding args as scripts/experiments/finalize_emnlp.py
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
        completions = [o["generated_text"] for o in outs]
        # Also expose the "single first predicted word" per completion,
        # normalised the same way `finalize_emnlp.py` did.
        first_words = []
        for c in completions:
            cont = c[len(tmpl):].strip().split()
            w = cont[0] if cont else ""
            # NORMALISE: our finalize path did lowercase + strip punctuation + unidecode
            w_norm = _uni(w.lower().strip(".,:;!?\"'()[]{}"))
            # RAW: what we'd hand to Nozza's word-mode scorer
            first_words.append({"raw": w, "norm": w_norm})
        out.append({
            "template": tmpl,
            "completions": completions,
            "first_words": first_words,
        })
    return out


def main() -> None:
    print("Loading Nozza en_binary templates ...")
    templates = load_templates()
    print(f"  {len(templates)} templates")

    print("Loading GPT-2 ...")
    gen = pipeline("text-generation", model="gpt2", device=0)

    for seed in SEEDS:
        print(f"[seed {seed}] generating {len(templates)}×{K}={len(templates)*K} candidates ...")
        entries = gen_for_seed(gen, templates, seed)
        path = OUT / f"honest_completions_seed{seed}.jsonl"
        with path.open("w") as f:
            for e in entries:
                f.write(json.dumps({"seed": seed, **e}) + "\n")
        print(f"  wrote {path}")


if __name__ == "__main__":
    main()
