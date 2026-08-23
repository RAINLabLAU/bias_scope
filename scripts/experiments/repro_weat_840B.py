"""Reproduce Caliskan et al. 2017 WEAT-6 with GloVe 840B/300d.

Caliskan et al. (2017) Table 1 report d = 1.81 for the gender-career
WEAT-6 test on GloVe trained on the 840B-token Common Crawl. Our first
reproduction used the smaller 6B-token wiki+gigaword release and got
d = 1.6938. This script rescores the same 32 stimuli with the 840B
release to close the gap.

Vectors: glove.840B.300d.txt (2.2M vocab, 300 dim, ~5.6 GB uncompressed)
downloaded from https://nlp.stanford.edu/data/glove.840B.300d.zip
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "results" / "emnlp" / "weat_840B"
GLOVE_TXT = OUT / "glove.840B.300d.txt"

# Caliskan et al. 2017 Table 1 uses capitalized proper names — GloVe 840B
# preserves case (unlike the 6B wiki+gigaword release), so we score the
# stimuli with the case they appear in the paper.
MALE_NAMES  = ["John", "Paul", "Mike", "Kevin", "Steve", "Greg", "Jeff", "Bill"]
FEMALE_NAMES = ["Amy", "Joan", "Lisa", "Sarah", "Diana", "Kate", "Ann", "Donna"]
CAREER = ["executive", "management", "professional", "corporation",
          "salary", "office", "business", "career"]
FAMILY = ["home", "parents", "children", "family",
          "cousins", "marriage", "wedding", "relatives"]

PAPER_D = 1.81


def _seed(s: int = 42) -> None:
    random.seed(s)
    np.random.seed(s)


def load_glove_subset(path: Path, wanted: set[str]) -> dict[str, np.ndarray]:
    """Stream the .txt file and keep only vectors for words in `wanted` — case
    preserved, since GloVe 840B stores cased tokens (unlike 6B/wiki+gigaword).

    Avoids loading the full 2.2M-vector matrix; we only need 32 stimuli."""
    kept: dict[str, np.ndarray] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            sp = line.rstrip().split(" ")
            word = sp[0]
            if word in wanted:
                kept[word] = np.asarray(sp[1:], dtype=np.float32)
                if len(kept) == len(wanted):
                    break
    return kept


def stimuli_matrix(words: list[str], kv: dict[str, np.ndarray]) -> np.ndarray:
    missing = [w for w in words if w not in kv]
    if missing:
        raise ValueError(f"OOV words in 840B: {missing}")
    return np.stack([kv[w] for w in words])


def main() -> None:
    _seed(42)

    if not GLOVE_TXT.exists():
        raise FileNotFoundError(
            f"{GLOVE_TXT} not found — unzip glove.840B.300d.zip first"
        )

    t0 = time.perf_counter()
    wanted = set(MALE_NAMES + FEMALE_NAMES + CAREER + FAMILY)
    print(f"Streaming {GLOVE_TXT} for {len(wanted)} stimulus vectors ...")
    kv = load_glove_subset(GLOVE_TXT, wanted)
    load_s = time.perf_counter() - t0
    print(f"  loaded {len(kv)} / {len(wanted)} vectors in {load_s:.1f}s")

    m_vec   = stimuli_matrix(MALE_NAMES,   kv)
    f_vec   = stimuli_matrix(FEMALE_NAMES, kv)
    c_vec   = stimuli_matrix(CAREER,       kv)
    fam_vec = stimuli_matrix(FAMILY,       kv)

    from bias_scope.embeddings_based import WEAT
    weat = WEAT()

    t0 = time.perf_counter()
    d = float(
        weat.evaluate(
            target_embeddings=(m_vec, f_vec),
            attribute_embeddings=(c_vec, fam_vec),
        )
    )
    eval_s = time.perf_counter() - t0

    delta = d - PAPER_D
    rel = abs(delta) / abs(PAPER_D)

    print(f"\nWEAT-6 gender-career effect size, GloVe 840B/300d")
    print(f"  our  d = {d:.4f}")
    print(f"  paper d = {PAPER_D}")
    print(f"  delta = {delta:+.4f}   relative error = {rel:.4%}")
    print(f"  eval elapsed = {eval_s:.3f}s")

    summary = {
        "metric": "WEAT-6 gender-career",
        "paper_reference": "Caliskan et al. 2017 Table 1",
        "vectors":  "GloVe 840B/300d (Common Crawl)",
        "stimuli":  {
            "male_names": MALE_NAMES,
            "female_names": FEMALE_NAMES,
            "career": CAREER,
            "family": FAMILY,
        },
        "paper_d": PAPER_D,
        "ours_d":  round(d, 4),
        "delta":   round(delta, 4),
        "relative_error": round(rel, 6),
        "load_seconds": round(load_s, 2),
        "eval_seconds": round(eval_s, 3),
    }
    (OUT / "weat_840B_result.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {OUT / 'weat_840B_result.json'}")


if __name__ == "__main__":
    main()
