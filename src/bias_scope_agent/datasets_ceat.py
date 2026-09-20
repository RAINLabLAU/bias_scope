"""The contexts provider for CEAT (Guo & Caliskan 2021).

CEAT needs, for every WEAT word, many naturally occurring contexts, and
combines the WEAT effect sizes over N random draws of contexts. The authors
sample from a Reddit corpus that is not vendored here. This provider draws
each word's contexts from BOLD's Wikipedia sentences (Dhamala et al. 2021,
vendored, CC-BY-SA), and CEAT embeds each context as a sentence with the
model under evaluation. Both are substitutions and both are written into the
result's protocol as deviations; the corpus was the user's call
(REVIEW_LATER RL-071).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

from bias_scope_agent.datasets_common import (
    _WEAT_BY_AXIS,
    DatasetSpec,
    _association_test,
    _init_kwargs,
    _sha256,
    _word_sets,
)

_WIKI_DIR = "bold/wikipedia"
_MAX_CONTEXTS_PER_WORD = 50  # bounds the embedding cost; recorded in the provenance
_N_SAMPLES = 1000  # Guo & Caliskan report N = 1,000 and 10,000; the metric's default is 100
_SEED = 42
_DEVIATIONS = [
    {
        "name": "context_corpus",
        "source": f"{_WIKI_DIR}/*_wiki.json",
        "deviation": "contexts drawn from BOLD's Wikipedia sentences, not the Reddit corpus "
                     "Guo & Caliskan sampled",
    },
    {
        "name": "context_embedding",
        "source": "CEAT.evaluate(pooling='cls')",
        "deviation": "each context is embedded as a whole sentence (position-0 pooling), not "
                     "as the target word's own token embedding inside the sentence",
    },
]

CEAT_DATASETS: Dict[str, DatasetSpec] = {
    "ceat_contexts": DatasetSpec(
        name="ceat_contexts",
        description=(
            "Contexts for CEAT: for each word of the Caliskan et al. 2017 test for "
            "the axis, the BOLD Wikipedia sentences containing it (whole word, any "
            "case, at most 50 per word), embedded by CEAT with the model under "
            "evaluation and resampled N=1,000 times. SUBSTITUTE CORPUS: the "
            "authors sampled Reddit; and each context is embedded as a sentence, "
            "not as the word's token. Both are recorded in the protocol as "
            "deviations. Words with no context are dropped and listed."
        ),
        metrics=("CEAT",),
        axes=tuple(_WEAT_BY_AXIS),
        source=f"{_WIKI_DIR}/*_wiki.json + sent-bias/tests/weat<n>.jsonl",
    ),
}


def _corpus(root: Path) -> Tuple[List[Path], List[str]]:
    """Every BOLD Wikipedia sentence, files and sentences in sorted file order."""
    files = sorted((root / _WIKI_DIR).glob("*_wiki.json"))
    if not files:
        raise ValueError(
            f"{root / _WIKI_DIR} has no *_wiki.json files. third_party/ is git-ignored; "
            "restore it with\n    python scripts/sources/fetch_sources.py --metric BOLD"
        )
    sentences: List[str] = []
    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        sentences.extend(s for group in data.values() for texts in group.values() for s in texts)
    return files, sentences


def _contexts_for(words: Sequence[str], sentences: List[str]) -> Tuple[List[str], List[str]]:
    """Sentences containing each word (whole word, case-insensitive), capped
    per word; returns (contexts, words that had none)."""
    contexts, dropped = [], []
    for word in words:
        pattern = re.compile(r"\b" + re.escape(word) + r"\b", re.IGNORECASE)
        found = [s for s in sentences if pattern.search(s)][:_MAX_CONTEXTS_PER_WORD]
        if not found:
            dropped.append(word)
        contexts.extend(found)
    return contexts, dropped


def _build_ceat_contexts(backend, metrics, axis, limit, root, allowed) -> Tuple[Dict, Dict]:
    test_path = _association_test(root, axis, _WEAT_BY_AXIS, "CEAT")
    files, sentences = _corpus(root)
    sets = _word_sets(test_path)
    built = [_contexts_for(words, sentences) for words in sets]
    (targ1, d1), (targ2, d2), (attr1, d3), (attr2, d4) = built
    protocol = {"dataset": f"{_WIKI_DIR} + {test_path.name}", "resources": _DEVIATIONS}
    inputs = {
        name: {
            "__init__": _init_kwargs(name, backend, allowed),
            "target_embeddings": (targ1, targ2),
            "attribute_embeddings": (attr1, attr2),
            "n_samples": _N_SAMPLES,
            "random_seed": _SEED,
            "__protocol__": protocol,
        }
        for name in metrics
    }
    provenance = {
        "source": str(test_path),
        "sha256": _sha256(test_path),
        "corpus": {"files": [str(p) for p in files],
                   "sha256": [_sha256(p) for p in files], "sentences": len(sentences)},
        "contexts": {"targets": [len(targ1), len(targ2)], "attributes": [len(attr1), len(attr2)]},
        "dropped_words": d1 + d2 + d3 + d4,
        "max_contexts_per_word": _MAX_CONTEXTS_PER_WORD,
        "n_samples": _N_SAMPLES,
        "seed": _SEED,
        "axis": axis,
        "deviations": [r["deviation"] for r in _DEVIATIONS],
    }
    return inputs, provenance


CEAT_BUILDERS: Dict[str, Callable[..., Tuple[Dict, Dict]]] = {
    "ceat_contexts": _build_ceat_contexts,
}
