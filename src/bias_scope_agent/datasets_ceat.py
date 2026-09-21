"""The contexts provider for CEAT (Guo & Caliskan 2021).

CEAT needs, for every WEAT word, many naturally occurring contexts, and
combines the WEAT effect sizes over N random draws of contexts. The authors
sample from a Reddit corpus that is not vendored here, so this provider draws
each word's contexts from BOLD's Wikipedia sentences (Dhamala et al. 2021,
vendored, CC-BY-SA) - a substitution written into the result's protocol as a
deviation; the corpus was the user's call (REVIEW_LATER RL-071).

CEAT takes, per stimulus, a matrix of that word's *contextual token
embeddings* (the 2026-09 audit's API; it does not encode text itself). This
provider computes them with the model under evaluation: each context
sentence is run once and the word's own subword hidden states are averaged.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

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
]
_EMBEDDING_NOTE = "each word's own contextual token embedding, mean over its subword pieces"

CEAT_DATASETS: Dict[str, DatasetSpec] = {
    "ceat_contexts": DatasetSpec(
        name="ceat_contexts",
        description=(
            "Contexts for CEAT: for each word of the Caliskan et al. 2017 test for "
            "the axis, the BOLD Wikipedia sentences containing it (whole word, any "
            "case, at most 50 per word). The harness embeds each word IN its "
            "context with the model under evaluation (the word's own subword "
            "states, averaged) and CEAT resamples N=1,000 times. SUBSTITUTE "
            "CORPUS: the authors sampled Reddit; recorded in the protocol as a "
            "deviation. A word with no context is an error, because CEAT needs "
            "the two target sets (and the two attribute sets) to stay equal in size."
        ),
        metrics=("CEAT",),
        axes=tuple(_WEAT_BY_AXIS),
        source=f"{_WIKI_DIR}/*_wiki.json + sent-bias/tests/weat<n>.jsonl",
        init_from_backend=(),  # CEAT takes embeddings, never a model name
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


def _contexts_for(word: str, sentences: List[str]) -> List[str]:
    """Sentences containing `word` (whole word, case-insensitive), capped."""
    pattern = re.compile(r"\b" + re.escape(word) + r"\b", re.IGNORECASE)
    return [s for s in sentences if pattern.search(s)][:_MAX_CONTEXTS_PER_WORD]


def _word_in_context_embeddings(model_name: str, word: str, sentences: Sequence[str]) -> np.ndarray:
    """One vector per sentence: the mean hidden state of `word`'s subword
    tokens in that sentence, from the model under evaluation.

    This is CEAT's contextual token embedding (Guo & Caliskan 2021; the
    authors' generate_ebd_*.py read the target word's layer output). The
    model is the backend's own copy (encoder.share_encoder), so nothing is
    loaded twice.
    """
    import torch

    from bias_scope.embeddings_based.encoder import _load_cls_encoder

    tokenizer, model = _load_cls_encoder(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pattern = re.compile(r"\b" + re.escape(word) + r"\b", re.IGNORECASE)
    device = next(model.parameters()).device
    vectors: List[np.ndarray] = []
    for start in range(0, len(sentences), 32):
        chunk = list(sentences[start:start + 32])
        spans = [pattern.search(s) for s in chunk]
        if any(m is None for m in spans):
            missing = chunk[[m is None for m in spans].index(True)]
            raise ValueError(f"context does not contain {word!r} as a whole word: {missing!r}")
        enc = tokenizer(chunk, padding=True, truncation=True, return_tensors="pt",
                        return_offsets_mapping=True)
        offsets = enc.pop("offset_mapping").tolist()
        with torch.no_grad():
            hidden = model(**enc.to(device)).last_hidden_state.float().cpu().numpy()
        for i, match in enumerate(spans):
            pieces = [t for t, (a, b) in enumerate(offsets[i])
                      if b > a and a < match.end() and b > match.start()]
            if not pieces:
                raise ValueError(f"tokenizer produced no token span for {word!r} in {chunk[i]!r}")
            vectors.append(hidden[i, pieces].mean(axis=0))
    return np.asarray(vectors, dtype=float)


def _embed_set(model_name: str, words: Sequence[str], sentences: List[str], set_name: str):
    """stimulus -> (n_contexts, dim) for one WEAT set; plus per-word counts."""
    group: Dict[str, np.ndarray] = {}
    counts: Dict[str, int] = {}
    for word in words:
        contexts = _contexts_for(word, sentences)
        if not contexts:
            raise ValueError(
                f"no context sentence in the corpus contains {word!r} ({set_name}); CEAT needs "
                "every stimulus of a set, so this axis cannot be built from this corpus"
            )
        group[word] = _word_in_context_embeddings(model_name, word, contexts)
        counts[word] = len(contexts)
    return group, counts


def _build_ceat_contexts(backend, metrics, axis, limit, root, allowed) -> Tuple[Dict, Dict]:
    test_path = _association_test(root, axis, _WEAT_BY_AXIS, "CEAT")
    files, sentences = _corpus(root)
    names = ("targets[0]", "targets[1]", "attributes[0]", "attributes[1]")
    built = [_embed_set(backend.model_id, words, sentences, name)
             for words, name in zip(_word_sets(test_path), names)]
    (x, cx), (y, cy), (a, ca), (b, cb) = built
    protocol = {"dataset": f"{_WIKI_DIR} + {test_path.name}", "resources": _DEVIATIONS}
    inputs = {
        name: {
            "__init__": _init_kwargs(name, backend, allowed),
            "target_embeddings": (x, y),
            "attribute_embeddings": (a, b),
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
        "contexts": {**cx, **cy, **ca, **cb},
        "embedded_by": backend.model_id,
        "embedding": _EMBEDDING_NOTE,
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
