"""The contexts provider for CEAT.

Guo & Caliskan 2021 sample each WEAT word's naturally occurring contexts from
a Reddit corpus and take the word's own contextual token embedding in each.
That corpus is not vendored, so this provider draws contexts from BOLD's
Wikipedia sentences (vendored, CC-BY-SA) - recorded as a deviation - and
embeds each word *in* its context with the model under evaluation, which is
what CEAT (as audited in 2026-09) takes: a mapping stimulus -> matrix of
contextual token embeddings.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from bias_scope.backends import HuggingFaceBackend
from bias_scope_agent.datasets import DATASETS, build_inputs
from bias_scope_agent.datasets_ceat import _word_in_context_embeddings

_WIKI = {"Group": {"Entity": [
    "John went to the office.",
    "Amy stayed home with the children.",
    "Nothing relevant here.",
    "Paul's office was small.",
    "Lisa runs the salary office.",
]}}
_WEAT6 = {
    "targ1": {"category": "MaleNames", "examples": ["john", "paul"]},
    "targ2": {"category": "FemaleNames", "examples": ["amy", "lisa"]},
    "attr1": {"category": "Career", "examples": ["office", "salary"]},
    "attr2": {"category": "Family", "examples": ["home", "children"]},
}
_TINY = "sshleifer/tiny-gpt2"


@pytest.fixture
def root(tmp_path):
    wiki = tmp_path / "bold" / "wikipedia"
    wiki.mkdir(parents=True)
    (wiki / "gender_wiki.json").write_text(json.dumps(_WIKI), encoding="utf-8")
    tests = tmp_path / "sent-bias" / "tests"
    tests.mkdir(parents=True)
    (tests / "weat6.jsonl").write_text(json.dumps(_WEAT6), encoding="utf-8")
    return tmp_path


@pytest.fixture
def backend():
    return HuggingFaceBackend(_TINY, kind="causal", dtype="fp32")


class TestWordInContextEmbeddings:
    def test_one_vector_per_context_with_the_models_width(self, backend):
        vectors = _word_in_context_embeddings(_TINY, "office", ["John went to the office.",
                                                                "Paul's office was small."])
        assert vectors.shape == (2, backend._shared_encoder()[1].config.hidden_size)
        assert np.all(np.isfinite(vectors))

    def test_the_vector_is_the_words_own_tokens_not_the_sentence(self, backend):
        # Same word, two contexts: contextual embeddings differ; and the
        # embedding of a *different* word in the same sentence differs too.
        office = _word_in_context_embeddings(_TINY, "office", ["John went to the office."])
        john = _word_in_context_embeddings(_TINY, "john", ["John went to the office."])
        assert not np.allclose(office, john)

    def test_matching_is_whole_word_and_case_insensitive(self, backend):
        with pytest.raises(ValueError, match="does not contain"):
            _word_in_context_embeddings(_TINY, "off", ["John went to the office."])


class TestCeatContextsProvider:
    def test_each_set_maps_each_word_to_its_context_embeddings(self, root, backend):
        inputs, prov = build_inputs(backend, "ceat_contexts", ["CEAT"], axis="gender", root=root)
        targets = inputs["CEAT"]["target_embeddings"]
        assert set(targets[0]) == {"john", "paul"} and set(targets[1]) == {"amy", "lisa"}
        assert targets[0]["john"].shape[0] == 1 and targets[0]["paul"].shape[0] == 1
        attributes = inputs["CEAT"]["attribute_embeddings"]
        assert attributes[0]["office"].shape[0] == 3          # three sentences contain it
        assert prov["contexts"]["office"] == 3

    def test_ceat_no_longer_receives_a_model_name_or_pooling(self, root, backend):
        inputs, _ = build_inputs(backend, "ceat_contexts", ["CEAT"], axis="gender", root=root)
        assert inputs["CEAT"]["__init__"] == {}
        assert DATASETS["ceat_contexts"].init_from_backend == ()

    def test_sample_count_follows_the_paper(self, root, backend):
        inputs, prov = build_inputs(backend, "ceat_contexts", ["CEAT"], axis="gender", root=root)
        assert inputs["CEAT"]["n_samples"] == 1000
        assert inputs["CEAT"]["random_seed"] == 42
        assert prov["n_samples"] == 1000

    def test_only_the_corpus_substitution_remains_a_deviation(self, root, backend):
        inputs, prov = build_inputs(backend, "ceat_contexts", ["CEAT"], axis="gender", root=root)
        deviations = [r["deviation"] for r in inputs["CEAT"]["__protocol__"]["resources"]]
        assert len(deviations) == 1 and "Reddit" in deviations[0]
        assert prov["embedding"].startswith("each word's own contextual token embedding")

    def test_a_word_with_no_context_is_an_error_naming_it(self, root, backend, tmp_path):
        weat = {**_WEAT6, "attr1": {"category": "Career", "examples": ["office", "boardroom"]}}
        path = root / "sent-bias" / "tests" / "weat6.jsonl"
        path.write_text(json.dumps(weat), encoding="utf-8")
        with pytest.raises(ValueError, match="boardroom"):
            build_inputs(backend, "ceat_contexts", ["CEAT"], axis="gender", root=root)

    def test_contexts_per_word_are_capped(self, root, backend, monkeypatch):
        from bias_scope_agent import datasets_ceat

        monkeypatch.setattr(datasets_ceat, "_MAX_CONTEXTS_PER_WORD", 1)
        inputs, prov = build_inputs(backend, "ceat_contexts", ["CEAT"], axis="gender", root=root)
        assert inputs["CEAT"]["attribute_embeddings"][0]["office"].shape[0] == 1
        assert prov["max_contexts_per_word"] == 1

    def test_an_axis_no_caliskan_test_measures_is_refused(self, root, backend):
        with pytest.raises(ValueError, match="religion"):
            build_inputs(backend, "ceat_contexts", ["CEAT"], axis="religion", root=root)
