"""The contexts provider for CEAT.

Guo & Caliskan 2021 sample each WEAT word's naturally occurring contexts from
a Reddit corpus and embed the word in each. That corpus is not vendored, so
this provider draws contexts from BOLD's Wikipedia sentences (vendored,
CC-BY-SA) and CEAT embeds each sentence with the model under evaluation.
Both substitutions are recorded in the protocol as deviations.
"""

from __future__ import annotations

import json

import pytest

from bias_scope.backends import StubBackend
from bias_scope_agent.datasets import DATASETS, build_inputs

_WIKI = {"Group": {"Entity": [
    "John went to the office.",
    "Amy stayed home with the children.",
    "Nothing relevant here.",
    "Paul's office was small.",
]}}
_WEAT6 = {
    "targ1": {"category": "MaleNames", "examples": ["john", "paul"]},
    "targ2": {"category": "FemaleNames", "examples": ["amy", "lisa"]},
    "attr1": {"category": "Career", "examples": ["office", "salary"]},
    "attr2": {"category": "Family", "examples": ["home", "children"]},
}


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
    return StubBackend(access=("embeddings",), model_id="stub/encoder")


class TestCeatContextsProvider:
    def test_each_set_becomes_the_sentences_containing_its_words(self, root, backend):
        inputs, _ = build_inputs(backend, "ceat_contexts", ["CEAT"], axis="gender", root=root)
        targets = inputs["CEAT"]["target_embeddings"]
        attributes = inputs["CEAT"]["attribute_embeddings"]
        assert targets[0] == ["John went to the office.", "Paul's office was small."]
        assert targets[1] == ["Amy stayed home with the children."]
        assert attributes[0] == ["John went to the office.", "Paul's office was small."]
        # one context per (word, sentence) pair: "home" and "children" share one
        assert attributes[1] == ["Amy stayed home with the children."] * 2

    def test_matching_is_whole_word_and_case_insensitive(self, root, backend):
        # "Paul's" matches paul; "John" matches john; nothing matches "salary"
        _, prov = build_inputs(backend, "ceat_contexts", ["CEAT"], axis="gender", root=root)
        assert set(prov["dropped_words"]) == {"lisa", "salary"}

    def test_sample_count_follows_the_paper_not_the_metric_default(self, root, backend):
        inputs, prov = build_inputs(backend, "ceat_contexts", ["CEAT"], axis="gender", root=root)
        assert inputs["CEAT"]["n_samples"] == 1000                 # Guo & Caliskan report N=1,000
        assert inputs["CEAT"]["random_seed"] == 42
        assert prov["n_samples"] == 1000

    def test_the_substitutions_are_recorded_deviations(self, root, backend):
        inputs, prov = build_inputs(backend, "ceat_contexts", ["CEAT"], axis="gender", root=root)
        deviations = [r["deviation"] for r in inputs["CEAT"]["__protocol__"]["resources"]]
        assert any("Reddit" in d for d in deviations)               # substitute corpus
        assert any("sentence" in d.lower() for d in deviations)     # sentence-level pooling
        assert prov["corpus"]["sentences"] == 4
        assert len(prov["corpus"]["files"]) == 1

    def test_contexts_per_word_are_capped(self, root, backend, monkeypatch):
        from bias_scope_agent import datasets_ceat

        monkeypatch.setattr(datasets_ceat, "_MAX_CONTEXTS_PER_WORD", 1)
        inputs, prov = build_inputs(backend, "ceat_contexts", ["CEAT"], axis="gender", root=root)
        assert inputs["CEAT"]["attribute_embeddings"][0] == ["John went to the office."]
        assert prov["max_contexts_per_word"] == 1

    def test_ceat_embeds_with_the_model_under_evaluation(self, root, backend):
        inputs, _ = build_inputs(backend, "ceat_contexts", ["CEAT"], axis="gender", root=root)
        assert inputs["CEAT"]["__init__"]["model_name"] == "stub/encoder"
        assert DATASETS["ceat_contexts"].axes == ("gender", "race", "age")

    def test_an_axis_no_caliskan_test_measures_is_refused(self, root, backend):
        with pytest.raises(ValueError, match="religion"):
            build_inputs(backend, "ceat_contexts", ["CEAT"], axis="religion", root=root)
