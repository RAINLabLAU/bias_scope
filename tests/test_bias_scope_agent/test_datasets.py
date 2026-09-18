"""Dataset providers: the loaders that keep evaluation data out of the transcript.

Every test builds its own tiny source tree under tmp_path, so nothing here
depends on third_party/ (which is git-ignored) and the assertions are about
the loader's behaviour rather than about a particular vendored file.
"""

from __future__ import annotations

import csv
import dataclasses
import json
from unittest import mock

import pytest

from bias_scope.backends import StubBackend
from bias_scope_agent.datasets import DATASETS, available_datasets, build_inputs

_ROWS = [
    {"sent_more": "gender more 1", "sent_less": "gender less 1", "bias_type": "gender"},
    {"sent_more": "race more 1", "sent_less": "race less 1", "bias_type": "race-color"},
    {"sent_more": "gender more 2", "sent_less": "gender less 2", "bias_type": "gender"},
]


@pytest.fixture
def root(tmp_path):
    crows = tmp_path / "crows-pairs" / "data"
    crows.mkdir(parents=True)
    path = crows / "crows_pairs_anonymized.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sent_more", "sent_less", "bias_type"])
        writer.writeheader()
        writer.writerows(_ROWS)

    tests = tmp_path / "sent-bias" / "tests"
    tests.mkdir(parents=True)
    payload = {
        "targ1": {"category": "MaleNames", "examples": ["john", "paul"]},
        "targ2": {"category": "FemaleNames", "examples": ["amy", "lisa"]},
        "attr1": {"category": "Career", "examples": ["office", "salary"]},
        "attr2": {"category": "Family", "examples": ["home", "children"]},
    }
    (tests / "weat6.jsonl").write_text(json.dumps(payload), encoding="utf-8")
    return tmp_path


@pytest.fixture
def backend():
    return StubBackend(access=("embeddings", "logits"), model_id="some/model")


class TestCrowsPairsProvider:
    def test_filters_to_the_requested_axis_and_keeps_file_order(self, root, backend):
        inputs, _ = build_inputs(backend, "crows_pairs", ["CrowSPairs"], axis="gender", root=root)
        assert inputs["CrowSPairs"]["sentence_pairs"] == [
            ["gender more 1", "gender less 1"],
            ["gender more 2", "gender less 2"],
        ]

    def test_limit_truncates_after_filtering_not_before(self, root, backend):
        inputs, prov = build_inputs(
            backend, "crows_pairs", ["CrowSPairs"], axis="gender", limit=1, root=root
        )
        assert inputs["CrowSPairs"]["sentence_pairs"] == [["gender more 1", "gender less 1"]]
        assert prov["pairs"] == 1

    def test_names_the_model_under_evaluation_in_the_constructor_block(self, root, backend):
        inputs, _ = build_inputs(backend, "crows_pairs", ["CrowSPairs"], axis="gender", root=root)
        assert inputs["CrowSPairs"]["__init__"]["model_name"] == "some/model"

    def test_every_served_metric_gets_the_same_pairs(self, root, backend):
        inputs, _ = build_inputs(
            backend, "crows_pairs", ["CrowSPairs", "AUL", "AULA"], axis="gender", root=root
        )
        pairs = [inputs[name]["sentence_pairs"] for name in ("CrowSPairs", "AUL", "AULA")]
        assert pairs[0] == pairs[1] == pairs[2]

    def test_provenance_carries_a_hash_of_the_file_actually_read(self, root, backend):
        _, prov = build_inputs(backend, "crows_pairs", ["CrowSPairs"], axis="gender", root=root)
        assert len(prov["sha256"]) == 64
        assert prov["dataset"] == "crows_pairs"

    def test_an_axis_the_file_does_not_contain_is_an_error(self, root, backend):
        with pytest.raises(ValueError, match="no axis"):
            build_inputs(backend, "crows_pairs", ["CrowSPairs"], axis="nonsense", root=root)

    def test_metrics_requiring_equal_length_pairs_are_not_served(self):
        # LMB and PairwiseLikelihoodPreference need both sentences of a pair to
        # tokenize to the same length (RedditBias's protocol). CrowS-Pairs pairs
        # routinely do not, so serving them this data would be the wrong dataset
        # for the metric rather than a convenience.
        served = DATASETS["crows_pairs"].metrics
        assert "LMB" not in served
        assert "PairwiseLikelihoodPreference" not in served


class TestAssociationTestProvider:
    def test_weat_returns_word_lists_for_the_metric_to_embed(self, root, backend):
        inputs, _ = build_inputs(backend, "weat", ["WEAT"], axis="gender", root=root)
        assert inputs["WEAT"]["target_embeddings"] == (["john", "paul"], ["amy", "lisa"])
        assert inputs["WEAT"]["attribute_embeddings"] == (
            ["office", "salary"],
            ["home", "children"],
        )

    def test_an_axis_no_caliskan_test_measures_is_refused_not_substituted(self, root, backend):
        with pytest.raises(ValueError, match="religion"):
            build_inputs(backend, "weat", ["WEAT"], axis="religion", root=root)

    def test_a_missing_source_file_names_the_command_that_restores_it(self, root, backend):
        with pytest.raises(ValueError, match="fetch_sources"):
            build_inputs(backend, "weat", ["WEAT"], axis="race", root=root)


class TestDatasetSelection:
    def test_unknown_dataset_lists_the_known_ones(self, root, backend):
        with pytest.raises(ValueError, match="unknown dataset"):
            build_inputs(backend, "nope", ["WEAT"], root=root)

    def test_a_dataset_is_refused_for_a_metric_it_does_not_serve(self, root, backend):
        with pytest.raises(ValueError, match="does not serve"):
            build_inputs(backend, "weat", ["CrowSPairs"], axis="gender", root=root)

    def test_available_datasets_can_be_filtered_to_one_metric(self):
        rows = available_datasets(["CrowSPairs"])
        assert [row["dataset"] for row in rows] == ["crows_pairs"]
        assert rows[0]["metrics"] == ["CrowSPairs"]


class TestConstructorInjectionIsDeclaredNotInferred:
    """Which constructor arguments come from the backend is a per-provider
    decision, never a signature guess.

    `model_name` means three different things across families: the model under
    test for CrowSPairs, the classifier Sheng et al. require for RegardScore,
    the sentence encoder for WEAT/SEAT. Inferring "it accepts model_name, so
    give it the backend's model" is the blanket injection rejected in RL-052 -
    it would silently replace RegardScore's classifier with the model being
    evaluated, which is the exact conflation the 0.2.0 fidelity audit fixed.
    A provider that serves such a metric must be able to say "fill nothing".
    """

    def test_a_provider_can_decline_to_fill_any_constructor_argument(self, root, backend):
        from bias_scope_agent import datasets

        spec = datasets.DATASETS["crows_pairs"]
        no_injection = dataclasses.replace(spec, init_from_backend=())
        with mock.patch.dict(datasets.DATASETS, {"crows_pairs": no_injection}):
            inputs, _ = build_inputs(
                backend, "crows_pairs", ["CrowSPairs"], axis="gender", root=root
            )
        assert inputs["CrowSPairs"]["__init__"] == {}

    def test_the_shipped_providers_all_evaluate_the_backend_model(self):
        from bias_scope_agent.datasets import DATASETS

        # Every current provider serves metrics whose model_name IS the model
        # under evaluation. If that ever stops being true for a new provider,
        # this test should be narrowed rather than the injection widened.
        for spec in DATASETS.values():
            assert spec.init_from_backend == ("model_name", "device"), spec.name
