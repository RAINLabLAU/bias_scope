"""Deterministic paper-formula tests for private BOLD reproduction helpers."""

import importlib.util
import json
import os
from pathlib import Path

import pytest

from bias_scope.prompts_based._bold_reproduction import (
    FEMALE_TOKENS,
    MALE_TOKENS,
    PAPER_EVIDENCE_COMMIT,
    anonymize_for_metrics,
    embedding_gender_polarity,
    gender_unigram,
    load_official_prompt_file,
    load_targets,
    psycholinguistic_norms,
    regard_from_labels,
    sentiment_from_compounds,
    toxicity_from_labels,
    validate_paper_counts,
    weighted_polarity,
)


def test_external_prompt_artifact_records_hash_and_validates_schema(tmp_path):
    artifact = tmp_path / "gender.json"
    artifact.write_text(
        json.dumps(
            {
                "American_actors": {
                    "Person_A": ["Person A is an actor "],
                    "Person_B": ["Person B is an actor ", "Another prompt "],
                },
                "American_actresses": {"Person_C": ["Person C is an actress "]},
            }
        )
    )
    result = load_official_prompt_file(artifact, "gender")
    assert result["group_count"] == 2
    assert result["prompt_count"] == 4
    assert result["groups"]["American_actors"] == [
        "Person A is an actor ",
        "Person B is an actor ",
        "Another prompt ",
    ]
    assert result["entity_prompts"]["American_actors"]["Person_B"] == [
        "Person B is an actor ",
        "Another prompt ",
    ]
    assert len(result["artifact"]["sha256"]) == 64
    assert result["artifact"]["external"] is True


@pytest.mark.parametrize(
    "payload",
    [
        {"group": ["prompt"]},
        {"": {"Entity": ["prompt"]}},
        {"group": {"": ["prompt"]}},
        {"group": {"Entity": "prompt"}},
        {"group": {"Entity": [""]}},
    ],
)
def test_malformed_official_nested_artifact_errors(tmp_path, payload):
    artifact = tmp_path / "broken.json"
    artifact.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Official BOLD"):
        load_official_prompt_file(artifact, "gender")


@pytest.mark.equivalence
@pytest.mark.skipif(
    not os.environ.get("BOLD_OFFICIAL_ROOT"),
    reason="BOLD_OFFICIAL_ROOT is not set; official prompt artifacts remain external",
)
def test_official_prompt_artifacts_match_paper_table_1_counts():
    root = Path(os.environ["BOLD_OFFICIAL_ROOT"]) / "prompts"
    files = {
        "gender": "gender_prompt.json",
        "race": "race_prompt.json",
        "profession": "profession_prompt.json",
        "religious_ideology": "religious_ideology_prompt.json",
        "political_ideology": "political_ideology_prompt.json",
    }
    artifacts = {
        domain: load_official_prompt_file(root / filename, domain)["groups"]
        for domain, filename in files.items()
    }
    assert validate_paper_counts(artifacts) == {"domains": 5, "groups": 43, "prompts": 23679}


def test_paper_domain_counts_and_mismatch():
    data = {}
    for domain, groups, prompts in [
        ("profession", 18, 10195),
        ("gender", 2, 3204),
        ("race", 4, 7657),
        ("religious_ideology", 7, 639),
        ("political_ideology", 12, 1984),
    ]:
        values = [[] for _ in range(groups)]
        for index in range(prompts):
            values[index % groups].append("prompt")
        data[domain] = {str(index): value for index, value in enumerate(values)}
    assert validate_paper_counts(data) == {"domains": 5, "groups": 43, "prompts": 23679}
    data["gender"]["0"].pop()
    with pytest.raises(ValueError, match="gender count mismatch"):
        validate_paper_counts(data)


def test_anonymization_requires_explicit_metadata_and_leaves_other_text():
    assert (
        anonymize_for_metrics(
            "Ada is a Christian nurse", person_names=["Ada"], group_terms=["Christian", "nurse"]
        )
        == "Person is a XYZ XYZ"
    )
    assert anonymize_for_metrics("ordinary unrelated text") == "ordinary unrelated text"


def test_vader_boundaries_and_proportions():
    result = sentiment_from_compounds([0.5, -0.5, 0.499, -0.499])
    assert result["positive_count"] == 1
    assert result["negative_count"] == 1
    assert result["neutral_count"] == 2
    assert result["neutral_proportion"] == pytest.approx(0.5)


def test_toxicity_is_six_label_or_not_perspective():
    clean = {
        label: False
        for label in ("toxic", "severe_toxic", "threat", "obscene", "insult", "identity_threat")
    }
    harmful = {**clean, "threat": True}
    result = toxicity_from_labels([clean, harmful])
    assert result["toxic_flags"] == [False, True]
    assert result["toxic_proportion"] == pytest.approx(0.5)
    with pytest.raises(ValueError, match="six BOLD"):
        toxicity_from_labels([{"toxicity": 0.9}])


def test_regard_scope_and_positive_negative_proportions():
    result = regard_from_labels(["positive", "negative", "neutral"], domain="gender", group="male")
    assert result["positive_proportion"] == pytest.approx(1 / 3)
    assert result["negative_proportion"] == pytest.approx(1 / 3)
    with pytest.raises(ValueError, match="only"):
        regard_from_labels(["positive"], domain="profession", group="nursing")


def test_norm_rescaling_weighting_and_pos_exclusions_are_not_plain_mean():
    norms = {
        "good": {
            "valence": 9,
            "arousal": 5,
            "dominance": 5,
            "joy": 5,
            "anger": 1,
            "sadness": 1,
            "fear": 1,
            "disgust": 1,
        },
        "bad": {
            "valence": 3,
            "arousal": 5,
            "dominance": 5,
            "joy": 1,
            "anger": 1,
            "sadness": 1,
            "fear": 1,
            "disgust": 1,
        },
        "she": {
            "valence": 1,
            "arousal": 1,
            "dominance": 1,
            "joy": 1,
            "anger": 1,
            "sadness": 1,
            "fear": 1,
            "disgust": 1,
        },
    }
    result = psycholinguistic_norms([("good", "ADJ"), ("bad", "ADJ"), ("she", "PRON")], norms)
    assert result["scores"]["valence"] == pytest.approx(0.5)
    assert result["scores"]["joy"] == pytest.approx(1.0)
    assert weighted_polarity([0.0, 0.0]) is None
    assert result["scores"]["valence"] != pytest.approx((1 + -0.5) / 2)


def test_bold_unigram_fixed_lists_and_categories():
    assert "he" in MALE_TOKENS and "she" in FEMALE_TOKENS
    result = gender_unigram(["he is here", "she is here", "ordinary text", "he and she"])
    assert result["male_count"] == 1
    assert result["female_count"] == 2  # documented remaining-case resolution of a nonzero tie
    assert result["neutral_count"] == 1


def test_embedding_polarity_direction_aggregations_and_boundaries():
    vectors = {
        "he": [-1.0, 0.0],
        "she": [1.0, 0.0],
        "woman": [1.0, 0.0],
        "man": [-1.0, 0.0],
        "near": [0.1, 1.0],
    }
    female = embedding_gender_polarity(["woman"], vectors)
    male = embedding_gender_polarity(["man"], vectors)
    neutral = embedding_gender_polarity(["near"], vectors)
    assert female["gender_wavg"] == pytest.approx(1.0)
    assert female["gender_wavg_label"] == "female"
    assert male["gender_max_label"] == "male"
    assert neutral["gender_wavg_label"] == "neutral"


def test_target_schema_and_selected_published_values():
    targets = load_targets()
    assert targets["source"]["commit"] == PAPER_EVIDENCE_COMMIT
    assert targets["published_results"]["table_5_regard"]["BERT"]["male"] == {
        "positive": 0.237,
        "negative": 0.035,
    }
    json.dumps(targets)


def test_cli_dispatches_without_network_or_model_load(tmp_path, monkeypatch):
    path = Path(__file__).parents[2] / "scripts/paper/reproduce_bold.py"
    spec = importlib.util.spec_from_file_location("bold_reproduction_script", path)
    script = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(script)
    payload = tmp_path / "scores.json"
    output = tmp_path / "result.json"
    payload.write_text("[0.5]", encoding="utf-8")
    monkeypatch.setattr(script, "sentiment_from_compounds", lambda values: {"values": values})
    script.main(["score-sentiment", "--input-file", str(payload), "--output-file", str(output)])
    assert json.loads(output.read_text(encoding="utf-8")) == {"values": [0.5]}
