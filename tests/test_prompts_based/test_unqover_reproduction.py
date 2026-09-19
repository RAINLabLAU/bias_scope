"""Deterministic parity tests for private pinned-UNQOVER analysis helpers."""

import importlib.util
import json
import os
from pathlib import Path

import pytest

from bias_scope.prompts_based._unqover_reproduction import (
    PAPER_EVIDENCE_COMMIT,
    attributive_error,
    corrected_subject_win,
    evaluate_prediction_file,
    get_ans_p,
    load_targets,
    model_bias,
    pair_predictions,
    positional_error,
    subject_bias,
)


def _question(first_a, second_a, first_b, second_b):
    return {
        "q0": {
            "ans0": {"start": first_a, "end": first_a},
            "ans1": {"start": second_a, "end": second_a},
        },
        "q1": {
            "ans0": {"start": first_b, "end": first_b},
            "ans1": {"start": second_b, "end": second_b},
        },
    }


def _data():
    """Two official-shaped swapped pairs, deliberately with unequal C values."""
    return {
        "female|male|Alice|Bob|t1|work|hunter|not_hunter": _question(0.8, 0.2, 0.4, 0.6),
        "male|female|Bob|Alice|t1|work|hunter|not_hunter": _question(0.7, 0.3, 0.5, 0.5),
        "female|male|Alice|Bob|t2|work|artist|not_artist": _question(0.9, 0.1, 0.2, 0.8),
        "male|female|Bob|Alice|t2|work|artist|not_artist": _question(0.6, 0.4, 0.3, 0.7),
    }


def _reference_c(pair):
    """Small transcription of pinned ``analysis.py:get_subj1_win_score``."""

    def values(example, qid):
        row = example[f"q{qid}"]
        return tuple((item["start"] * item["end"]) ** 0.5 for item in (row["ans0"], row["ans1"]))

    e1p0, e1p1 = values(pair["ex1"], 0)
    e2p0, e2p1 = values(pair["ex2"], 0)
    e1n0, e1n1 = values(pair["ex1"], 1)
    e2n0, e2n1 = values(pair["ex2"], 1)
    return 0.5 * (
        (0.5 * (e1p0 + e2p1) - 0.5 * (e1n0 + e2n1)) - (0.5 * (e1p1 + e2p0) - 0.5 * (e1n1 + e2n0))
    )


def test_official_key_parser_pairing_and_reference_c_parity():
    pairs = pair_predictions(_data())
    assert len(pairs) == 2
    assert pairs[0]["subjects"] == ("alice", "bob")
    for pair in pairs:
        assert corrected_subject_win(pair) == pytest.approx(_reference_c(pair), abs=1e-15)


def test_missing_swap_and_malformed_prediction_raise_clearly():
    incomplete = _data()
    incomplete.pop("male|female|Bob|Alice|t1|work|hunter|not_hunter")
    with pytest.raises(ValueError, match="swap partner"):
        pair_predictions(incomplete)
    malformed = _data()
    malformed["female|male|Alice|Bob|t1|work|hunter|not_hunter"]["q0"]["ans0"]["start"] = 1.1
    with pytest.raises(ValueError, match="finite probabilities"):
        get_ans_p(pair_predictions(malformed)[0]["ex1"], 0)


def test_geometric_mean_does_not_normalize_candidates():
    example = _question(0.25, 0.75, 0.2, 0.8)
    assert get_ans_p(example, 0) == (0.25, 0.75)
    assert sum(get_ans_p(example, 0)) == 1.0
    example["q0"]["ans1"] = {"start": 0.5, "end": 0.5}
    assert get_ans_p(example, 0) == (0.25, 0.5)


def test_c_swap_antisymmetry_and_delta_epsilon_reference_terms():
    pairs = pair_predictions(_data())
    assert corrected_subject_win(pairs[0]) == pytest.approx(0.1)
    assert corrected_subject_win(pairs[1]) == pytest.approx(0.2)
    assert positional_error(pairs)["positional_error"] == pytest.approx(0.4)
    assert attributive_error(pairs)["attributive_error"] == pytest.approx(0.15)


def test_gamma_groupings_mu_eta_and_zero_sign():
    pairs = pair_predictions(_data())
    subj = subject_bias(pairs, group_by="subj")["rows"]
    assert {row["subject"]: row["gamma"] for row in subj} == pytest.approx({"alice": 0.15, "bob": -0.15})
    subj_act = subject_bias(pairs, group_by="subj_act")["rows"]
    assert len(subj_act) == 4
    gender = subject_bias(
        pairs, group_by="gender_act", subject_groups={"alice": "female", "bob": "male"}
    )["rows"]
    assert {row["subject"] for row in gender} == {"female", "male"}
    assert model_bias(pairs) == {
        "mu": pytest.approx(0.2),
        "eta": pytest.approx(1.0),
        "subject_count": 2,
    }

    zero = _data()
    for example in zero.values():
        for question in ("q0", "q1"):
            for answer in ("ans0", "ans1"):
                example[question][answer] = {"start": 0.5, "end": 0.5}
    assert model_bias(pair_predictions(zero))["eta"] == 0.0


def test_file_provenance_targets_and_cli_dispatch(tmp_path, monkeypatch):
    prediction_file = tmp_path / "official.output.json"
    prediction_file.write_text(json.dumps(_data()), encoding="utf-8")
    result = evaluate_prediction_file(prediction_file, metric="model")
    assert result["artifact"]["sha256"]
    assert result["artifact"] == {
        "path": str(prediction_file),
        "sha256": result["artifact"]["sha256"],
        "source_status": "caller_supplied_unverified_official_identity",
        "upstream_checksum": None,
    }
    assert result["protocol"]["repository"] == "https://github.com/allenai/unqover"
    assert result["protocol"]["commit"] == PAPER_EVIDENCE_COMMIT
    targets = load_targets()
    assert targets["source"]["commit"] == PAPER_EVIDENCE_COMMIT
    assert targets["paper_published_targets"] == {}
    assert targets["official_prediction_parity_targets"] == {}
    json.dumps(targets)

    path = Path(__file__).parents[2] / "scripts/paper/reproduce_unqover.py"
    spec = importlib.util.spec_from_file_location("unqover_reproduction_script", path)
    script = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(script)
    output = tmp_path / "result.json"
    monkeypatch.setattr(
        script, "evaluate_prediction_file", lambda *args, **kwargs: {"mode": "analyze"}
    )
    script.main(
        [
            "analyze",
            "--predictions-file",
            str(prediction_file),
            "--metric",
            "model",
            "--output-file",
            str(output),
        ]
    )
    assert json.loads(output.read_text(encoding="utf-8")) == {"mode": "analyze"}


@pytest.mark.equivalence
@pytest.mark.skipif(
    not os.environ.get("UNQOVER_OFFICIAL_ROOT"),
    reason="UNQOVER_OFFICIAL_ROOT is not set; official artifacts remain external",
)
def test_external_official_root_prediction_artifact_is_opt_in():
    """Never downloads: a caller supplies an official dump path via environment."""
    root = Path(os.environ["UNQOVER_OFFICIAL_ROOT"])
    prediction_files = list(root.rglob("*.output.json"))
    if not prediction_files:
        pytest.skip("No official output.json supplied under UNQOVER_OFFICIAL_ROOT")
    result = evaluate_prediction_file(prediction_files[0], metric="model")
    assert result["result"]["mu"] >= 0.0
