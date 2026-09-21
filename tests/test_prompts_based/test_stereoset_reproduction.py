"""Deterministic parity tests for the private paper-era StereoSet evaluator."""

import importlib.util
import json
import os
from pathlib import Path

import pytest

from bias_scope.prompts_based._stereoset_reproduction import (
    PAPER_EVIDENCE_COMMIT,
    bert_intrasentence_score,
    bert_nsp_score,
    evaluate_official,
    load_targets,
    score_examples,
)


def cluster(identifier, target, domain, pro, anti, unrelated):
    return {
        "id": identifier,
        "target": target,
        "bias_type": domain,
        "context": "A BLANK context.",
        "sentences": [
            {"id": pro, "gold_label": "stereotype", "sentence": "pro"},
            {"id": anti, "gold_label": "anti-stereotype", "sentence": "anti"},
            {"id": unrelated, "gold_label": "unrelated", "sentence": "other"},
        ],
    }


def test_lms_counts_both_meaningful_comparisons_and_ss_is_strict():
    row = cluster("one", "target", "gender", "p", "a", "u")
    result = score_examples([row], {"p": 0.8, "a": 0.3, "u": 0.5})
    assert result["LM Score"] == 50.0
    assert result["SS Score"] == 100.0


def test_ss_opposite_and_tie_match_official_else_branch():
    row = cluster("one", "target", "gender", "p", "a", "u")
    assert score_examples([row], {"p": 0.1, "a": 0.8, "u": 0.0})["SS Score"] == 0.0
    tied = score_examples([row], {"p": 0.5, "a": 0.5, "u": 0.0})
    assert tied["SS Score"] == 0.0
    assert tied["LM Score"] == 100.0


def test_icat_and_target_term_macro_aggregation_are_not_flat_micro():
    rows = [cluster(str(i), "frequent", "gender", f"p{i}", f"a{i}", f"u{i}") for i in range(3)]
    rows.append(cluster("rare", "rare", "gender", "pr", "ar", "ur"))
    scores = {key: 0.0 for row in rows for key in [s["id"] for s in row["sentences"]]}
    for row in rows[:3]:
        scores[row["sentences"][0]["id"]] = 0.9
        scores[row["sentences"][1]["id"]] = 0.8
    scores["pr"], scores["ar"], scores["ur"] = 0.2, 0.9, 0.1
    result = score_examples(rows, scores)
    assert result["SS Score"] == 50.0  # macro: (100 + 0) / 2, not 75
    assert result["LM Score"] == 100.0
    assert result["ICAT Score"] == 100.0


def test_domain_task_and_overall_organization_and_id_mapping():
    intra = cluster("i", "term-i", "gender", "ip", "ia", "iu")
    inter = cluster("e", "term-e", "race", "ep", "ea", "eu")
    gold = {"data": {"intrasentence": [intra], "intersentence": [inter]}}
    predictions = {
        "intrasentence": [
            {"id": "ip", "score": 0.9},
            {"id": "ia", "score": 0.8},
            {"id": "iu", "score": 0.1},
        ],
        "intersentence": [
            {"id": "ep", "score": 0.1},
            {"id": "ea", "score": 0.9},
            {"id": "eu", "score": 0.2},
        ],
    }
    result = evaluate_official(gold, predictions)
    assert result["intrasentence"]["gender"]["SS Score"] == 100.0
    assert result["intersentence"]["race"]["SS Score"] == 0.0
    assert result["overall"]["SS Score"] == 50.0


def test_missing_and_malformed_predictions_error():
    row = cluster("one", "target", "gender", "p", "a", "u")
    gold = {"data": {"intrasentence": [row], "intersentence": []}}
    with pytest.raises(KeyError, match="Missing official"):
        evaluate_official(gold, {"intrasentence": [{"id": "p", "score": 1.0}]})
    with pytest.raises(ValueError, match="id and score"):
        evaluate_official(gold, {"intrasentence": [{"id": "p"}]})


def test_target_fixture_distinguishes_paper_test_and_official_dev():
    targets = load_targets()
    assert targets["source"]["commit"] == PAPER_EVIDENCE_COMMIT
    assert targets["paper_test_likelihood_targets"]["BERT-base"] == {
        "lms": 85.4,
        "ss": 58.3,
        "icat": 71.2,
    }
    assert (
        "predictions_bert-base-cased_BertNextSentence_BertLM.json"
        in targets["official_dev_parity_targets"]
    )
    json.dumps(targets)


def test_reproduction_cli_dispatches_without_loading_a_model(tmp_path, monkeypatch):
    path = Path(__file__).parents[2] / "scripts/paper/reproduce_stereoset.py"
    spec = importlib.util.spec_from_file_location("stereoset_reproduction_script", path)
    script = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(script)
    output = tmp_path / "output.json"
    monkeypatch.setattr(
        script, "evaluate_official_files", lambda gold, predictions: {"mode": "parity"}
    )
    script.main(
        [
            "official-parity",
            "--gold-file",
            "gold.json",
            "--predictions-file",
            "pred.json",
            "--output-file",
            str(output),
        ]
    )
    assert json.loads(output.read_text()) == {"mode": "parity"}

    gold = tmp_path / "gold.json"
    gold.write_text("{}", encoding="utf-8")
    observed = {}
    monkeypatch.setattr(
        script,
        "reconstruct_bert_base",
        lambda payload, device, model_id: (
            observed.update(payload=payload, device=device, model_id=model_id)
            or {
                "runtime": {
                    "classification": "historical_local_reconstruction_not_exact_paper_reproduction"
                }
            }
        ),
    )
    script.main(["bert-base", "--gold-file", str(gold), "--output-file", str(output)])
    assert observed == {"payload": {}, "device": "cpu", "model_id": "bert-base-cased"}
    assert json.loads(output.read_text())["runtime"]["classification"].startswith("historical_")


def test_bert_scorers_use_left_to_right_masks_and_nsp_positive_class():
    torch = pytest.importorskip("torch")

    class Tokenizer:
        mask_token, mask_token_id, cls_token, sep_token = "[MASK]", 99, "[CLS]", "[SEP]"
        masked_texts = []

        def encode(self, text, add_special_tokens=False):
            return {"multi": [1, 2]}.get(text, [10, 99, 11])

        def decode(self, ids):
            return "" if not ids else "prefix"

        def encode_plus(self, text, **kwargs):
            self.masked_texts.append(text)
            return {"input_ids": torch.tensor([[10, 99, 11]])}

        def tokenize(self, text):
            return text.split()

        def convert_tokens_to_ids(self, tokens):
            return list(range(1, len(tokens) + 1))

    class MLM:
        def __call__(self, **kwargs):
            logits = torch.zeros((1, 3, 100))
            logits[0, 1, 1] = 2
            logits[0, 1, 2] = 4
            return type("Output", (), {"logits": logits})()

    class NSP:
        def __call__(self, **kwargs):
            assert kwargs["token_type_ids"].tolist() == [[0, 0, 0, 1, 1]]
            return type("Output", (), {"logits": torch.tensor([[2.0, 0.0]])})()

    tokenizer = Tokenizer()
    score = bert_intrasentence_score(MLM(), tokenizer, "A BLANK context.", "multi")
    assert 0 < score < 1
    assert tokenizer.masked_texts == ["A [MASK] context.", "A prefix[MASK] context."]
    assert bert_nsp_score(NSP(), tokenizer, "first", "second") == pytest.approx(0.88079708)


@pytest.mark.equivalence
@pytest.mark.skipif(
    not os.environ.get("STEREOSET_OFFICIAL_ROOT"),
    reason="STEREOSET_OFFICIAL_ROOT is not set; official artifacts remain external",
)
def test_official_dev_bert_base_prediction_parity():
    """Opt-in: pinned-repo dev.json plus supplied BERT-base predictions."""
    from bias_scope.prompts_based._stereoset_reproduction import evaluate_official_files

    root = Path(os.environ["STEREOSET_OFFICIAL_ROOT"])
    prediction_name = "predictions_bert-base-cased_BertNextSentence_BertLM.json"
    result = evaluate_official_files(
        root / "data/dev.json", root / f"code/predictions/{prediction_name}"
    )
    expected = load_targets()["official_dev_parity_targets"][prediction_name]
    for section, values in expected.items():
        task = section.removesuffix("_overall") if section != "overall" else "overall"
        actual = result[task]["overall"] if task != "overall" else result["overall"]
        assert actual["LM Score"] == pytest.approx(values["lms"], abs=1e-12)
        assert actual["SS Score"] == pytest.approx(values["ss"], abs=1e-12)
        assert actual["ICAT Score"] == pytest.approx(values["icat"], abs=1e-12)
