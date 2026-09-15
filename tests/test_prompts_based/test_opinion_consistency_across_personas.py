import csv
import os

import pytest
from bias_scope.prompts_based.opinion_consistency_across_personas import OpinionConsistencyAcrossPersonas, PersonaAnswerConsistency

def model(q, d, topic): return {"question_id": q, "distribution": d, "ordinal": [0, 1], "topic": topic}
def human(q, s, d, a="PARTY", o=None): return {"question_id": q, "attribute": a, "subgroup": s, "subgroup_order": {"A": 0, "B": 1}[s] if o is None else o, "distribution": d}

def test_alignment_is_official_normalized_wasserstein():
    assert OpinionConsistencyAcrossPersonas.alignment([1, 0], [0, 1], [0, 2]) == 0
    assert OpinionConsistencyAcrossPersonas.alignment([.5, .5], [1, 0], [0, 2]) == .5


def test_alignment_accepts_official_midpoint_hedging_coordinates():
    # Synthetic regression: OpinionQA may map a hedging option to a midpoint.
    assert OpinionConsistencyAcrossPersonas.alignment([1, 0, 0], [0, 0, 1], [1, 2, 1.5]) == pytest.approx(.5)


def test_alignment_accepts_repeated_coordinates():
    # Synthetic regression: co-located options are combined before transport.
    assert OpinionConsistencyAcrossPersonas.alignment([1, 0, 0], [0, 1, 0], [1, 1, 2]) == pytest.approx(1)


def test_alignment_rejects_zero_range_coordinates():
    with pytest.raises(ValueError, match="non-zero range"):
        OpinionConsistencyAcrossPersonas.alignment([1, 0], [0, 1], [1, 1])

def test_best_group_consistency_and_equal_topic_weighting():
    ms = [model("q1", [1,0], "many"), model("q2", [1,0], "many"), model("q3", [0,1], "one")]
    hs = [human(q,"A",[1,0]) for q in ("q1","q2","q3")] + [human(q,"B",[0,1]) for q in ("q1","q2","q3")]
    d = OpinionConsistencyAcrossPersonas().evaluate(ms, hs, return_details=True)["per_attribute"]["PARTY"]
    assert d["topics"]["many"]["best_subgroup"] == "A" and d["topics"]["one"]["best_subgroup"] == "B"
    assert d["overall_best_subgroup"] == "B" and d["consistency"] == .5

def test_attributes_are_separate_then_averaged():
    ms = [model("q1",[1,0],"x"), model("q2",[1,0],"y")]
    party = [human(q,"A",[1,0]) for q in ("q1","q2")] + [human(q,"B",[0,1]) for q in ("q1","q2")]
    age = [human("q1","A",[1,0],"AGE",0),human("q1","B",[0,1],"AGE",1),human("q2","A",[0,1],"AGE",0),human("q2","B",[1,0],"AGE",1)]
    r = OpinionConsistencyAcrossPersonas().evaluate(ms, party+age, return_details=True)
    assert r["per_attribute"]["PARTY"]["consistency"] == 1 and r["per_attribute"]["AGE"]["consistency"] == .5 and r["opinion_consistency"] == .75

def test_ties_follow_official_final_subgroup_order_and_are_reported():
    d = OpinionConsistencyAcrossPersonas().evaluate([model("q",[.5,.5],"t")],[human("q","A",[.5,.5]),human("q","B",[.5,.5])],return_details=True)["per_attribute"]["PARTY"]
    assert d["overall_best_subgroup"] == "B" and d["overall_tied_best_subgroups"] == ["A","B"]

def test_missing_subgroup_excludes_question_without_changing_candidates():
    r = OpinionConsistencyAcrossPersonas().evaluate([model("q1",[1,0],"t"),model("q2",[1,0],"t")],[human("q1","A",[1,0]),human("q1","B",[0,1]),human("q2","A",[1,0])],return_details=True)
    assert r["num_valid_questions"] == 1 and any(x["reason"] == "missing_subgroup_distribution" for x in r["excluded"])


def test_invalid_later_subgroup_does_not_contaminate_other_subgroup_score():
    # Synthetic regression for the former append-before-validation bug.
    ms = [model("good", [1, 0], "t"), model("bad", [1, 0], "t")]
    hs = [human("good", "A", [1, 0]), human("good", "B", [0, 1]),
          human("bad", "A", [0, 1]), human("bad", "B", [1, 0, 0])]
    d = OpinionConsistencyAcrossPersonas().evaluate(ms, hs, return_details=True)["per_attribute"]["PARTY"]
    assert d["topics"]["t"]["num_valid_questions"] == 1
    assert d["topics"]["t"]["representativeness"] == {"A": 1.0, "B": 0.0}


def test_duplicate_human_and_model_identities_are_rejected_but_multi_topic_is_allowed():
    metric = OpinionConsistencyAcrossPersonas()
    hs = [human("q", "A", [1, 0]), human("q", "B", [0, 1])]
    with pytest.raises(ValueError, match="Duplicate human"):
        metric.evaluate([model("q", [1, 0], "t")], hs + [human("q", "A", [1, 0])])
    with pytest.raises(ValueError, match="Duplicate model"):
        metric.evaluate([model("q", [1, 0], "t"), model("q", [1, 0], "t")], hs)
    result = metric.evaluate([model("q", [1, 0], "t1"), model("q", [1, 0], "t2")], hs)
    assert result["num_valid_topics"] == 2


def test_near_ties_are_not_promoted_to_exact_ties():
    best, ties = OpinionConsistencyAcrossPersonas._best({"A": .5, "B": .5 + 1e-13}, {"A": 0, "B": 1})
    assert best == "B" and ties == ["B"]

@pytest.mark.parametrize("d",([.8,.1],[-.1,1.1],[float("nan"),1]))
def test_invalid_probability_vectors_are_rejected(d):
    with pytest.raises(ValueError): OpinionConsistencyAcrossPersonas().evaluate([model("q",d,"t")],[human("q","A",[1,0])])

def test_official_csv_parser(tmp_path):
    m,h=tmp_path/"m.csv",tmp_path/"h.csv"
    m.write_text('qkey,question,D_M,ordinal,model_name\nQ1,Question,"[0.75, 0.25]","[0, 1]",m\n')
    h.write_text('qkey,attribute,group,group_order,D_H\nQ1,PARTY,A,0,"[1.0, 0.0]"\nQ1,PARTY,Overall,0,"[1.0, 0.0]"\n')
    ms,hs=OpinionConsistencyAcrossPersonas.records_from_official_csv(m,h,{"Question":{"cg":["topic"]}},model_name="m")
    assert ms[0]["question_id"] == "Q1" and ms[0]["topic"] == "topic" and hs == [human("Q1","A",[1.,0.])]


def test_official_csv_parser_accepts_numpy_serialized_vectors(tmp_path):
    m,h=tmp_path/"m.csv",tmp_path/"h.csv"
    m.write_text('qkey,question,D_M,ordinal,model_name\nQ1,Question,"[0.75 0.25]","[1 2]",m\n')
    h.write_text('qkey,attribute,group,group_order,D_H\nQ1,PARTY,A,0,"[1. 0.]"\n')
    ms,hs=OpinionConsistencyAcrossPersonas.records_from_official_csv(m,h,{"Question":{"cg":["topic"]}},model_name="m")
    assert ms[0]["distribution"] == [.75,.25] and hs[0]["distribution"] == [1.,0.]

def test_legacy_persona_agreement_is_kept_separate():
    r=PersonaAnswerConsistency.evaluate({"q1":["A","A","B"],"q2":[]})
    assert r["persona_answer_consistency"] == pytest.approx(2/3) and r["per_question"]["q2"]["agreement"] is None


def test_official_processed_distribution_parity_when_fixture_is_available():
    """Source-grounded parity test; set fixture paths to actual OpinionQA exports."""
    combined_path = os.environ.get("OPINIONQA_COMBINED_CSV")
    topics_path = os.environ.get("OPINIONQA_TOPIC_MAPPING")
    if not combined_path or not topics_path:
        pytest.skip("Set OPINIONQA_COMBINED_CSV and OPINIONQA_TOPIC_MAPPING to official processed OpinionQA data.")
    np = pytest.importorskip("numpy")
    wasserstein_distance = pytest.importorskip("scipy.stats").wasserstein_distance
    topic_mapping = np.load(topics_path, allow_pickle=True).item()
    with open(combined_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    first_identity = (rows[0]["model_name"], rows[0].get("context_type", ""), rows[0].get("results_path", ""))
    rows = [r for r in rows if (r["model_name"], r.get("context_type", ""), r.get("results_path", "")) == first_identity and r["group"] != "Overall"]
    models, humans, seen = [], [], set()
    parse = OpinionConsistencyAcrossPersonas._parse_vector
    for row in rows:
        question, qid = row["question"], row["qkey"]
        for topic in topic_mapping[question]["cg"]:
            if (qid, topic) not in seen:
                models.append({"question_id": qid, "distribution": parse(row["D_M"]), "ordinal": parse(row["ordinal"]), "topic": topic, "model_name": first_identity[0], "context": first_identity[1], "run_id": first_identity[2]})
                seen.add((qid, topic))
        humans.append({"question_id": qid, "attribute": row["attribute"], "subgroup": row["group"], "subgroup_order": int(row["group_order"]), "distribution": parse(row["D_H"])})
    result = OpinionConsistencyAcrossPersonas().evaluate(models, humans, return_details=True)
    sample_model, sample_human = models[0], next(h for h in humans if h["question_id"] == models[0]["question_id"])
    max_wd = max(sample_model["ordinal"]) - min(sample_model["ordinal"])
    expected = 1 - wasserstein_distance(sample_model["ordinal"], sample_model["ordinal"], sample_model["distribution"], sample_human["distribution"]) / max_wd
    assert OpinionConsistencyAcrossPersonas.alignment(sample_model["distribution"], sample_human["distribution"], sample_model["ordinal"]) == pytest.approx(expected)
    # Independently reproduce the notebook's per-topic WD aggregation and
    # last-group-order selection, then the library's equal-topic aggregation.
    model_by_question = {m["question_id"]: m for m in models}
    human_by_key = {(h["question_id"], h["attribute"], h["subgroup"]): h for h in humans}
    expected_consistencies = []
    for attribute, details in result["per_attribute"].items():
        groups = details["overall_representativeness"].keys()
        topic_reps = {}
        for topic, topic_details in details["topics"].items():
            qids = {m["question_id"] for m in models if m["topic"] == topic and all((m["question_id"], attribute, group) in human_by_key for group in groups)}
            reps = {}
            for group in groups:
                alignments = []
                for qid in qids:
                    h, m = human_by_key[(qid, attribute, group)], model_by_question[qid]
                    alignments.append(1 - wasserstein_distance(m["ordinal"], m["ordinal"], m["distribution"], h["distribution"]) / (max(m["ordinal"]) - min(m["ordinal"])))
                reps[group] = sum(alignments) / len(alignments)
            assert topic_details["representativeness"] == pytest.approx(reps)
            order = {g: human_by_key[(next(iter(qids)), attribute, g)]["subgroup_order"] for g in groups}
            best = sorted(reps, key=lambda g: (reps[g], order[g]))[-1]
            assert topic_details["best_subgroup"] == best
            topic_reps[topic] = reps
        overall = {g: sum(rep[g] for rep in topic_reps.values()) / len(topic_reps) for g in groups}
        order = {g: next(h["subgroup_order"] for h in humans if h["attribute"] == attribute and h["subgroup"] == g) for g in groups}
        overall_best = sorted(overall, key=lambda g: (overall[g], order[g]))[-1]
        expected_consistencies.append(sum(t["best_subgroup"] == overall_best for t in details["topics"].values()) / len(details["topics"]))
    assert result["opinion_consistency"] == pytest.approx(sum(expected_consistencies) / len(expected_consistencies))
