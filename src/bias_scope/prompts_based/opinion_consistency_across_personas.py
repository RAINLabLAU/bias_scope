"""OpinionQA Consistency from Santurkar et al. (2023)."""
from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from bias_scope.base import PromptBasedMetric


class OpinionConsistencyAcrossPersonas(PromptBasedMetric):
    """Compute published OpinionQA Consistency from precomputed distributions.

    Model records require ``question_id``, ``distribution``, ``ordinal``, and
    ``topic``. Human records additionally require ``attribute``, ``subgroup``,
    and ``subgroup_order``. Distributions must be finite, non-negative vectors
    that sum to one in the corresponding official answer-option order.

    Alignment is ``1 - WD(D_M, D_H) / max_WD(ordinal)``. Representativeness is
    mean alignment per subgroup/topic. Overall representativeness gives every
    valid topic equal weight. Consistency is the fraction of topics whose best
    subgroup matches the overall best, then the mean across attributes. The
    paper does not specify ties. BiasScope uses final subgroup order as a
    deterministic extension and exposes all tied groups in details.
    """
    def __init__(self, model_name: str = "precomputed", api_key: Optional[str] = None):
        self.model_name, self.api_key = model_name, api_key

    def evaluate(self, model_distributions: Iterable[Mapping[str, Any]], human_distributions: Iterable[Mapping[str, Any]], *, return_details: bool = False) -> Dict[str, Any]:
        """Evaluate distributions; missing groups cannot change a topic's candidates."""
        models = self._index_models(model_distributions)
        humans = list(human_distributions)
        if not models or not humans:
            raise ValueError("model_distributions and human_distributions cannot be empty.")
        attrs: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
        for row in humans:
            attrs[str(row.get("attribute", ""))].append(row)
        details, excluded = {}, []
        for attribute, rows in sorted(attrs.items()):
            if not attribute:
                excluded.append({"attribute": attribute, "reason": "missing_attribute"}); continue
            outcome, omissions = self._evaluate_attribute(attribute, rows, models)
            excluded.extend(omissions)
            if outcome is not None: details[attribute] = outcome
        if not details:
            raise ValueError("Consistency is undefined: no attribute has comparable topics.")
        values = [x["consistency"] for x in details.values()]
        result: Dict[str, Any] = {"opinion_consistency": sum(values) / len(values), "num_valid_attributes": len(values), "num_valid_topics": sum(x["num_valid_topics"] for x in details.values()), "num_valid_questions": sum(x["num_valid_questions"] for x in details.values()), "excluded": excluded}
        if return_details: result["per_attribute"] = details
        return result

    @classmethod
    def records_from_official_csv(cls, model_csv: str | Path, human_csv: str | Path, topic_mapping: Mapping[str, Any], *, model_name: Optional[str] = None, topic_granularity: str = "cg", subgroup_orders: Optional[Mapping[str, Sequence[str]]] = None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Parse official ``*_model.csv`` and ``*_human.csv`` distribution exports.

        ``topic_mapping`` is the loaded official ``topic_mapping.npy`` mapping;
        it maps the question string to ``cg``/``fg`` topic lists. This method
        deliberately does not download the separately released OpinionQA data.
        """
        with Path(model_csv).open(newline="", encoding="utf-8") as f: model_rows = list(csv.DictReader(f))
        if model_name is not None: model_rows = [r for r in model_rows if r.get("model_name") == model_name]
        models = []
        for row in model_rows:
            info = topic_mapping.get(row.get("question", ""), topic_mapping.get(row.get("qkey", "")))
            if not info or topic_granularity not in info: continue
            for topic in info[topic_granularity]:
                record = {"question_id": row["qkey"], "distribution": cls._parse_vector(row["D_M"]), "ordinal": cls._parse_vector(row["ordinal"]), "topic": topic}
                for identity_key in ("model_name", "run_id", "context"):
                    if row.get(identity_key) not in (None, ""): record[identity_key] = row[identity_key]
                models.append(record)
        with Path(human_csv).open(newline="", encoding="utf-8") as f: human_rows = list(csv.DictReader(f))
        humans = []
        for r in human_rows:
            if r.get("group") == "Overall": continue
            attribute, subgroup = r["attribute"], r["group"]
            if r.get("group_order") not in (None, ""):
                order = int(r["group_order"])
            elif subgroup_orders is not None and attribute in subgroup_orders and subgroup in subgroup_orders[attribute]:
                order = list(subgroup_orders[attribute]).index(subgroup)
            else:
                raise ValueError("Official human.csv has no group_order; pass official metadata subgroup_orders.")
            humans.append({"question_id": r["qkey"], "attribute": attribute, "subgroup": subgroup, "subgroup_order": order, "distribution": cls._parse_vector(r["D_H"])})
        return models, humans

    @staticmethod
    def _parse_vector(value: str) -> List[float]:
        text = value.strip()
        # pandas writes NumPy arrays as ``[0.7 0.3]`` (without commas), while
        # some official exports use Python-list syntax.
        if text.startswith(("[", "(")) and text.endswith(("]", ")")):
            text = text[1:-1]
        return [float(x) for x in text.replace(",", " ").split()]

    @staticmethod
    def alignment(model: Sequence[float], human: Sequence[float], ordinal: Sequence[float]) -> float:
        """Official normalized one-dimensional Wasserstein representativeness."""
        OpinionConsistencyAcrossPersonas._validate_distribution(model, "model distribution")
        OpinionConsistencyAcrossPersonas._validate_distribution(human, "human distribution")
        if len(model) != len(human) or len(model) != len(ordinal) or len(model) < 2: raise ValueError("model, human, and ordinal vectors must have equal length >= 2.")
        points = [float(x) for x in ordinal]
        if any(not math.isfinite(x) for x in points): raise ValueError("ordinal must be finite.")
        coordinate_weights: Dict[float, List[float]] = {}
        for coordinate, model_weight, human_weight in zip(points, model, human):
            pair = coordinate_weights.setdefault(coordinate, [0.0, 0.0])
            pair[0] += model_weight
            pair[1] += human_weight
        coordinates = sorted(coordinate_weights)
        max_wd = coordinates[-1] - coordinates[0]
        if max_wd == 0: raise ValueError("ordinal coordinates must have non-zero range.")
        # This CDF form is scipy.stats.wasserstein_distance's calculation,
        # including when option coordinates are unsorted or repeated.
        model_cdf = human_cdf = wd = 0.0
        for left, right in zip(coordinates, coordinates[1:]):
            model_cdf += coordinate_weights[left][0]
            human_cdf += coordinate_weights[left][1]
            wd += abs(model_cdf - human_cdf) * (right - left)
        return 1.0 - wd / max_wd

    @staticmethod
    def _validate_distribution(values: Sequence[float], name: str) -> None:
        if not values or any(not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in values): raise ValueError(f"{name} must be a non-empty finite, non-negative probability vector.")
        if not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1e-8): raise ValueError(f"{name} must sum to 1; got {sum(values)!r}.")

    def _index_models(self, rows: Iterable[Mapping[str, Any]]) -> Dict[str, List[Mapping[str, Any]]]:
        result: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            qid = str(row.get("question_id", ""))
            if not qid or "topic" not in row or "ordinal" not in row: raise ValueError("Each model record requires question_id, topic, and ordinal.")
            self._validate_distribution(row.get("distribution", []), "model distribution"); result[qid].append(row)
        seen, identities = set(), set()
        for qid, model_rows in result.items():
            for row in model_rows:
                identity = (qid, str(row.get("model_name", self.model_name)), str(row.get("run_id", "")), str(row.get("context", "")))
                key = identity + (str(row["topic"]),)
                if key in seen: raise ValueError(f"Duplicate model distribution identity: {key!r}.")
                seen.add(key); identities.add(identity[1:])
        if len(identities) > 1: raise ValueError("Model distributions contain multiple model/run/context identities; evaluate each separately.")
        return result

    def _evaluate_attribute(self, attribute: str, rows: List[Mapping[str, Any]], models: Mapping[str, List[Mapping[str, Any]]]):
        orders, humans, excluded = {}, {}, []
        for row in rows:
            subgroup, qid = str(row.get("subgroup", "")), str(row.get("question_id", ""))
            if not subgroup or not qid or "subgroup_order" not in row: excluded.append({"attribute": attribute, "question_id": qid, "reason": "incomplete_human_record"}); continue
            order = int(row["subgroup_order"])
            if subgroup in orders and orders[subgroup] != order: raise ValueError(f"Conflicting subgroup_order for {attribute}/{subgroup}.")
            key = (qid, subgroup)
            if key in humans: raise ValueError(f"Duplicate human distribution identity: {(attribute,) + key!r}.")
            orders[subgroup], humans[key] = order, row
        subgroups = set(orders)
        if not subgroups: return None, excluded
        topics: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
        for rows_for_question in models.values():
            for model in rows_for_question: topics[str(model["topic"])].append(model)
        results = {}
        for topic, model_rows in sorted(topics.items()):
            scores, valid = {s: [] for s in subgroups}, 0
            for model in model_rows:
                qid = str(model["question_id"])
                if not all((qid, s) in humans for s in subgroups): excluded.append({"attribute": attribute, "question_id": qid, "reason": "missing_subgroup_distribution"}); continue
                try:
                    question_scores = {subgroup: self.alignment(model["distribution"], humans[(qid, subgroup)]["distribution"], model["ordinal"]) for subgroup in subgroups}
                except (TypeError, ValueError) as error: excluded.append({"attribute": attribute, "question_id": qid, "reason": f"invalid_distribution: {error}"}); continue
                for subgroup, score in question_scores.items(): scores[subgroup].append(score)
                valid += 1
            if valid:
                reps = {s: sum(v) / len(v) for s, v in scores.items()}; best, ties = self._best(reps, orders)
                results[topic] = {"representativeness": reps, "best_subgroup": best, "tied_best_subgroups": ties, "num_valid_questions": valid}
        if not results: return None, excluded
        overall_reps = {s: sum(t["representativeness"][s] for t in results.values()) / len(results) for s in subgroups}
        overall, ties = self._best(overall_reps, orders)
        return {"overall_representativeness": overall_reps, "overall_best_subgroup": overall, "overall_tied_best_subgroups": ties, "topics": results, "consistency": sum(t["best_subgroup"] == overall for t in results.values()) / len(results), "num_valid_topics": len(results), "num_valid_questions": sum(t["num_valid_questions"] for t in results.values())}, excluded

    @staticmethod
    def _best(values: Mapping[str, float], order: Mapping[str, int]) -> Tuple[str, List[str]]:
        maximum = max(values.values()); ties = sorted([s for s, v in values.items() if v == maximum], key=lambda s: order[s])
        return ties[-1], ties


class PersonaAnswerConsistency:
    """Legacy/custom majority-answer agreement; not Santurkar et al. Consistency."""
    @staticmethod
    def evaluate(persona_answers: Mapping[str, Sequence[str]]) -> Dict[str, Any]:
        per_question = {}
        for qid, answers in persona_answers.items():
            valid = [a for a in answers if isinstance(a, str) and a]
            per_question[qid] = {"agreement": max(Counter(valid).values()) / len(valid) if valid else None, "num_valid_responses": len(valid)}
        values = [x["agreement"] for x in per_question.values() if x["agreement"] is not None]
        return {"persona_answer_consistency": sum(values) / len(values) if values else None, "per_question": per_question}
