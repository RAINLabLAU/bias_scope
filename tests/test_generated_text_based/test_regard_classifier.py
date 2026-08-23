"""RegardScore must measure regard, not sentiment (Sheng et al. 2019).

The paper's whole contribution is that the two differ. Its Table 2 gives
counterexamples:

    "XYZ was a pimp and her friend was happy."      sentiment +, regard −
    "XYZ, known for his kindness, had died alone."  sentiment −, regard +

v0.1.1 defaulted to `siebert/sentiment-roberta-large-english`, a sentiment
classifier, and described itself as measuring "sentiment differences". See
docs/fidelity/regard_score.md.
"""

from unittest.mock import patch

import pytest

from bias_scope.generated_text_based.regard_score import (
    DEFAULT_REGARD_MODEL,
    REGARD_LABELS,
    RegardScore,
)


def fake_pipeline(labels_by_text):
    """A pipeline stand-in returning a fixed label per text."""

    def _pipe(texts, **_):
        return [[{"label": labels_by_text[t], "score": 1.0}] for t in texts]

    return _pipe


class TestDefaultClassifier:
    def test_the_default_is_a_regard_classifier_not_a_sentiment_one(self):
        assert "regard" in DEFAULT_REGARD_MODEL.lower()
        assert "sentiment" not in DEFAULT_REGARD_MODEL.lower()

    def test_the_four_regard_labels_are_declared(self):
        assert REGARD_LABELS == ("negative", "neutral", "positive", "other")


class TestLabelMapping:
    def _metric(self, mapping):
        with patch("bias_scope.generated_text_based.regard_score.pipeline") as p:
            p.return_value = fake_pipeline(mapping)
            return RegardScore()

    def test_regardv3_labels_map_to_the_four_buckets(self):
        """sasha/regardv3: LABEL_0..3 = negative, neutral, positive, other."""
        metric = self._metric({})
        assert metric._normalize_label("LABEL_0") == "negative"
        assert metric._normalize_label("LABEL_1") == "neutral"
        assert metric._normalize_label("LABEL_2") == "positive"
        assert metric._normalize_label("LABEL_3") == "other"

    def test_other_is_not_folded_into_neutral(self):
        """v0.1.1 mapped everything unrecognised to neutral, losing `other`.

        `other` means the classifier could not place the text on the regard
        scale at all — which is different from placing it in the middle.
        """
        metric = self._metric({})
        assert metric._normalize_label("LABEL_3") != "neutral"

    def test_named_labels_still_work(self):
        metric = self._metric({})
        assert metric._normalize_label("positive") == "positive"
        assert metric._normalize_label("NEGATIVE") == "negative"
        assert metric._normalize_label("other") == "other"


class TestDistribution:
    def test_distribution_covers_all_four_labels(self):
        texts = {"a": "LABEL_0", "b": "LABEL_2", "c": "LABEL_3", "d": "LABEL_1"}
        with patch("bias_scope.generated_text_based.regard_score.pipeline") as p:
            p.return_value = fake_pipeline(texts)
            metric = RegardScore()
            result = metric.evaluate([["a", "b"]], [["c", "d"]], return_details=True)

        for label in REGARD_LABELS:
            assert f"group_a_{label}" in result
            assert f"group_b_{label}" in result

    def test_identical_groups_give_zero_differences(self):
        texts = {"a": "LABEL_0", "b": "LABEL_0"}
        with patch("bias_scope.generated_text_based.regard_score.pipeline") as p:
            p.return_value = fake_pipeline(texts)
            result = RegardScore().evaluate([["a"]], [["b"]], return_details=True)
        for label in REGARD_LABELS:
            assert result[f"{label}_diff"] == pytest.approx(0.0)

    def test_a_fully_negative_vs_fully_positive_split(self):
        texts = {"a": "LABEL_0", "b": "LABEL_2"}
        with patch("bias_scope.generated_text_based.regard_score.pipeline") as p:
            p.return_value = fake_pipeline(texts)
            result = RegardScore().evaluate([["a"]], [["b"]], return_details=True)
        assert result["negative_diff"] == pytest.approx(1.0)
        assert result["positive_diff"] == pytest.approx(-1.0)
