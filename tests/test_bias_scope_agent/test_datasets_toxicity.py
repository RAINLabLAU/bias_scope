"""The RealToxicityPrompts provider for EMT.

Gehman et al. 2020 score K=25 nucleus-sampled continuations per prompt with
the Perspective API. No key is available here, so the provider scores with a
local classifier and must say so in the protocol: the metric class is
`faithful`, the run is an adaptation, and the badge cannot tell the reader.
The dataset and the classifier are monkeypatched here; nothing downloads.
"""

from __future__ import annotations

import pytest

from bias_scope.backends import StubBackend
from bias_scope_agent import datasets_common, datasets_toxicity
from bias_scope_agent.datasets import DATASETS, build_inputs


@pytest.fixture
def offline(tmp_path, monkeypatch):
    monkeypatch.setattr(datasets_common, "_GENERATION_CACHE", tmp_path / "cache")
    monkeypatch.setattr(
        datasets_toxicity, "_rtp_prompts", lambda limit: ("rev123", ["So then he", "The"][:limit])
    )
    monkeypatch.setattr(
        datasets_toxicity, "_toxicity_scores",
        lambda texts: [0.9 if "idiot" in t else 0.1 for t in texts],
    )
    return tmp_path


@pytest.fixture
def causal():
    return StubBackend(
        answers=["you idiot", "said hello"], access=("embeddings", "completions"),
        model_id="stub/causal",
    )


class TestRtpToxicityProvider:
    def test_k_sampled_continuations_per_prompt(self, offline, causal):
        inputs, prov = build_inputs(causal, "rtp_toxicity", ["EMT"], root=offline)
        completions = inputs["EMT"]["completions"]
        assert len(completions) == 2 and all(len(c) == 25 for c in completions)
        assert prov["k"] == 25                                   # Gehman Sec. 3.2
        assert prov["decoding"] == {"max_new_tokens": 20, "do_sample": True, "top_p": 0.9}
        assert len(causal.calls) == 50

    def test_scores_are_aligned_with_completions(self, offline, causal):
        inputs, _ = build_inputs(causal, "rtp_toxicity", ["EMT"], root=offline)
        for texts, scores in zip(inputs["EMT"]["completions"], inputs["EMT"]["toxicity_scores"]):
            assert len(texts) == len(scores)
            for text, score in zip(texts, scores):
                assert score == (0.9 if "idiot" in text else 0.1)

    def test_the_substitute_classifier_is_a_recorded_deviation(self, offline, causal):
        inputs, prov = build_inputs(causal, "rtp_toxicity", ["EMT"], root=offline)
        resources = inputs["EMT"]["__protocol__"]["resources"]
        assert any("Perspective" in r.get("deviation", "") for r in resources)
        assert datasets_toxicity._TOXICITY_CLASSIFIER in str(resources)
        assert prov["toxicity_classifier"] == datasets_toxicity._TOXICITY_CLASSIFIER
        assert "Perspective" in prov["deviation"]

    def test_the_dataset_revision_is_pinned_in_the_protocol(self, offline, causal):
        inputs, prov = build_inputs(causal, "rtp_toxicity", ["EMT"], root=offline)
        assert inputs["EMT"]["__protocol__"]["dataset_revision"] == "rev123"
        assert prov["dataset_revision"] == "rev123"
        assert prov["prompts"] == 2

    def test_limit_caps_prompts(self, offline, causal):
        _, prov = build_inputs(causal, "rtp_toxicity", ["EMT"], root=offline, limit=1)
        assert prov["prompts"] == 1
        assert len(causal.calls) == 25

    def test_declared_shape(self):
        spec = DATASETS["rtp_toxicity"]
        assert spec.metrics == ("EMT",)
        assert spec.requires_access == ("completions",)
        assert spec.init_from_backend == ()
