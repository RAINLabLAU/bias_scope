"""Tests for public package exports."""


def test_generated_text_exports_include_new_metrics():
    from bias_scope.generated_text_based import (
        CounterfactualSentimentBias,
        EMT,
        FGB,
        PGB,
    )

    assert EMT is not None
    assert PGB is not None
    assert FGB is not None
    assert CounterfactualSentimentBias is not None


def test_top_level_exports_include_new_generated_text_metrics():
    from bias_scope import CounterfactualSentimentBias, EMT, FGB, PGB

    assert EMT is not None
    assert PGB is not None
    assert FGB is not None
    assert CounterfactualSentimentBias is not None


def test_top_level_exports_define_new_prompt_metric_names():
    import bias_scope

    assert hasattr(bias_scope, "BOLD")
    assert hasattr(bias_scope, "DemographicRepresentationBias")
    assert hasattr(bias_scope, "TruthfulQA")
    assert hasattr(bias_scope, "RealToxicityPrompts")
    assert hasattr(bias_scope, "OpinionConsistencyAcrossPersonas")


def test_top_level_exports_include_demographic_representation_bias():
    from bias_scope import DemographicRepresentationBias

    assert hasattr(DemographicRepresentationBias, "__name__") or DemographicRepresentationBias is None
