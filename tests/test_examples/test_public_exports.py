"""Tests for public package exports."""


def test_generated_text_exports_include_new_metrics():
    from bias_scope.generated_text_based import (
        EMT,
        FGB,
        PGB,
        CounterfactualSentimentBias,
    )

    assert EMT is not None
    assert PGB is not None
    assert FGB is not None
    assert CounterfactualSentimentBias is not None


def test_top_level_exports_include_new_generated_text_metrics():
    from bias_scope import EMT, FGB, PGB, CounterfactualSentimentBias

    assert EMT is not None
    assert PGB is not None
    assert FGB is not None
    assert CounterfactualSentimentBias is not None


def test_top_level_exports_define_new_prompt_metric_names():
    import bias_scope

    assert hasattr(bias_scope, "BOLD")
    assert hasattr(bias_scope, "OccupationPronounSkew")
    # Renamed in 0.2.0; the old name stays importable until 0.3.0.
    assert hasattr(bias_scope, "DemographicRepresentationBias")
    assert hasattr(bias_scope, "TruthfulQA")
    assert hasattr(bias_scope, "RealToxicityPrompts")
    assert hasattr(bias_scope, "OpinionConsistencyAcrossPersonas")


def test_top_level_exports_include_demographic_representation_bias():
    from bias_scope import DemographicRepresentationBias, OccupationPronounSkew

    assert hasattr(OccupationPronounSkew, "__name__")
    # The deprecated alias resolves to a subclass of the renamed class, so
    # existing isinstance checks keep working while callers migrate.
    assert issubclass(DemographicRepresentationBias, OccupationPronounSkew)
