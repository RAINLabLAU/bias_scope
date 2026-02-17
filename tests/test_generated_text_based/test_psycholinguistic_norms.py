import pytest

from bias_scope.generated_text_based.psycholinguistic_norms import PsycholinguisticNorms


def test_psycholinguistic_norms_equation():
    """
    Check PN_d = mean over completion-level means C_d(c).
    """
    metric = PsycholinguisticNorms()
    completions = [
        ["kind smart", "lazy thief"],
        ["doctor calm", "nurse kind"],
    ]
    lexicon = {
        "kind": {"valence": 8.0, "arousal": 3.0},
        "smart": {"valence": 7.0, "arousal": 4.0},
        "lazy": {"valence": 3.0, "arousal": 2.0},
        "thief": {"valence": 2.0, "arousal": 6.0},
        "doctor": {"valence": 6.0, "arousal": 4.0},
        "calm": {"valence": 7.0, "arousal": 2.0},
        "nurse": {"valence": 6.0, "arousal": 3.0},
    }

    # Completion means:
    # c1: val=7.5 ar=3.5
    # c2: val=2.5 ar=4.0
    # c3: val=6.5 ar=3.0
    # c4: val=7.0 ar=3.0
    # Global means: val=5.875, ar=3.375
    result = metric.evaluate(completions, lexicon)

    assert result["pn::valence"] == pytest.approx(5.875)
    assert result["pn::arousal"] == pytest.approx(3.375)


def test_psycholinguistic_norms_coverage_details():
    metric = PsycholinguisticNorms()
    completions = [
        ["kind unknownword", "unknownonly"],
    ]
    lexicon = {"kind": {"valence": 8.0, "arousal": 3.0}}

    result = metric.evaluate(completions, lexicon, return_details=True)

    assert result["pn::valence"] == pytest.approx(8.0)
    assert result["pn::arousal"] == pytest.approx(3.0)
    assert result["num_completions"] == 2.0
    assert result["num_scored_completions"] == 1.0
    assert result["completion_coverage_rate"] == pytest.approx(0.5)


def test_psycholinguistic_norms_raises_when_uncovered_not_skipped():
    metric = PsycholinguisticNorms()
    completions = [["unknown only"]]
    lexicon = {"kind": {"valence": 8.0, "arousal": 3.0}}

    with pytest.raises(ValueError):
        metric.evaluate(
            completions,
            lexicon,
            skip_uncovered_completions=False,
        )


def test_psycholinguistic_norms_raises_on_missing_dimension():
    metric = PsycholinguisticNorms()
    completions = [["kind"]]
    lexicon = {"kind": {"valence": 8.0, "arousal": 3.0}}

    with pytest.raises(ValueError):
        metric.evaluate(completions, lexicon, dimensions=["valence", "dominance"])


def test_psycholinguistic_norms_raises_on_lexicon_dimension_mismatch():
    metric = PsycholinguisticNorms()
    completions = [["kind smart"]]
    lexicon = {
        "kind": {"valence": 8.0, "arousal": 3.0},
        "smart": {"valence": 7.0},  # missing arousal
    }

    with pytest.raises(ValueError):
        metric.evaluate(completions, lexicon)


def test_psycholinguistic_norms_category_property():
    metric = PsycholinguisticNorms()
    assert metric.category == "generated_text"
